// SPDX-License-Identifier: GPL-3.0-or-later
// Clunk — LLVM IR superoptimiser
// Copyright (C) 2025 Clunk contributors
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

#pragma once
/*
 * Clunk Evolutionary Search — population-based optimisation with
 * crossover and mutation. Escapes local optima that stochastic search
 * can get stuck in.
 *
 * Crossover validates children (drops and clones a parent if the child
 * references undefined SSA values). Supports parallel population evaluation
 * via ThreadPool, fitness-sharing diversity preservation, time-budget and
 * target-score early stop, structural-hash score cache, and PatternLibrary
 * integration for population seeding and pattern-guided mutation. Fixed
 * inbreeding by re-drawing p2_idx when equal to p1_idx.
 */
#include <chrono>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <random>
#include "clunk/IR/Function.h"
#include "clunk/Evaluator/EvaluationEngine.h"
#include "clunk/Search/StochasticSearch.h"
#include "clunk/Search/ThreadPool.h"
#include "clunk/Pattern/PatternLibrary.h"

namespace clunk::search {

// ── Evolutionary configuration ──────────────────────────────────────────
struct EvolutionaryConfig {
    size_t population_size = 50;            // Number of individuals
    size_t max_generations = 200;           // Maximum generations
    double crossover_rate = 0.7;            // Probability of crossover
    double mutation_rate = 0.3;             // Probability of mutation
    double elite_fraction = 0.1;            // Fraction preserved as elite
    size_t tournament_size = 3;             // Tournament selection size
    // size_t.
    size_t stagnation_limit = 20;           // Generations without improvement before restart
    unsigned seed = 0;                      // 0 = random seed

    // ── New fields (backward-compatible: all default to disabled) ──────
    // Optional wall-clock budget in seconds . If set, the
    // generational loop checks steady_clock periodically and exits.
    std::optional<double> time_budget_seconds;

    // Optional target score; the loop exits as soon as the best
    // individual's fitness exceeds it .
    std::optional<double> target_score;

    // Parallel population evaluation . Default OFF —
    // the serial path uses the score cache (which the parallel path
    // cannot share without synchronisation), and for typical clunk
    // workloads (small functions, many candidates) the cache hit rate
    // dominates over the parallelism benefit. Enable with --parallel-eval
    // for large functions where cache hit rate is low.
    bool parallel_evaluation = false;

    // Number of worker threads for parallel evaluation. 0 = auto
    // (hardware_concurrency, clamped to [1, 32]).
    size_t num_eval_threads = 0;

    // Diversity preservation via fitness sharing .
    // 0.0 = disabled. When > 0, fitness is scaled by
    //   1 / (1 + sum_{j != i} sh(d(i,j)))
    // where d(i,j) is the Hamming distance of the structural hashes
    // (interpreted as a proxy for structural distance) and sh is a
    // triangular sharing function with radius = diversity_radius.
    double diversity_radius = 0.0;

    // Maximum number of candidates to return from search() .
    // 0 = return the entire final population (legacy behaviour).
    size_t max_results = 0;

    // Enable PatternLibrary integration . When true,
    // the underlying StochasticSearch will use PatternGuided mutations
    // (if a library is set), and the population will be seeded with
    // pattern-optimised variants.
    bool use_pattern_library = true;
};

// ── Individual in the population ────────────────────────────────────────
struct Individual {
    std::shared_ptr<ir::Function> function;
    double fitness;                         // Evaluation engine score
    size_t age;                             // Number of generations survived
    std::string lineage;                    // Description of how it was created

    // Explicit `evaluated` flag replaces the fitness==0.0 sentinel.
    // Fitness can legitimately be 0.0 (or negative) for an evaluated
    // individual, so the old sentinel was unsafe.
    bool evaluated = false;

    // Cached structural hash for dedup / diversity distance.
    uint64_t structural_hash = 0;

    // True iff this individual was derived from the original solely via
    // semantics-preserving mutations (see Candidate::sound). Crossover
    // children are never sound — splicing blocks from two parents gives
    // no equivalence guarantee.
    bool sound = false;

    bool operator<(const Individual& other) const {
        return fitness > other.fitness; // Higher fitness is better
    }
};

// ── Crossover strategies ────────────────────────────────────────────────
enum class CrossoverKind : uint8_t {
    BlockSwap,          // Swap a basic block between parents
    InstructionSwap,    // Swap corresponding instructions
    SegmentSplice,      // Splice instruction sequences at a cut point
    Hybrid              // Combine strategies based on context
};

// ── Evolutionary Search ─────────────────────────────────────────────────
class EvolutionarySearch final {
public:
    // Primary constructor (backward-compatible).
    explicit EvolutionarySearch(const EvolutionaryConfig& config = {},
                                 evaluator::EvaluationEngine* engine = nullptr);

    // Extended constructor: accepts an optional PatternLibrary pointer
    // for the PatternGuided mutation kind . When `lib`
    // is null, PatternGuided is silently skipped.
    EvolutionarySearch(const EvolutionaryConfig& config,
                        evaluator::EvaluationEngine* engine,
                        pattern::PatternLibrary* lib);

    // Run evolutionary search on a function
    // Optionally seed with candidates from stochastic search
    std::vector<Candidate> search(
        const ir::Function& original,
        const std::vector<Candidate>& seed_candidates = {});

    // Configuration
    EvolutionaryConfig& config() { return config_; }
    const EvolutionaryConfig& config() const { return config_; }

    // PatternLibrary accessor .
    void set_pattern_library(pattern::PatternLibrary* lib);
    pattern::PatternLibrary* pattern_library() const { return pattern_lib_; }

    // Statistics
    struct Stats {
        size_t generations_run = 0;
        size_t crossovers_performed = 0;
        size_t crossovers_rejected_invalid = 0;
        size_t mutations_performed = 0;
        double best_fitness = -std::numeric_limits<double>::infinity();
        double avg_fitness = 0.0;
        size_t stagnation_count = 0;
        size_t restarts = 0;
        size_t score_cache_hits = 0;
        size_t parallel_eval_threads = 0;
        double elapsed_seconds = 0.0;
    };
    const Stats& stats() const { return stats_; }

private:
    // ── Selection ───────────────────────────────────────────────────────
    // Returns the index of the selected individual in the population
    // (avoids copying an Individual — which contains a
    // shared_ptr<Function> and would trigger atomic refcount ops on
    // every tournament, ~54k per run).
    size_t tournament_select(const std::vector<Individual>& population);

    // ── Crossover ───────────────────────────────────────────────────────
    std::pair<Individual, Individual> crossover(
        const Individual& parent1, const Individual& parent2);

    Individual block_swap_crossover(const Individual& p1, const Individual& p2);
    Individual segment_splice_crossover(const Individual& p1, const Individual& p2);

    // ── Mutation ────────────────────────────────────────────────────────
    Individual mutate(const Individual& individual);

    // ── Population management ───────────────────────────────────────────
    std::vector<Individual> initialise_population(
        const ir::Function& original,
        const std::vector<Candidate>& seeds);
    void evaluate_population(std::vector<Individual>& population);
    void evaluate_population_serial(std::vector<Individual>& population);
    void evaluate_population_parallel(std::vector<Individual>& population);
    void apply_fitness_sharing(std::vector<Individual>& population);
    std::vector<Individual> select_elite(const std::vector<Individual>& population);
    bool check_convergence(const std::vector<Individual>& population);

    // Score an individual's function via score_candidate_with_cached_orig,
    // using the structural-hash cache.
    double score_individual(const ir::Function& fn);
    // Compute structural hash, using StochasticSearch::structural_hash for
    // consistency.
    static uint64_t hash_function(const ir::Function& fn);

    // Time-budget check .
    bool time_budget_exceeded() const;

    EvolutionaryConfig config_;
    evaluator::EvaluationEngine* engine_;
    pattern::PatternLibrary* pattern_lib_ = nullptr;
    std::mt19937 rng_;
    Stats stats_;

    // Cached original-function analysis for relative scoring.
    const ir::Function* original_ = nullptr;
    evaluator::FunctionAnalysis cached_original_analysis_;
    bool original_analysis_cached_ = false;

    // Structural-hash score cache.
    std::unordered_map<uint64_t, double> score_cache_;

    // Thread pool for parallel evaluation. Lazily created on
    // first use; size determined by config_.num_eval_threads.
    std::unique_ptr<ThreadPool> pool_;

    // Reused StochasticSearch instance. Avoids constructing
    // a fresh StochasticSearch (~18k per evolutionary run) on every
    // mutate() / initialise_population() call. Re-seeded via reset()
    // before each use to preserve per-call determinism.
    StochasticSearch stochastic_helper_;

    // Start time for time-budget adherence .
    std::chrono::steady_clock::time_point start_time_;
};

} // namespace clunk::search
