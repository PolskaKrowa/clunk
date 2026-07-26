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

// Clunk Evolutionary Search — population-based optimisation with
// crossover and mutation. Escapes local optima that stochastic search
// can get stuck in.

#include "clunk/Search/EvolutionarySearch.h"
#include "clunk/Search/StochasticSearch.h"
#include "clunk/IR/Clone.h"

#include <iostream>

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_map>
#include <unordered_set>

namespace clunk::search {

// validate_function lives in clunk::ir (clunk/IR/Clone.h) and is reused
// here for crossover validation.
using ir::validate_function;

// Lineage strings are a debugging aid, but crossover concatenates BOTH
// parents' lineages into the child's ("block_swap(p1, p2)"), doubling the
// string length every generation — exponential memory growth over a long
// run. Clip to a fixed prefix so lineage stays O(1) per individual.
static std::string clip_lineage(std::string s) {
    constexpr size_t MAX_LINEAGE = 64;
    if (s.size() > MAX_LINEAGE) {
        s.resize(MAX_LINEAGE);
        s += "...";
    }
    return s;
}

// ── Constructor ─────────────────────────────────────────────────────────────

EvolutionarySearch::EvolutionarySearch(const EvolutionaryConfig& config,
                                        evaluator::EvaluationEngine* engine)
    : EvolutionarySearch(config, engine, nullptr) {}

EvolutionarySearch::EvolutionarySearch(const EvolutionaryConfig& config,
                                        evaluator::EvaluationEngine* engine,
                                        pattern::PatternLibrary* lib)
    : config_(config), engine_(engine), pattern_lib_(lib),
      stochastic_helper_({}, engine, lib) {
    if (config_.seed == 0) {
        std::random_device rd;
        rng_.seed(rd());
    } else {
        rng_.seed(config_.seed);
    }
    stats_ = {};
}

// ── PatternLibrary setter ──────────────────────────────────────────────────────

void EvolutionarySearch::set_pattern_library(pattern::PatternLibrary* lib) {
    pattern_lib_ = lib;
    stochastic_helper_.set_pattern_library(lib);
}

// ── Structural hash / score caching ───────────────────────────────────────────────

uint64_t EvolutionarySearch::hash_function(const ir::Function& fn) {
    return StochasticSearch::structural_hash(fn);
}

double EvolutionarySearch::score_individual(const ir::Function& fn) {
    if (!engine_) return 0.0;
    uint64_t h = hash_function(fn);
    auto it = score_cache_.find(h);
    if (it != score_cache_.end()) {
        stats_.score_cache_hits++;
        return it->second;
    }
    double s = engine_->score_candidate_with_cached_orig(cached_original_analysis_, fn);
    score_cache_[h] = s;
    return s;
}

// ── Time-budget check ───────────────────────────────────────────────────────────

bool EvolutionarySearch::time_budget_exceeded() const {
    if (!config_.time_budget_seconds.has_value()) return false;
    auto now = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = now - start_time_;
    return elapsed.count() >= *config_.time_budget_seconds;
}

// ── Main search loop ───────────────────────────────────────────────────────

std::vector<Candidate> EvolutionarySearch::search(
    const ir::Function& original,
    const std::vector<Candidate>& seed_candidates) {

    // Cache the original-function analysis so we can score candidates
    // relative to it. Cleared on every search() call.
    original_ = &original;
    original_analysis_cached_ = false;
    score_cache_.clear();
    if (engine_) {
        cached_original_analysis_ = engine_->analyse(original);
        original_analysis_cached_ = true;
    }

    // Start the wall-clock timer.
    start_time_ = std::chrono::steady_clock::now();

    // Initialise population
    auto population = initialise_population(original, seed_candidates);
    evaluate_population(population);

    // Track best fitness for convergence detection
    double best_fitness_ever = -std::numeric_limits<double>::infinity();
    size_t stagnation_counter = 0;

    for (size_t gen = 0; gen < config_.max_generations; ++gen) {
        stats_.generations_run = gen + 1;

        if (time_budget_exceeded()) {
            break;
        }

        // Target-score early stop.
        if (config_.target_score.has_value() &&
            best_fitness_ever >= *config_.target_score) {
            break;
        }

        // Sort population by fitness (descending, because operator< uses >)
        std::sort(population.begin(), population.end());

        // Track best fitness
        double current_best = population.empty() ? -std::numeric_limits<double>::infinity()
                                                  : population.front().fitness;
        if (current_best > best_fitness_ever) {
            best_fitness_ever = current_best;
            stagnation_counter = 0;
        } else {
            stagnation_counter++;
        }

        stats_.best_fitness = best_fitness_ever;

        // Compute average fitness
        if (!population.empty()) {
            double sum = 0.0;
            for (auto& ind : population) sum += ind.fitness;
            stats_.avg_fitness = sum / static_cast<double>(population.size());
        }

        // Apply fitness sharing. Modifies fitness in
        // place; population is sorted afterwards by select_elite etc.
        if (config_.diversity_radius > 0.0) {
            apply_fitness_sharing(population);
        }

        // Check convergence / stagnation
        if (check_convergence(population)) {
            stats_.stagnation_count = stagnation_counter;
            // Restart: re-initialise part of the population
            stats_.restarts++;

            // Keep the elite and re-initialise the rest
            auto elite = select_elite(population);
            auto new_blood = initialise_population(original, {});
            evaluate_population(new_blood);

            // Merge elite and new blood
            population = elite;
            size_t remaining = config_.population_size - elite.size();
            for (size_t i = 0; i < remaining && i < new_blood.size(); ++i) {
                population.push_back(new_blood[i]);
            }
            // Fill up to population size with mutated originals
            while (population.size() < config_.population_size) {
                Individual ind;
                ind.function = ir::deep_copy_function(original);
                ind.fitness = 0.0;
                ind.age = 0;
                ind.sound = true;  // clone of the original
                ind.lineage = "restart";
                ind.evaluated = false;
                // Apply random mutation
                auto mutated = mutate(ind);
                population.push_back(mutated);
            }

            stagnation_counter = 0;
            continue;
        }

        stats_.stagnation_count = stagnation_counter;

        // ── Create offspring ──────────────────────────────────────────────
        std::vector<Individual> offspring;

        // Select elite (preserved unchanged)
        auto elite = select_elite(population);

        // Fill rest of next generation with offspring
        size_t target_offspring = config_.population_size - elite.size();

        // Individual size cap (hoisted out of the offspring loop — the
        // original's size is invariant for the whole search).
        // Cap: 2x the original function size + 50, max 200. This allows
        // growth for legitimate optimisations (e.g. loop unrolling) while
        // preventing runaway bloat: without it, mutations that add
        // instructions cause exponential growth across generations.
        auto count_insts = [](const Individual& ind) -> size_t {
            if (!ind.function) return 0;
            size_t n = 0;
            for (auto& bb : ind.function->blocks()) {
                if (bb) n += bb->size();
            }
            return n;
        };
        size_t orig_size = 0;
        for (auto& bb : original.blocks()) {
            if (bb) orig_size += bb->size();
        }
        const size_t MAX_INDIVIDUAL_SIZE =
            std::min(size_t(200), orig_size * 2 + 50);

        std::uniform_real_distribution<double> prob_dist(0.0, 1.0);

        while (offspring.size() < target_offspring) {
            // Select two parents via tournament selection:
            // returns an index, avoiding a shared_ptr atomic refcount copy).
            size_t p1_idx = tournament_select(population);
            size_t p2_idx = tournament_select(population);

            // Re-draw p2_idx if equal to p1_idx (up to a
            // small number of retries) to avoid inbreeding.
            if (population.size() > 1) {
                size_t retries = 0;
                while (p2_idx == p1_idx && retries < 4) {
                    p2_idx = tournament_select(population);
                    ++retries;
                }
                // If we still have p1_idx == p2_idx after retries, give up
                // and just increment p2_idx mod size — crossover of an
                // individual with itself is harmless (produces a clone).
                if (p2_idx == p1_idx) {
                    p2_idx = (p1_idx + 1) % population.size();
                }
            }

            const Individual& parent1 = population[p1_idx];
            const Individual& parent2 = population[p2_idx];

            Individual child1, child2;

            // Apply crossover with probability crossover_rate
            if (prob_dist(rng_) < config_.crossover_rate) {
                auto [c1, c2] = crossover(parent1, parent2);
                child1 = std::move(c1);
                child2 = std::move(c2);
                stats_.crossovers_performed++;
            } else {
                child1 = parent1;
                child2 = parent2;
                // Children are not yet "evaluated" in their new context
                // (their fitness will be re-evaluated). Mark them so the
                // evaluator knows to re-score.
                child1.evaluated = false;
                child2.evaluated = false;
            }

            // Apply mutation with probability mutation_rate
            if (prob_dist(rng_) < config_.mutation_rate) {
                child1 = mutate(child1);
                stats_.mutations_performed++;
            }
            if (prob_dist(rng_) < config_.mutation_rate) {
                child2 = mutate(child2);
                stats_.mutations_performed++;
            }

            // Age the children
            child1.age = 0;
            child2.age = 0;

            // Cap individual size: if a child has grown too large (e.g.
            // via pattern application or crossover that added blocks),
            // skip it.
            if (count_insts(child1) <= MAX_INDIVIDUAL_SIZE) {
                offspring.push_back(std::move(child1));
            }
            if (offspring.size() < target_offspring &&
                count_insts(child2) <= MAX_INDIVIDUAL_SIZE) {
                offspring.push_back(std::move(child2));
            }

            // Time-budget check inside the offspring loop — if a single
            // generation is taking too long (e.g. due to large individuals),
            // bail out early with what we have.
            if (time_budget_exceeded()) break;
        }

        // Evaluate offspring
        evaluate_population(offspring);

        // Age the elite
        for (auto& ind : elite) {
            ind.age++;
        }

        // Combine elite + offspring for next generation
        population.clear();
        population.reserve(config_.population_size);
        for (auto& ind : elite) {
            population.push_back(std::move(ind));
        }
        for (auto& ind : offspring) {
            if (population.size() < config_.population_size) {
                population.push_back(std::move(ind));
            }
        }
    }

    // ── Build result: top candidates from the final population ─────────────
    std::sort(population.begin(), population.end());

    // Dedup by structural hash. The final population often
    // contains near-duplicates (crossover of identical parents, mutated
    // copies that mutated back, etc.).
    std::vector<Candidate> results;
    std::unordered_set<uint64_t> seen_hashes;
    for (auto& ind : population) {
        if (!ind.function) continue;
        uint64_t h = ind.structural_hash;
        if (h == 0) h = hash_function(*ind.function);
        if (seen_hashes.count(h)) continue;
        seen_hashes.insert(h);

        Candidate cand;
        cand.function = ind.function;
        cand.score = ind.fitness;
        cand.iteration_found = stats_.generations_run;
        cand.description = ind.lineage;
        cand.structural_hash = h;
        cand.sound = ind.sound;
        results.push_back(cand);

        if (config_.max_results > 0 && results.size() >= config_.max_results) {
            break;
        }
    }

    // Record total elapsed time.
    auto end_time = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time_;
    stats_.elapsed_seconds = elapsed.count();

    return results;
}

// ── Tournament selection ───────────────────────────────────────────────────

size_t EvolutionarySearch::tournament_select(
    const std::vector<Individual>& population) {
    // Precondition: population is non-empty. Callers (the main generation
    // loop and the restart path) only invoke this after
    // initialise_population, which always returns at least one individual.
    // The empty-population branch is defensive — returning 0 from an empty
    // population would be UB, but it cannot occur in practice.
    if (population.empty()) return 0;

    std::uniform_int_distribution<size_t> dist(0, population.size() - 1);

    // Pick tournament_size random individuals — return the index of the best.
    size_t best_idx = dist(rng_);
    for (size_t i = 1; i < config_.tournament_size; ++i) {
        size_t challenger_idx = dist(rng_);
        // operator< uses > for fitness, so the "less" individual has higher fitness
        if (population[challenger_idx] < population[best_idx]) {
            best_idx = challenger_idx;
        }
    }

    return best_idx;
}

// ── Crossover ──────────────────────────────────────────────────────────────

std::pair<Individual, Individual> EvolutionarySearch::crossover(
    const Individual& parent1, const Individual& parent2) {
    // Choose a crossover strategy
    std::uniform_int_distribution<int> strategy_dist(0, 1);
    int strategy = strategy_dist(rng_);

    if (strategy == 0) {
        // Block swap crossover
        auto child1 = block_swap_crossover(parent1, parent2);
        auto child2 = block_swap_crossover(parent2, parent1);
        return {child1, child2};
    } else {
        // Segment splice crossover
        auto child1 = segment_splice_crossover(parent1, parent2);
        auto child2 = segment_splice_crossover(parent2, parent1);
        return {child1, child2};
    }
}

Individual EvolutionarySearch::block_swap_crossover(
    const Individual& p1, const Individual& p2) {
    Individual child;
    child.age = 0;
    child.lineage = clip_lineage("block_swap(" + p1.lineage + ", " + p2.lineage + ")");
    child.evaluated = false;

    if (!p1.function || !p2.function) {
        child.function = p1.function ? ir::deep_copy_function(*p1.function) : nullptr;
        child.fitness = p1.fitness;
        child.evaluated = p1.evaluated;
        child.sound = p1.sound;
        return child;
    }

    // Start with a copy of p1
    child.function = ir::deep_copy_function(*p1.function);

    // Pick a random block from p2 and swap it into child
    if (p2.function->blocks().empty() || child.function->blocks().empty()) {
        child.fitness = p1.fitness;
        child.evaluated = p1.evaluated;
        child.sound = p1.sound;
        return child;
    }

    std::uniform_int_distribution<size_t> block_dist(
        0, p2.function->blocks().size() - 1);
    size_t swap_idx = block_dist(rng_);

    auto& src_block = p2.function->blocks()[swap_idx];
    if (!src_block) {
        child.fitness = p1.fitness;
        child.evaluated = p1.evaluated;
        child.sound = p1.sound;
        return child;
    }
    const std::string& block_name = src_block->name();

    // Find or create the block in child
    auto child_block = child.function->block(block_name);
    if (child_block) {
        // Replace the block's instructions with p2's version. (No undo
        // vector needed — validation failure below discards the whole
        // child and re-clones p1.)
        child_block->instructions().clear();
        for (auto& inst : src_block->instructions()) {
            if (!inst) continue;
            auto new_inst = std::make_shared<ir::Instruction>(
                inst->opcode(), inst->type(), inst->name());
            for (auto& op : inst->operands()) {
                new_inst->add_operand(op);
            }
            for (auto& [k, v] : inst->metadata()) {
                new_inst->set_metadata(k, v);
            }
            new_inst->binop_flags() = inst->binop_flags();
            if (inst->alignment()) new_inst->set_alignment(inst->alignment().value());
            new_inst->set_volatile(inst->is_volatile());
            child_block->add_instruction(new_inst);
        }

        // Validate the child. If it references SSA values
        // that don't exist in its scope, discard it and return a clone of p1.
        if (!validate_function(*child.function)) {
            stats_.crossovers_rejected_invalid++;
            child.function = ir::deep_copy_function(*p1.function);
            child.fitness = p1.fitness;
            child.evaluated = p1.evaluated;
            child.sound = p1.sound;
        child.sound = p1.sound;
            child.lineage = clip_lineage("block_swap_invalid_fallback(" + p1.lineage + ")");
            return child;
        }
    }
    // If the block doesn't exist in child, we skip the swap (conservative)

    child.fitness = 0.0; // Will be evaluated later
    return child;
}

Individual EvolutionarySearch::segment_splice_crossover(
    const Individual& p1, const Individual& p2) {
    Individual child;
    child.age = 0;
    child.lineage = clip_lineage("segment_splice(" + p1.lineage + ", " + p2.lineage + ")");
    child.evaluated = false;

    if (!p1.function || !p2.function) {
        child.function = p1.function ? ir::deep_copy_function(*p1.function) : nullptr;
        child.fitness = p1.fitness;
        child.evaluated = p1.evaluated;
        child.sound = p1.sound;
        return child;
    }

    // For simplicity, we work on the entry block (the most common case)
    // Splice: take the first N instructions from p1, then the rest from p2
    child.function = ir::deep_copy_function(*p1.function);

    // Find the first block in both parents
    auto p1_entry = p1.function->entry_block();
    auto p2_entry = p2.function->entry_block();

    if (!p1_entry || !p2_entry) {
        child.fitness = p1.fitness;
        child.evaluated = p1.evaluated;
        child.sound = p1.sound;
        return child;
    }

    auto child_entry = child.function->entry_block();
    if (!child_entry) {
        child.fitness = p1.fitness;
        child.evaluated = p1.evaluated;
        child.sound = p1.sound;
        return child;
    }

    // Pick a cut point
    size_t p1_size = p1_entry->size();
    size_t p2_size = p2_entry->size();
    if (p1_size == 0 || p2_size == 0) {
        child.fitness = p1.fitness;
        child.evaluated = p1.evaluated;
        child.sound = p1.sound;
        return child;
    }

    // Cut after some fraction of p1's instructions (but before the terminator)
    size_t max_cut = p1_size > 0 ? p1_size - 1 : 0; // Don't cut past the terminator
    std::uniform_int_distribution<size_t> cut_dist(1, std::max(size_t(1), max_cut));
    size_t cut_point = cut_dist(rng_);

    // Build a new instruction list:
    // Instructions [0, cut_point) from p1 (child already has these)
    // Instructions [cut_point, end) from p2 (but skip p2's terminator, keep p1's)
    child_entry->instructions().clear();

    // First part: from p1
    for (size_t i = 0; i < cut_point && i < p1_entry->size(); ++i) {
        auto inst = p1_entry->instruction(i);
        if (!inst) continue;
        auto new_inst = std::make_shared<ir::Instruction>(
            inst->opcode(), inst->type(), inst->name());
        for (auto& op : inst->operands()) {
            if (op) new_inst->add_operand(op);
        }
        for (auto& [k, v] : inst->metadata()) new_inst->set_metadata(k, v);
        new_inst->binop_flags() = inst->binop_flags();
        if (inst->alignment()) new_inst->set_alignment(inst->alignment().value());
        new_inst->set_volatile(inst->is_volatile());
        child_entry->add_instruction(new_inst);
    }

    // Second part: from p2 (skip terminators, we'll keep p1's terminator at the end)
    for (size_t i = cut_point; i < p2_entry->size(); ++i) {
        auto inst = p2_entry->instruction(i);
        if (!inst) continue;
        if (inst->is_terminator()) continue; // Skip p2's terminator
        auto new_inst = std::make_shared<ir::Instruction>(
            inst->opcode(), inst->type(), inst->name() + "_spliced");
        for (auto& op : inst->operands()) {
            if (op) new_inst->add_operand(op);
        }
        for (auto& [k, v] : inst->metadata()) new_inst->set_metadata(k, v);
        new_inst->binop_flags() = inst->binop_flags();
        if (inst->alignment()) new_inst->set_alignment(inst->alignment().value());
        new_inst->set_volatile(inst->is_volatile());
        child_entry->add_instruction(new_inst);
    }

    // Add p1's terminator back
    auto p1_term = p1_entry->terminator();
    if (p1_term) {
        auto new_term = std::make_shared<ir::Instruction>(
            p1_term->opcode(), p1_term->type(), p1_term->name());
        for (auto& op : p1_term->operands()) {
            if (op) new_term->add_operand(op);
        }
        for (auto& [k, v] : p1_term->metadata()) new_term->set_metadata(k, v);
        new_term->binop_flags() = p1_term->binop_flags();
        if (p1_term->alignment()) new_term->set_alignment(p1_term->alignment().value());
        new_term->set_volatile(p1_term->is_volatile());
        child_entry->add_instruction(new_term);
    }

    // Validate the child. Segment splice is the most
    // dangerous crossover (it mixes instructions from unrelated functions
    // in the same block), so this is where validation matters most.
    if (!validate_function(*child.function)) {
        stats_.crossovers_rejected_invalid++;
        child.function = ir::deep_copy_function(*p1.function);
        child.fitness = p1.fitness;
        child.evaluated = p1.evaluated;
        child.sound = p1.sound;
        child.lineage = clip_lineage("segment_splice_invalid_fallback(" + p1.lineage + ")");
        return child;
    }

    child.fitness = 0.0; // Will be evaluated later
    return child;
}

// ── Mutation ───────────────────────────────────────────────────────────────

Individual EvolutionarySearch::mutate(const Individual& individual) {
    Individual mutant;
    mutant.age = individual.age;
    mutant.lineage = clip_lineage(individual.lineage + "+mut");
    mutant.evaluated = false;

    if (!individual.function) {
        mutant.function = nullptr;
        mutant.fitness = individual.fitness;
        mutant.evaluated = individual.evaluated;
        mutant.sound = individual.sound;
        return mutant;
    }

    // Reuse the StochasticSearch instance. Re-seed its RNG with
    // a fresh draw from our own RNG to preserve per-call determinism
    // (equivalent to the previous `StochasticSearch(sconfig, engine_)`
    // construction, but without the per-call allocation).
    stochastic_helper_.reset(static_cast<unsigned>(rng_()));

    // Keep the pattern library pointer in sync.
    if (pattern_lib_ && config_.use_pattern_library) {
        stochastic_helper_.set_pattern_library(pattern_lib_);
        stochastic_helper_.config().use_pattern_library = true;
    } else {
        stochastic_helper_.set_pattern_library(nullptr);
        stochastic_helper_.config().use_pattern_library = false;
    }

    auto mut = stochastic_helper_.random_mutation(*individual.function);
    auto mutated_fn = stochastic_helper_.apply_mutation(*individual.function, mut);

    if (mutated_fn) {
        mutant.function = mutated_fn;
        mutant.fitness = 0.0; // Will be evaluated later
        mutant.sound = individual.sound && StochasticSearch::is_sound_kind(mut.kind);
        // Lazy description: only build the string when the
        // mutation was actually applied.
        mutant.lineage += "[" + StochasticSearch::describe_mutation(mut) + "]";
    } else {
        // Mutation was not applicable — keep original
        mutant.function = ir::deep_copy_function(*individual.function);
        mutant.fitness = individual.fitness;
        mutant.evaluated = individual.evaluated;
        mutant.sound = individual.sound;
    }

    return mutant;
}

// ── Population management ──────────────────────────────────────────────────

std::vector<Individual> EvolutionarySearch::initialise_population(
    const ir::Function& original,
    const std::vector<Candidate>& seeds) {

    std::vector<Individual> population;

    // Add the original function
    {
        Individual ind;
        ind.function = ir::deep_copy_function(original);
        ind.fitness = 0.0;
        ind.age = 0;
        ind.lineage = "original";
        ind.evaluated = false;
        ind.sound = true;
        population.push_back(ind);
    }

    // Add seed candidates
    for (auto& seed : seeds) {
        if (seed.function) {
            Individual ind;
            ind.function = ir::deep_copy_function(*seed.function);
            ind.fitness = seed.score;
            ind.age = 0;
            ind.lineage = "seed";
            ind.evaluated = true;  // we trust the seed's score
            ind.sound = seed.sound;
            population.push_back(ind);
        }
    }

    // If a pattern library is available, seed the
    // population with pattern-optimised variants of the original.
    if (pattern_lib_ && config_.use_pattern_library && pattern_lib_->size() > 0) {
        pattern::ArchDescriptor arch;  // generic
        auto matches = pattern_lib_->match(original, arch);
        for (auto& m : matches) {
            if (population.size() >= config_.population_size) break;
            auto optimised = pattern_lib_->apply(original, m, arch);
            if (optimised && validate_function(*optimised)) {
                Individual ind;
                ind.function = optimised;
                ind.fitness = 0.0;
                ind.age = 0;
                ind.lineage = "pattern:" + m.pattern_id;
                ind.evaluated = false;
                ind.sound = false;  // library rewrites carry no proof
                population.push_back(ind);
                pattern_lib_->record_application(m.pattern_id);
            }
        }
    }

    // Fill the rest with random mutations of the original.
    // Reuse the StochasticSearch instance.
    stochastic_helper_.reset(static_cast<unsigned>(rng_()));
    if (pattern_lib_ && config_.use_pattern_library) {
        stochastic_helper_.set_pattern_library(pattern_lib_);
    }

    while (population.size() < config_.population_size) {
        Individual ind;
        ind.function = ir::deep_copy_function(original);
        ind.fitness = 0.0;
        ind.age = 0;
        ind.evaluated = false;
        ind.sound = true;  // clone of the original; degraded per mutation below

        // Apply 1-3 random mutations
        std::uniform_int_distribution<int> num_mutations(1, 3);
        int n_mut = num_mutations(rng_);

        for (int m = 0; m < n_mut; ++m) {
            if (!ind.function) break;
            auto mut = stochastic_helper_.random_mutation(*ind.function);
            auto mutated = stochastic_helper_.apply_mutation(*ind.function, mut);
            if (mutated) {
                ind.function = mutated;
                ind.sound = ind.sound && StochasticSearch::is_sound_kind(mut.kind);
            }
        }

        ind.lineage = "init_mutated";
        population.push_back(std::move(ind));
    }

    // Trim to population size
    if (population.size() > config_.population_size) {
        population.resize(config_.population_size);
    }

    return population;
}

// ── Serial / parallel evaluation ───────────────────────────────────────────

void EvolutionarySearch::evaluate_population_serial(std::vector<Individual>& population) {
    if (!engine_) return;
    for (auto& ind : population) {
        if (ind.function && !ind.evaluated) {
            // Time-budget check: abort evaluation if we've run out of time.
            // The unevaluated individuals keep fitness=0.0 and will be
            // ranked low by select_elite, effectively discarding them.
            if (time_budget_exceeded()) {
                ind.fitness = 0.0;
                ind.evaluated = true;
                continue;
            }
            ind.fitness = score_individual(*ind.function);
            ind.structural_hash = hash_function(*ind.function);
            ind.evaluated = true;
        }
    }
}

void EvolutionarySearch::evaluate_population_parallel(std::vector<Individual>& population) {
    if (!engine_) return;

    // Lazily create the thread pool on first parallel evaluation.
    if (!pool_) {
        size_t n = config_.num_eval_threads;
        if (n == 0) {
            n = std::thread::hardware_concurrency();
            if (n == 0) n = 1;
            if (n > 32) n = 32;
        }
        pool_ = std::make_unique<ThreadPool>(n);
        stats_.parallel_eval_threads = pool_->worker_count();
    }

    // Submit one task per unevaluated individual. Each task computes
    // the individual's fitness and structural hash, then writes them
    // back to the individual. The engine is assumed to be thread-safe
    // (the evaluator agent is adding a thread-safe cache; analyse()
    // itself only reads the function and writes to a local
    // FunctionAnalysis, so concurrent calls are safe).
    //
    // We do NOT touch score_cache_ from worker threads (it's a shared
    // unordered_map without synchronisation). Instead, each worker
    // computes the score from scratch. The serial path uses the cache;
    // the parallel path trades caching for parallelism. (A future
    // improvement would be a thread-safe cache.)
    std::vector<std::future<std::pair<double, uint64_t>>> futs;
    std::vector<Individual*> targets;  // parallel to futs; index i writes to targets[i]

    for (auto& ind : population) {
        if (ind.function && !ind.evaluated) {
            // Capture the shared_ptr BY VALUE, not a raw Function*.
            //
            // A raw pointer capture is unsafe when futures can be
            // abandoned (e.g. on time-budget-exceeded early return):
            // the abandoned tasks may still be running on worker threads
            // while the population is cleared/replaced, freeing the
            // Function and leaving the task with a dangling pointer.
            //
            // A shared_ptr capture keeps the Function alive for as long
            // as the (possibly abandoned) task holds it, so it's safe to
            // drop a future without waiting on it.
            std::shared_ptr<const ir::Function> fn = ind.function;
            auto fut = pool_->submit([this, fn]() -> std::pair<double, uint64_t> {
                double s = engine_->score_candidate_with_cached_orig(
                    cached_original_analysis_, *fn);
                uint64_t h = StochasticSearch::structural_hash(*fn);
                return {s, h};
            });
            futs.push_back(std::move(fut));
            targets.push_back(&ind);
        }
    }

    // Wait for all tasks and write back results. If the time budget is
    // exceeded, we stop waiting and leave remaining individuals
    // unevaluated (fitness=0.0, ranked low by select_elite). Any
    // still-running tasks keep their own shared_ptr<Function> alive (see
    // above), so this is now safe even though we never call .get() on
    // their futures.
    for (size_t i = 0; i < futs.size(); ++i) {
        if (time_budget_exceeded()) {
            // Mark remaining as evaluated with fitness 0 so they don't
            // block future generations.
            for (size_t j = i; j < futs.size(); ++j) {
                targets[j]->fitness = 0.0;
                targets[j]->evaluated = true;
            }
            return;
        }
        auto [s, h] = futs[i].get();
        targets[i]->fitness = s;
        targets[i]->structural_hash = h;
        targets[i]->evaluated = true;
        // Populate the score cache for future serial reads.
        score_cache_[h] = s;
    }
}

void EvolutionarySearch::evaluate_population(std::vector<Individual>& population) {
    if (!engine_) return;
    if (config_.parallel_evaluation) {
        evaluate_population_parallel(population);
    } else {
        evaluate_population_serial(population);
    }
}

// ── Fitness sharing ──────────────────────────────────────────────────────────────

void EvolutionarySearch::apply_fitness_sharing(std::vector<Individual>& population) {
    if (config_.diversity_radius <= 0.0 || population.size() < 2) return;

    // Distance proxy: Hamming distance of the 64-bit structural hashes.
    // (A full structural edit distance would be more accurate but
    // expensive; the hash Hamming distance is a reasonable cheap proxy
    // for "how different are these two functions structurally".)
    //
    // Triangular sharing function:
    //   sh(d) = 1 - d / radius   if d < radius, else 0
    //
    // Scaled fitness:
    //   f'(i) = f(i) / (1 + sum_{j != i} sh(d(i,j)))

    auto sharing_fn = [radius = config_.diversity_radius](uint64_t h1, uint64_t h2) -> double {
        if (h1 == h2) return 1.0;  // identical → maximum sharing
        uint64_t x = h1 ^ h2;
        // Hamming distance of a 64-bit value.
        int dist = __builtin_popcountll(x);
        double d = static_cast<double>(dist);
        if (d >= radius) return 0.0;
        return 1.0 - d / radius;
    };

    // Compute raw distances once.
    std::vector<double> denom(population.size(), 1.0);
    for (size_t i = 0; i < population.size(); ++i) {
        uint64_t hi = population[i].structural_hash;
        if (hi == 0 && population[i].function) {
            hi = hash_function(*population[i].function);
            population[i].structural_hash = hi;
        }
        for (size_t j = i + 1; j < population.size(); ++j) {
            uint64_t hj = population[j].structural_hash;
            if (hj == 0 && population[j].function) {
                hj = hash_function(*population[j].function);
                population[j].structural_hash = hj;
            }
            double sh = sharing_fn(hi, hj);
            denom[i] += sh;
            denom[j] += sh;
        }
    }

    // Scale fitness. We preserve sign (worse individuals stay worse).
    for (size_t i = 0; i < population.size(); ++i) {
        if (denom[i] > 0.0) {
            population[i].fitness = population[i].fitness / denom[i];
        }
    }
}

// ── Elite selection / convergence ───────────────────────────────────────────

std::vector<Individual> EvolutionarySearch::select_elite(
    const std::vector<Individual>& population) {
    // The population should already be sorted by fitness (descending)
    size_t elite_count = static_cast<size_t>(
        static_cast<double>(population.size()) * config_.elite_fraction);
    elite_count = std::max(elite_count, size_t(1)); // Keep at least 1

    std::vector<Individual> elite;
    for (size_t i = 0; i < elite_count && i < population.size(); ++i) {
        elite.push_back(population[i]);
    }
    return elite;
}

bool EvolutionarySearch::check_convergence(
    const std::vector<Individual>& population) {
    // If we've been stagnant for too long, we should restart.
    // stagnation_limit is size_t to avoid truncation.
    (void)population;  // reserved for future diversity-based triggers
    return stats_.stagnation_count >= config_.stagnation_limit;
}

} // namespace clunk::search
