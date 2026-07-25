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
 * Clunk Stochastic Search — explores the optimisation space randomly.
 * Quickly finds good candidate instruction sequences.
 */
#include <chrono>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <random>
#include <functional>
#include "clunk/IR/Function.h"
#include "clunk/Evaluator/EvaluationEngine.h"
#include "clunk/Pattern/PatternLibrary.h"  // for ArchDescriptor

namespace clunk::search {

// RAII transactional guard for in-place IR mutation (see MutationScope.h).
// Forward-declared here so the hot-loop helpers can take it by reference
// without pulling the full header into every translation unit.
class MutationScope;

// ── Search configuration ────────────────────────────────────────────────
struct StochasticConfig {
    size_t max_iterations = 10000;         // Total random attempts
    size_t max_candidates = 100;           // Max candidates to collect
    double temperature = 1.0;              // Exploration temperature
    double temperature_decay = 0.9995;     // Per-iteration decay
    double min_temperature = 0.01;         // Stop cooling below this
    size_t max_instruction_count = 1024;   // Max instructions in candidate
    unsigned seed = 0;                     // 0 = random seed

    // ── New fields (backward-compatible: all default to disabled) ──────
    // Optional wall-clock budget in seconds. If set, the inner loop
    // checks std::chrono::steady_clock periodically and exits gracefully
    // when the budget is exhausted.
    std::optional<double> time_budget_seconds;

    // Optional target score; the loop exits as soon as a candidate
    // exceeds it.
    std::optional<double> target_score;

    // Stagnation limit: if no improvement in `best_score` for this many
    // iterations, re-seed from a fresh clone of the original.
    // 0 = disabled.
    size_t stagnation_limit = 0;

    // Enable the PatternGuided mutation kind. Requires a PatternLibrary
    // pointer to be passed to the constructor; if no library is set,
    // PatternGuided is silently skipped even if this flag is true.
    bool use_pattern_library = true;

    // Enable in-place mutation via MutationScope. When
    // true (default), the inner loop applies mutations in place and
    // undoes them on rejection, eliminating the deep-copy churn. When
    // false, the legacy deep-copy path is used. The deep-copy fallback
    // is always used for mutations that the scope cannot undo (none
    // currently, but reserved for future crossover-in-SA use).
    bool use_in_place_mutation = true;

    // Target architecture descriptor for pattern matching.
    // Defaults to a generic CPU arch. Set this to the actual target so
    // PatternLibrary::match filters patterns by compatibility.
    pattern::ArchDescriptor target_arch{};

    // ── STOKE-style stochastic search knobs ──────────────────────────────
    //
    // `allow_unsound_mutations`: when true, the search may propose
    // the five new STOKE-style mutation kinds (OpcodeReplace,
    // OperandReplace, OperandSwap, InstructionInsert, InstructionReplace)
    // in addition to the existing sound-by-construction rewrites. These
    // new kinds are NOT semantics-preserving — candidates they produce
    // carry Candidate::sound == false and MUST be SMT-verified before
    // adoption. Default OFF to preserve the existing behaviour.
    //
    // Enabling this makes clunk's search ergodic over the instruction
    // space (the headline STOKE win), at the cost of producing unsound
    // candidates that require verification.
    bool allow_unsound_mutations = false;

    // `test_vector_count`: when > 0, every improving candidate
    // produced by search() is pre-filtered through a fast interpreter
    // pass on `test_vector_count` random input vectors. Candidates that
    // disagree with the original on any vector are NOT inserted into the
    // returned candidate list (Stats::candidates_rejected_by_test_vectors
    // is incremented). 0 = disabled (pure SMT gating, the legacy
    // behaviour). Default 32 matches STOKE §5.1.
    size_t test_vector_count = 32;

    // `two_phase_mode`: when true, search() runs in two phases:
    //   Phase 1 (Synthesis): `two_phase_split` of `max_iterations` with
    //     unsound mutations ENABLED, exploring the full instruction
    //     space (including incorrect programs) to find a promising
    //     basin. Phase-1 candidates are NOT sound.
    //   Phase 2 (Optimization): the remaining iterations with unsound
    //     mutations DISABLED, starting from the best phase-1 candidate
    //     (or the original if no phase-1 candidate was accepted). Phase-2
    //     mutations are sound kinds, but phase-2 candidates inherit the
    //     unsoundness of their phase-1 ancestor and MUST still be SMT-
    //     verified before adoption.
    // Default OFF. When OFF, `allow_unsound_mutations` controls the
    // single-phase behaviour as documented above.
    bool two_phase_mode = false;

    // Fraction of `max_iterations` allocated to phase 1 (synthesis) when
    // `two_phase_mode` is true. Must be in [0, 1]. Default 0.3 (30%
    // synthesis, 70% optimization) per STOKE §4.4.
    double two_phase_split = 0.3;
};

// ── Optimisation candidate ──────────────────────────────────────────────
struct Candidate {
    std::shared_ptr<ir::Function> function;  // The candidate function
    double score;                             // Evaluation engine score
    size_t iteration_found;                   // When was this found
    std::string description;                  // Human-readable description

    // New field (backward-compatible): structural hash of the candidate
    // function. Used for deduplication instead of the description string
    // so that two structurally different functions with the same
    // mutation description are both kept.
    uint64_t structural_hash = 0;

    // True iff this candidate was derived from the search baseline SOLELY
    // via semantics-preserving rewrites (DCE with use check, constant
    // folding, CSE, algebraic identities, strength reduction, guarded
    // swaps). Such candidates are correct by construction and can be
    // adopted without SMT verification — crucial for functions the
    // verifier cannot model (memory ops, floats, loops).
    bool sound = false;

    bool operator<(const Candidate& other) const {
        return score < other.score;
    }
};

// ── Mutation operations ─────────────────────────────────────────────────
enum class MutationKind : uint8_t {
    ReplaceInstruction,    // Replace an instruction with an equivalent
    DeleteInstruction,     // Remove a redundant instruction
    SwapInstructions,      // Reorder two independent instructions
    StrengthReduce,        // Replace expensive op with cheaper (mul/udiv/urem -> shift/mask)
    FoldConstant,          // Fold a constant expression
    UnrollLoop,            // Unroll a small loop
    CombineInstructions,   // Merge two instructions into one
    SubstituteVariable,    // Replace a variable with an equivalent one
    PatternGuided,         // Apply a pattern from the PatternLibrary
    SimplifyIdentity,      // Algebraic identity (x+0 -> x, x^x -> 0, ...)
    EliminateCommonSubexpr,// Same-block CSE: reuse an identical earlier result

    // ── STOKE-style moves — UNSOUND, require SMT verification ────────
    // These five kinds explore the *equivalent-program space* rather than
    // the *semantics-preserving-rewrite* space. They make clunk's search
    // ergodic over the instruction space (STOKE ASPLOS'13 §4.3) but
    // produce candidates that are NOT correct by construction —
    // is_sound_kind() returns false for them and the Pipeline must verify
    // them via SMT before adoption (or reject them via the test-
    // vector pre-filter).
    OpcodeReplace,         // Swap the opcode for a different one (Add -> Sub)
    OperandReplace,        // Replace one operand with another existing value
    OperandSwap,           // Swap the two operands of a binary op
    InstructionInsert,     // Insert a random instruction at a random position
    InstructionReplace,    // Replace an instruction with a random one (full)
};

struct Mutation {
    MutationKind kind;
    size_t instruction_index;        // Where in the block
    std::string block_name;          // Which basic block
    std::string description;
    // Secondary index (backward-compatible; 0 = unset). Meaning by kind:
    //   CombineInstructions:    index of the SECOND instruction (the use);
    //                           0/<=instruction_index = legacy adjacent pair
    //   EliminateCommonSubexpr: index of the EARLIER duplicate to reuse
    //   OperandReplace:         index of the operand to replace (0-based)
    size_t aux_index = 0;
};

// A concrete applicable-mutation site discovered by scanning a function.
// Site-driven mutation makes every search iteration meaningful on large
// functions, where a blind random (block, index, kind) triple almost
// never lands on an applicable rewrite.
struct MutationSite {
    MutationKind kind;
    std::string block_name;
    size_t instruction_index;
    size_t aux_index = 0;
};

// ── Stochastic Search ───────────────────────────────────────────────────
class StochasticSearch final {
public:
    // Primary constructor (backward-compatible).
    explicit StochasticSearch(const StochasticConfig& config = {},
                               evaluator::EvaluationEngine* engine = nullptr);

    // Extended constructor: accepts an optional PatternLibrary pointer
    // for the PatternGuided mutation kind. When `lib`
    // is null, PatternGuided is silently skipped.
    StochasticSearch(const StochasticConfig& config,
                      evaluator::EvaluationEngine* engine,
                      pattern::PatternLibrary* lib);

    // Run stochastic search on a function
    std::vector<Candidate> search(const ir::Function& original);

    // Reset internal state for reuse. Re-seeds the RNG and clears the
    // cached original-function analysis. Allows EvolutionarySearch to
    // reuse a single StochasticSearch instance across many mutate() calls
    // instead of constructing a fresh one each time.
    void reset(unsigned new_seed);

    // Format a Mutation's human-readable description. Decoupled from
    // random_mutation() so callers that reject most mutations (e.g., the
    // simulated-annealing search loop) can avoid the string-formatting
    // cost on every iteration.
    static std::string describe_mutation(const Mutation& m);

    // Configuration
    StochasticConfig& config() { return config_; }
    const StochasticConfig& config() const { return config_; }

    // PatternLibrary accessor.
    void set_pattern_library(pattern::PatternLibrary* lib) { pattern_lib_ = lib; }
    pattern::PatternLibrary* pattern_library() const { return pattern_lib_; }

    // Statistics
    struct Stats {
        size_t iterations_run = 0;
        size_t candidates_found = 0;
        size_t mutations_tried = 0;
        size_t mutations_accepted = 0;
        size_t mutations_rejected_by_validation = 0;  // IR-corruption guard rejections
        size_t pattern_guided_attempts = 0;
        size_t pattern_guided_applied = 0;
        size_t stagnation_restarts = 0;
        size_t score_cache_hits = 0;
        double best_score = -std::numeric_limits<double>::infinity();
        double final_temperature = 0.0;
        // Wall-clock time spent in search(), in seconds.
        double elapsed_seconds = 0.0;

        // ── STOKE-feature stats ──────────────────────────────────────────
        // How many of the STOKE-style mutations were proposed.
        size_t stoke_moves_tried = 0;
        // Candidates rejected by the test-vector pre-filter.
        size_t candidates_rejected_by_test_vectors = 0;
        // Score-cache hits attributable to the canonical-form key
        // (renaming-equivalent functions collapsing to one entry). Tracked
        // separately from `score_cache_hits` so the canonical-form win is
        // measurable; both counters increment on a canonical-form hit.
        size_t canonical_cache_hits = 0;
        // Scoring short-circuits — improving candidates that
        // short-circuit accept() without drawing the Metropolis RNG. The
        // alternative "sample u first then abort partial scoring" would
        // require evaluator changes we don't own; we implement the
        // documented fallback (skip RNG for always-accept improvements).
        size_t score_evaluations_short_circuited = 0;
    };
    const Stats& stats() const { return stats_; }

    // Generate a random mutation (public for use by EvolutionarySearch).
    // Prefers an applicable site from collect_mutation_sites; falls back
    // to a blind random pick when no sites exist (pure exploration).
    Mutation random_mutation(const ir::Function& fn);

    // Scan the function for all applicable mutation sites (dead code,
    // foldable constants, CSE pairs, identities, strength reductions,
    // combinable pairs, legal swaps). Public for tests and for
    // EvolutionarySearch.
    std::vector<MutationSite> collect_mutation_sites(const ir::Function& fn) const;

    // Scan for sites, optionally including the
    // five STOKE-style unsound sites. When `allow_unsound` is true,
    // every non-terminator instruction is also a site for OpcodeReplace
    // / OperandReplace / InstructionReplace, every binary instruction
    // is a site for OperandSwap, and every position-between-instructions
    // is a site for InstructionInsert. The single-arg overload above
    // delegates to this one with `config_.allow_unsound_mutations`.
    std::vector<MutationSite> collect_mutation_sites(const ir::Function& fn,
                                                      bool allow_unsound) const;

    // True iff a mutation of this kind is semantics-preserving by
    // construction (see Candidate::sound).
    static bool is_sound_kind(MutationKind kind) {
        switch (kind) {
            case MutationKind::PatternGuided:
            // STOKE-style moves are NOT sound by construction.
            case MutationKind::OpcodeReplace:
            case MutationKind::OperandReplace:
            case MutationKind::OperandSwap:
            case MutationKind::InstructionInsert:
            case MutationKind::InstructionReplace:
                return false;
            default:
                return true;
        }
    }

    // True iff a mutation of this kind can be applied transactionally in
    // place (and undone by a MutationScope) rather than by deep-copying the
    // whole function. All current kinds except PatternGuided — which
    // delegates to PatternLibrary::apply and materialises a fresh Function —
    // are amenable to in-place application. The five new STOKE-style kinds
    // record ordinary Replace / Insert actions on the scope, so they
    // undo cleanly.
    static bool is_in_place_kind(MutationKind kind) {
        return kind != MutationKind::PatternGuided;
    }

    // Apply a mutation to a function, returning the mutated function (public for use by EvolutionarySearch)
    std::shared_ptr<ir::Function> apply_mutation(const ir::Function& fn, const Mutation& mut);

    // Compute a 64-bit structural hash of a function (FNV-1a over
    // opcode + operand-name stream per block, XOR-combined). Public so
    // EvolutionarySearch can reuse it for its own dedup / cache.
    static uint64_t structural_hash(const ir::Function& fn);

    // ── Test-vector pre-filter ───────────────────────────────────────
    // Run the original and the candidate through `num_vectors` random
    // test inputs via the existing evaluator::Interpreter. Returns true
    // iff the candidate agrees with the original on every vector. If the
    // interpreter cannot evaluate either function (memory ops, floats,
    // loops, etc.) the filter is skipped (returns true) so the candidate
    // proceeds to SMT — the test-vector layer is a strict pre-filter,
    // never a soundness gate. Non-const because it increments
    // Stats::candidates_rejected_by_test_vectors on a mismatch.
    bool passes_test_vectors(const ir::Function& original,
                              const ir::Function& candidate,
                              size_t num_vectors = 32);

    // ── Canonical structural hash ─────────────────────────────────────
    // Hash the function under a canonical renaming: SSA values are
    // renamed in def-order (%v0, %v1, …) and integer constants are
    // renamed per-bitwidth (%c0, %c1, …). Two functions that are
    // structurally identical up to renaming share a canonical hash, so
    // the score-cache key collapses renaming-equivalent functions to a
    // single entry (Bansal & Aiken ASPLOS'06, Table 1: 50× search-space
    // reduction at length 3). Used internally by score_with_cache; the
    // existing structural_hash is preserved for backward compat with
    // EvolutionarySearch's dedup.
    static uint64_t canonical_structural_hash(const ir::Function& fn);

private:
    // Simulated annealing acceptance criterion. Takes the current
    // temperature as an explicit parameter so the schedule is correct
    // Non-const because it consumes the RNG.
    bool accept(double current_score, double candidate_score, double temperature);

    // Lazily compute and cache the analysis of the original function.
    // The original function is immutable for the duration of a single
    // search() call, so its analysis is computed at most once and reused
    // across all ~10,000 iterations.
    const evaluator::FunctionAnalysis& original_analysis() const;

    // Instruction-level mutations. Each returns a fresh Function on
    // success (deep-copy semantics for the public API) or nullptr on
    // failure / not-applicable.
    std::shared_ptr<ir::Function> strength_reduce(const ir::Function& fn, size_t idx, const std::string& bb);
    std::shared_ptr<ir::Function> fold_constant(const ir::Function& fn, size_t idx, const std::string& bb);
    std::shared_ptr<ir::Function> combine_instructions(const ir::Function& fn, size_t idx_a, size_t idx_b, const std::string& bb);
    std::shared_ptr<ir::Function> delete_dead_code(const ir::Function& fn, size_t idx, const std::string& bb);
    std::shared_ptr<ir::Function> simplify_identity(const ir::Function& fn, size_t idx, const std::string& bb);
    std::shared_ptr<ir::Function> eliminate_cse(const ir::Function& fn, size_t idx_dup, size_t idx_orig, const std::string& bb);

    // Pattern-guided mutation: pick a random pattern match and apply it.
    // Returns nullptr if no library is set or no match is found.
    std::shared_ptr<ir::Function> pattern_guided_mutate(const ir::Function& fn);

    // ── STOKE-style unsound mutations ──────────────────────────────────
    // Deep-copy variants (used by EvolutionarySearch::mutate and by the
    // legacy deep-copy fallback in apply_mutation_impl). Each returns a
    // fresh Function on success or nullptr if the mutation is not
    // applicable to the chosen site.
    std::shared_ptr<ir::Function> opcode_replace(const ir::Function& fn, size_t idx, const std::string& bb);
    std::shared_ptr<ir::Function> operand_replace(const ir::Function& fn, size_t idx, size_t op_idx, const std::string& bb);
    std::shared_ptr<ir::Function> operand_swap(const ir::Function& fn, size_t idx, const std::string& bb);
    std::shared_ptr<ir::Function> instruction_insert(const ir::Function& fn, size_t pos, const std::string& bb);
    std::shared_ptr<ir::Function> instruction_replace_full(const ir::Function& fn, size_t idx, const std::string& bb);

    // ── In-place mutation ──────────────────────────────────────────────
    // Transactional counterparts to the deep-copy helpers above. Each
    // prechecks applicability on the LIVE function held by `scope`,
    // applies the edit in place, and records undo actions on the scope so
    // rejection restores the function exactly. Returns true iff the
    // mutation was applied (and something was recorded), false if not
    // applicable (in which case the function is left untouched).
    bool apply_mutation_in_place(MutationScope& scope, const Mutation& mut,
                                 size_t inst_count);
    bool strength_reduce_in_place(MutationScope& scope, size_t idx, const std::string& bb);
    bool fold_constant_in_place(MutationScope& scope, size_t idx, const std::string& bb);
    bool combine_instructions_in_place(MutationScope& scope, size_t idx_a, size_t idx_b, const std::string& bb);
    bool delete_dead_code_in_place(MutationScope& scope, size_t idx, const std::string& bb);
    bool simplify_identity_in_place(MutationScope& scope, size_t idx, const std::string& bb);
    bool eliminate_cse_in_place(MutationScope& scope, size_t idx_dup, size_t idx_orig, const std::string& bb);
    bool swap_instructions_in_place(MutationScope& scope, size_t idx, const std::string& bb);

    // STOKE-style in-place variants — same transactional contract as above.
    bool opcode_replace_in_place(MutationScope& scope, size_t idx, const std::string& bb);
    bool operand_replace_in_place(MutationScope& scope, size_t idx, size_t op_idx, const std::string& bb);
    bool operand_swap_in_place(MutationScope& scope, size_t idx, const std::string& bb);
    bool instruction_insert_in_place(MutationScope& scope, size_t pos, const std::string& bb);
    bool instruction_replace_in_place(MutationScope& scope, size_t idx, const std::string& bb);

    // Shared precheck: the combine pattern is soundness-critical, so both
    // the copy and in-place paths build the merged instruction through this
    // one routine (returns nullptr if the combine is not applicable/safe).
    std::shared_ptr<ir::Instruction> build_combined_instruction(
        const ir::Function& fn, size_t idx_a, size_t idx_b, const std::string& bb);

    // When the function is near the instruction cap, only destructive /
    // size-neutral mutations are permitted so per-iteration work stays
    // bounded. Returns false if `kind` should be skipped at `inst_count`.
    bool mutation_allowed_near_cap(MutationKind kind, size_t inst_count) const;

    // Pick a mutation given a pre-computed site list (the hot search loop
    // caches sites across rejected iterations). Falls back to a blind
    // random mutation for exploration / when no sites exist.
    // `allow_unsound` controls whether the five STOKE-style unsound kinds
    // are eligible; when false they are filtered out of the site list and
    // excluded from the blind fallback.
    Mutation pick_mutation(const ir::Function& fn,
                            const std::vector<MutationSite>& sites,
                            bool allow_unsound);

    // Legacy blind random pick (exploration fallback). `allow_unsound`
    // controls whether the STOKE-style kinds are eligible.
    Mutation blind_random_mutation(const ir::Function& fn, bool allow_unsound);

    // Score a candidate, using the structural-hash cache to short-circuit
    // re-evaluation of structurally identical functions.
    // If `hash_out` is non-null, receives the structural hash computed for
    // the lookup, so callers don't hash the same function twice.
    double score_with_cache(const ir::Function& fn, uint64_t* hash_out = nullptr);

    // apply_mutation with a pre-computed instruction count for `fn`, so the
    // hot search loop doesn't re-count on every iteration.
    std::shared_ptr<ir::Function> apply_mutation_impl(const ir::Function& fn,
                                                       const Mutation& mut,
                                                       size_t inst_count);

    // Check if the time budget has been exceeded. Returns true if the
    // loop should exit now.
    bool time_budget_exceeded() const;

    StochasticConfig config_;
    evaluator::EvaluationEngine* engine_;
    pattern::PatternLibrary* pattern_lib_ = nullptr;
    std::mt19937 rng_;
    Stats stats_;

    // ── Cached original-function analysis ────────────────────────────────
    // The original function passed to search() is immutable during the
    // search, so its analysis is computed at most once per search() call.
    const ir::Function* original_ = nullptr;
    mutable evaluator::FunctionAnalysis cached_original_analysis_;
    mutable bool original_analysis_cached_ = false;

    // ── Structural-hash score cache ─────────────────────────────────────
    // Keyed by canonical_structural_hash(fn): renaming-equivalent
    // functions collapse to one entry, dramatically improving the cache
    // hit rate (Bansal Table 1). Cleared in reset().
    std::unordered_map<uint64_t, double> score_cache_;

    // ── Time-budget bookkeeping ─────────────────────────────────────────
    std::chrono::steady_clock::time_point start_time_;

    // ── Fresh-SSA-name counter for InstructionInsert / ─────────────────
    // InstructionReplace. Each new instruction is named "_stoke_<n>" so
    // the structural hash distinguishes distinct insertions. Incremented
    // monotonically per StochasticSearch instance so runs are
    // deterministic given the same seed.
    size_t next_stoke_name_ = 0;
};

} // namespace clunk::search
