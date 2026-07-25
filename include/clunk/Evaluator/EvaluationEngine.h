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
 * Clunk Evaluation Engine — static semantic analysis using weighted heuristics.
 * The heart of Clunk: reasons about code properties without executing it.
 *
 * ────────────────────────────────────────────────────────────────────────────
 * SIGN / RATIO CONVENTION:
 *
 *   The evaluator's `score` field follows the convention
 *       score = -cost,  cost ≥ 0   (i.e. "more negative = worse").
 *   This is the SAME convention used by the existing pipeline and search
 *   code, where `cand_score > orig_score` means "candidate is better"
 *   (less negative).
 *
 *   `score_candidate_with_cached_orig(...)` returns a *ratio* such that
 *       ratio > 1.0  → candidate is BETTER (cheaper) than original
 *       ratio == 1.0 → candidate is equivalent
 *       ratio < 1.0  → candidate is WORSE  (more expensive) than original
 *
 *   This matches the call-site expectations in
 *     - src/Search/StochasticSearch.cpp (SA `accept()` treats
 *       higher-ratio = better; `if (candidate_score > baseline_score)`);
 *     - src/Search/EvolutionarySearch.cpp.
 *
 *   Implementation: ratio = orig_abs / cand_abs  (both abs values are
 *   strictly positive after the no-NaN/Inf guard in analyse()).
 *
 *   DO NOT change this convention without updating the call sites in
 *   StochasticSearch.cpp and EvolutionarySearch.cpp — those files are
 *   owned by another agent and rely on `ratio > 1 == improvement`.
 * ────────────────────────────────────────────────────────────────────────────
 */
#include <memory>
#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include "clunk/IR/Module.h"
#include "clunk/IR/Function.h"
#include "clunk/IR/LoopAnalysis.h"
#include "clunk/Evaluator/CostModel.h"
#include "clunk/Evaluator/EvaluationCache.h"

namespace clunk::evaluator {

// ── Memory hierarchy layers ─────────────────────────────────────────────
enum class MemoryLayer {
    Register,   // Fastest — data in registers
    L1,         // L1 cache
    L2,         // L2 cache
    L3,         // L3 cache (shared across cores)
    DRAM,       // Main memory
    GPU_Global, // GPU global memory
    GPU_Shared, // GPU shared memory
    GPU_Local,  // GPU local memory
    NVMe,       // Persistent storage
    Network     // Remote storage / network
};

// ── Heuristic weights ───────────────────────────────────────────────────
struct HeuristicWeights {
    // Memory layer penalties — progressively stronger for slower layers
    double register_penalty   =    0.0;
    double l1_penalty         =   -1.0;
    double l2_penalty         =   -3.0;
    double l3_penalty         =   -8.0;
    double dram_penalty       =  -20.0;
    double gpu_global_penalty =  -25.0;
    double gpu_shared_penalty =   -5.0;
    double gpu_local_penalty  =  -15.0;
    double nvme_penalty       = -100.0;
    double network_penalty    = -500.0;

    // Task weight — computational heaviness
    double task_weight_multiplier = 1.0;

    // E-core cache penalty — applied when E-cores access large cache regions
    double ecore_cache_penalty = -50.0;

    // Instruction penalties
    double branch_mispredict_penalty = -5.0;
    double load_latency_base = -4.0;
    double store_latency_base = -3.0;
    double div_latency_penalty = -20.0;

    // Constant materialisation cost (POSITIVE — added to the per-instruction
    // cost accumulator, not a negative penalty). Encodes LLVM/TTI knowledge
    // that a "large" immediate — one that does not fit a typical instruction
    // immediate field — must be materialised with an extra mov / constant-pool
    // load rather than riding along for free in the ALU op. Constants that fit
    // in a signed 32-bit field cost nothing (they inline as immediates).
    double constant_materialisation_cost = 1.0;

    // Optimisation bonuses
    double loop_unroll_bonus = 10.0;
    double vectorize_bonus = 15.0;
    double register_reuse_bonus = 5.0;
};

// ── Analysis results for a single function ──────────────────────────────
struct FunctionAnalysis {
    std::string function_name;

    // Memory access profile
    std::map<MemoryLayer, size_t> memory_accesses; // layer -> access count

    // Instruction mix
    size_t total_instructions = 0;
    size_t memory_instructions = 0;
    size_t branch_instructions = 0;
    size_t compute_instructions = 0;
    size_t call_instructions = 0;

    // Task weight (0.0 = trivial, 1.0 = very heavy)
    double task_weight = 0.0;

    // Overall score (higher is better for optimisation potential)
    double score = 0.0;

    // Per-basic-block scores
    std::map<std::string, double> block_scores;

    // Suggested thread placement
    enum class CorePreference { P_Core, E_Core, Any };
    CorePreference core_preference = CorePreference::Any;

    // Hot data — values that are accessed frequently
    std::vector<std::string> hot_values;

    // Loop analysis
    size_t estimated_loop_count = 0;
    bool has_nested_loops = false;

    // Estimated execution-frequency multiplier of the hottest basic block
    // in the function, relative to a block outside any loop (which has
    // frequency 1.0). Derived from natural-loop nesting depth (8^depth,
    // capped at 64 — the same "likely taken" heuristic LLVM's
    // BlockFrequencyInfo defaults to for unprofiled back-edges). This is
    // what §2.1 block-frequency weighting is built on: task_weight,
    // critical_path_cycles, and the per-block cost accumulation in
    // analyse() all key off the same per-block frequency estimate rather
    // than a flat "count of loops present" bump, so instructions in a hot
    // inner loop are weighted differently from instructions on a cold path.
    double max_block_frequency = 1.0;

    // GPU-specific
    bool is_gpu_kernel = false;
    size_t estimated_register_pressure = 0;
    double estimated_warp_divergence = 0.0;

    // Critical-path analysis: longest weighted dependency chain
    // across all basic blocks, in cycles, using the CostModel.
    //   - Independent instructions contribute 0 to this sum (they can
    //     issue in parallel); only the longest chain is counted.
    //   - A function with the same total uops but shorter critical path
    //     has more ILP and scores better.
    double critical_path_cycles = 0.0;

    // Structural hash of the IR (used by EvaluationCache; exposed for
    // debugging / tests). 0 if never computed.
    uint64_t structural_hash = 0;
};

// ── The Evaluation Engine ───────────────────────────────────────────────
//
// Thread-safety: `analyse()` and `score_candidate*()` are const-correct
// and read only immutable state plus the thread-safe EvaluationCache.
// Multiple threads MAY call these methods concurrently on the same
// instance. The `set_*` mutators must NOT be called concurrently with
// any analyse/score call.
class EvaluationEngine final {
public:
    // Construct with default (X86_64) CostModel and default weights.
    EvaluationEngine();

    // Construct with explicit weights; CostModel defaults to X86_64.
    explicit EvaluationEngine(const HeuristicWeights& weights);

    // Construct with explicit weights + cost model.
    EvaluationEngine(const HeuristicWeights& weights,
                     std::shared_ptr<const CostModel> cost_model);

    // Analyse a single function
    FunctionAnalysis analyse(const ir::Function& fn) const;

    // Analyse all functions in a module
    std::map<std::string, FunctionAnalysis> analyse_module(const ir::Module& mod) const;

    // Score a candidate optimisation against the original
    // Returns the improvement ratio (>1 means improvement)
    double score_candidate(const ir::Function& original,
                           const ir::Function& candidate) const;

    // Score a candidate against a PRE-COMPUTED analysis of the original.
    // Allows the caller to cache the (immutable) original's analysis across
    // thousands of iterations of search.
    double score_candidate_with_cached_orig(const FunctionAnalysis& orig_analysis,
                                             const ir::Function& candidate) const;

    // Get/set weights
    const HeuristicWeights& weights() const { return weights_; }
    void set_weights(const HeuristicWeights& w) { weights_ = w; }

    // Cost model access
    const CostModel& cost_model() const { return *cost_model_; }
    void set_cost_model(std::shared_ptr<const CostModel> cm) {
        cost_model_ = cm ? std::move(cm) : make_cost_model(Arch::X86_64);
    }

    // Evaluation cache control
    void clear_cache() const;
    CacheStats cache_stats() const;
    void set_cache_capacity(size_t capacity) const;

    // Memory layer penalty helper
    double memory_penalty(MemoryLayer layer) const;

private:
    // Instruction-level analysis
    void analyse_instructions(const ir::Function& fn, FunctionAnalysis& result) const;

    // Memory access pattern analysis
    void analyse_memory_access(const ir::Function& fn, FunctionAnalysis& result) const;

    // Control flow analysis
    void analyse_control_flow(const ir::Function& fn, FunctionAnalysis& result) const;

    // Task weight estimation
    void estimate_task_weight(FunctionAnalysis& result) const;

    // Core placement decision
    void decide_core_placement(FunctionAnalysis& result) const;

    // Identify hot values (frequently accessed)
    void identify_hot_values(const ir::Function& fn, FunctionAnalysis& result) const;

    // Build a per-operand-name use-count map. Reused by both
    // identify_hot_values and the register-reuse bonus block
    // in analyse().
    std::unordered_map<std::string, size_t>
    compute_use_counts(const ir::Function& fn) const;

    // Compute the longest weighted dependency chain (critical path)
    // across all basic blocks, in cycles, using the CostModel. Each
    // instruction's latency contribution is scaled by its containing
    // block's estimated execution frequency (see `loop_info`/§2.1), so a
    // dependency chain inside a hot loop body counts for ~freq times its
    // one-shot cost — representing the total dynamic critical path rather
    // than a single static pass through the function.
    double compute_critical_path(
        const ir::Function& fn,
        const std::vector<ir::NaturalLoop>& loop_info) const;

    // Highest per-block execution-frequency estimate in the function (see
    // `FunctionAnalysis::max_block_frequency`).
    double compute_max_block_frequency(
        const ir::Function& fn,
        const std::vector<ir::NaturalLoop>& loop_info) const;

    // Validate HeuristicWeights: reject NaN/Inf, normalise so the
    // scoring convention holds (cost ≥ 0 after combining penalties
    // and bonuses). Mutates `weights_` in place; called from the ctor.
    void validate_weights();

    HeuristicWeights weights_;
    std::shared_ptr<const CostModel> cost_model_;
    // `mutable` because analyse() is const but the cache is an
    // implementation detail (transparent to callers).
    mutable EvaluationCache<FunctionAnalysis> cache_;
};

// ── Utility ─────────────────────────────────────────────────────────────
std::string memory_layer_name(MemoryLayer layer);

} // namespace clunk::evaluator