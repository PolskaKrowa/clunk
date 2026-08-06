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
 * Clunk Hole-Based Progressive-Deepening Synthesiser.
 *
 * A complementary search strategy to the stochastic / evolutionary phases.
 * Instead of mutating the existing function body and checking each mutation
 * for SMT equivalence, HoleSynth replaces the function with a *hole* — a
 * blank body whose only constraint is "must end with `ret <something>`" —
 * and enumerates candidate fillings in order of increasing size:
 *
 *   depth 1: one instruction equivalent to the whole function
 *            (e.g. `ret i32 %x` for the identity, `ret i32 5` for a
 *             constant, `ret i32 (shl i32 %x, 1)` for x*2)
 *   depth 2: two instructions
 *            (e.g. `mul i32 %x, %x` + `ret`)
 *   depth 3, 4, ...: capped by max_depth
 *
 * At each depth, every enumerated candidate is SMT-verified against the
 * original via the existing SMTVerifier. The first verified candidate the
 * cost model scores strictly cheaper than the original is returned. If no
 * candidate at depth N verifies (or all verified candidates score worse),
 * the search advances to depth N+1.
 *
 * This is the classical "superoptimisation by enumeration" strategy
 * (Massalin & Pu 1987) layered on top of the existing SMT / cost-model
 * infrastructure — a deterministic, completeness-bounded complement to
 * the stochastic and evolutionary phases, which are probabilistic.
 *
 * Scope (intentionally narrow, for tractability):
 *   - Single-block integer functions only. The synthesised candidate is a
 *     single straight-line block ending in `ret`. (Multi-block originals
 *     are passed through; their candidates come from the other phases.)
 *   - Integer arguments and integer return type. Pointers, floats, and
 *     vectors are refused (the SMT verifier already refuses them).
 *   - The original must be SMT-modelable (no memory / calls / loops), or
 *     the verify() call returns Unknown and the candidate is rejected.
 *
 * Soundness: every returned candidate carries an SMT equivalence proof
 * (`proven` is set true). If Z3 is unavailable the pass returns nullptr
 * (it cannot prove anything without a real prover). When `trust_unverified`
 * is set, candidates that pass the test-vector differential pre-filter
 * AND score cheaper may be returned with `proven = false`.
 */
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "clunk/IR/Function.h"
#include "clunk/Evaluator/EvaluationEngine.h"
#include "clunk/Search/SMTVerifier.h"
#include "clunk/Search/ThreadPool.h"

namespace clunk::search {

struct HoleSynthConfig {
    // Maximum hole depth to enumerate. Depth 1 = a single instruction;
    // depth 2 = two instructions; and so on. The enumeration cost is
    // roughly O((|opcodes| × |operand_pool|²)^depth), so the default
    // keeps the cap modest — most superoptimiser wins live at depth ≤ 3.
    size_t max_depth = 3;

    // Wall-clock budget in seconds for a single synthesize() call.
    // 0 = no internal cap (rely on the pipeline-level time budget).
    // Default 2s — generous enough to find depth-3 equivalents on
    // small functions, bounded enough to keep the pipeline responsive
    // on functions where no equivalent exists. The pipeline further
    // caps this at the remaining wall-clock budget.
    double time_budget_seconds = 2.0;

    // Number of random test vectors used as a pre-filter before SMT.
    // A candidate that does not agree with the original on every test
    // vector is skipped without an SMT call (sound: the pre-filter only
    // ever widens "not equivalent", never blesses a wrong candidate).
    size_t test_vector_count = 16;

    // Return unproven rewrites (score-gated + test-vector-passing) when
    // SMT is unavailable or returns Unknown. Off by default: no proof,
    // no rewrite.
    bool trust_unverified = false;

    // Per-candidate SMT timeout in milliseconds.
    unsigned smt_timeout_ms = 5000;

    // Cap on the original function's instruction count. Functions above
    // this cap are skipped (the synthesiser is most effective on small
    // integer idioms where it can find a 1-2 instruction equivalent).
    size_t max_original_instructions = 32;

    // ── Per-function multithreading (work-stealing) ──────────────────────
    // When a ThreadPool is attached to the synthesiser (see
    // set_thread_pool) AND parallel_search is true, synthesize() splits
    // the search space at each depth into independent "first-step" work
    // items and dispatches them to the pool. Each worker enumerates the
    // full subtree under its claimed first step, collecting ALL verified
    // candidates it finds (NOT short-circuiting on the first hit). When
    // the depth is exhausted, the synthesiser picks the verified
    // candidate with the LOWEST cost score — i.e. the fastest equivalent
    // sequence at that depth — and returns it. If no candidate was found
    // at depth N, the search advances to depth N+1 (the standard
    // progressive-deepening behaviour).
    //
    // This implements the user's "per-function multithreading"
    // requirement: idle threads (from finished functions or a small
    // program) help search the in-progress function's codespace, and
    // when multiple threads find different optimal candidates at the
    // same depth, the score arbitrates. The depth-bounded search
    // ensures we never miss a shorter equivalent: if depth N has any
    // candidate, we don't look at depth N+1, even if more workers
    // could be put to work there.
    //
    // The pool is OPTIONAL: when null or parallel_search is false,
    // synthesize() runs the existing single-threaded progressive
    // deepening. Default ON so callers that attach a pool get the
    // parallel behaviour automatically.
    bool parallel_search = true;

    // Minimum number of "first-step" work items required to actually
    // engage parallel search. Below this threshold the overhead of
    // dispatching to the pool dominates, so the synthesiser runs the
    // single-threaded path even when a pool is attached. Default 16 —
    // roughly the work-item count of a 1-arg function at depth 1
    // (9 opcodes × (1 arg + 16 consts)² = 9 × 17² = 2601, well above
    // the threshold; the threshold mainly guards tiny depth-1
    // special-case enumerations).
    size_t parallel_min_work_items = 16;
};

class HoleSynthesizer {
public:
    explicit HoleSynthesizer(evaluator::EvaluationEngine* engine,
                              const HoleSynthConfig& config = {});

    // Try to synthesise a cheaper equivalent of `fn` by progressive
    // deepening. Returns the rewritten function, or nullptr if no
    // candidate at any depth ≤ max_depth verified (or all verified
    // candidates scored worse than the original). `proven` (optional
    // out) is set true iff the returned rewrite carries an SMT
    // equivalence proof.
    //
    // When a ThreadPool is attached (see set_thread_pool) AND
    // config_.parallel_search is true, the per-depth search is
    // parallelised — see HoleSynthConfig::parallel_search for the
    // semantics.
    std::shared_ptr<ir::Function> synthesize(const ir::Function& fn,
                                              bool* proven = nullptr);

    // Attach a thread pool for parallel per-depth search. The pool is
    // NOT owned by the synthesiser; the caller must keep it alive for
    // the lifetime of the synthesiser. Passing nullptr disables
    // parallel search (the synthesiser falls back to the
    // single-threaded path). Safe to call between synthesize()
    // invocations.
    void set_thread_pool(ThreadPool* pool) { pool_ = pool; }

    struct Stats {
        size_t functions_seen = 0;
        size_t functions_skipped = 0;       // outside scope (multi-block, FP, ...)
        size_t candidates_enumerated = 0;   // total candidates tried
        size_t candidates_test_vector_pruned = 0;
        size_t candidates_smt_verified = 0;
        size_t candidates_equivalent = 0;   // SMT-proven equivalent
        size_t rejected_by_score = 0;       // verified but not cheaper
        size_t proven = 0;                  // rewrites returned with a proof
        size_t best_depth = 0;              // depth of the returned rewrite
        double elapsed_ms = 0.0;

        // ── Parallel-search statistics ────────────────────────────────
        // When parallel_search is engaged, these report the work-stealing
        // dispatch: `parallel_depths_run` is the number of depths that
        // were searched in parallel; `parallel_work_items_dispatched` is
        // the total number of first-step subtrees submitted to the pool;
        // `parallel_candidates_per_work_item_avg` is the average subtree
        // size (candidates_enumerated / work_items_dispatched). They are
        // all zero on the single-threaded path.
        size_t parallel_depths_run = 0;
        size_t parallel_work_items_dispatched = 0;
        double parallel_candidates_per_work_item_avg = 0.0;
    };
    const Stats& stats() const { return stats_; }

    HoleSynthConfig& config() { return config_; }
    const HoleSynthConfig& config() const { return config_; }

private:
    evaluator::EvaluationEngine* engine_;
    HoleSynthConfig config_;
    ir::TypeContext type_ctx_;
    Stats stats_{};
    ThreadPool* pool_ = nullptr;  // optional; not owned
};

} // namespace clunk::search
