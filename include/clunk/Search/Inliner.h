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
 * Clunk Inliner — interprocedural call-site inlining.
 *
 * Clunk optimises functions independently, and a `call` is a hard wall:
 * the SMT verifier reports Unknown for any function containing one, the
 * cost model bills it at 20-30 cycles, and no search phase can see
 * through it. Inlining small module-internal callees removes that wall —
 * the caller becomes cheaper AND verifiable, so every downstream phase
 * (miner, CEGIS, e-graph) can suddenly reason about the combined body.
 *
 * TWO inlining modes:
 *
 *  1. `inline_calls()` — backward-compatible single-block inliner
 *     (the original). Conservative eligibility: callee has exactly one
 *     basic block ending in `ret`, no calls/invokes/phis/allocas inside.
 *     Used as the per-function pre-pass before run_on_function().
 *
 *  2. `inline_calls_multiblock()` — the cross-function extension. Uses
 *     a module-level CallGraph to walk the call graph bottom-up and
 *     inline multi-block callees into their callers. Handles:
 *       - Multi-block callees (CFG cloned verbatim, all blocks renamed).
 *       - Phi nodes at the callee's entry (rewired to caller-side values
 *         via the predecessor-edge mapping).
 *       - Alloca instructions in the callee (hoisted to the caller's
 *         entry block — preserves per-call-stack-frame semantics).
 *       - Recursive call graphs: refuses to inline a callee into itself
 *         (call-graph SCC check), and applies a depth cap on the
 *         transitive inlining chain to avoid exponential blow-up.
 *
 * The multi-block inliner is gated by `enable_multiblock` (default true
 * at opt_level >= 2) and `max_caller_instructions_after` (default 256 —
 * refuse to grow a caller past this size).
 */
#include <cstddef>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "clunk/IR/Function.h"
#include "clunk/IR/Module.h"

namespace clunk::analysis { class CallGraph; }

namespace clunk::search {

struct InlinerConfig {
    size_t max_callee_instructions = 32;        // single-block mode (excl. ret)
    size_t max_inlines_per_function = 8;        // both modes
    // ── Multi-block inliner controls ──────────────────────────────────
    size_t max_multiblock_callee_instructions = 64;  // body size cap
    size_t max_multiblock_callee_blocks = 16;        // CFG size cap
    size_t max_caller_instructions_after = 256;      // refuse to grow past this
    size_t max_inline_depth = 4;                     // transitive inlining depth
    bool enable_multiblock = true;
};

class Inliner {
public:
    explicit Inliner(const InlinerConfig& config = {});

    // Single-block inlining (legacy). Returns the rewritten function or
    // nullptr if nothing was inlined.
    std::shared_ptr<ir::Function> inline_calls(const ir::Function& fn,
                                               const ir::Module& mod);

    // ── Multi-block, call-graph-aware inlining ───────────────────────
    // Returns a new function with eligible multi-block callees inlined
    // at their call sites, or nullptr if nothing changed. The optional
    // `cg` parameter is a pre-built call graph; if null, one is built
    // internally. `visited` is the recursion-guard set (callers pass an
    // empty set; the inliner adds callee names as it descends).
    std::shared_ptr<ir::Function> inline_calls_multiblock(
        const ir::Function& fn,
        const ir::Module& mod,
        const analysis::CallGraph* cg = nullptr,
        std::unordered_set<std::string> visited = {});

    struct Stats {
        size_t call_sites_seen = 0;
        size_t call_sites_inlined = 0;
        size_t multiblock_inlined = 0;
        size_t recursion_refused = 0;
        size_t size_refused = 0;
    };
    const Stats& stats() const { return stats_; }

    // Reset per-Inliner stats (useful between functions).
    void reset_stats() { stats_ = Stats{}; }

private:
    InlinerConfig config_;
    Stats stats_{};
};

} // namespace clunk::search
