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
 * Clunk Cross-Function Optimisation Passes — module-level transforms
 * that span function boundaries.
 *
 * These passes operate on a whole Module and use the CallGraph to reason
 * about call relationships. They run BEFORE per-function superoptimisation,
 * producing a smaller / cheaper module that the per-function pipeline then
 * attacks with its full search stack.
 *
 * Two passes are provided:
 *
 * 1. DeadFunctionElimination — remove module-internal functions that are
 *    never called (and are not externally visible via linkage or by being
 *    named as an entry point). Removes dead code that the per-function
 *    pipeline would otherwise waste time analysing.
 *
 * 2. InterproceduralConstantPropagation (IPCP) — for every function
 *    argument where ALL direct callers pass the SAME constant value,
 *    clone the function with that argument replaced by the constant,
 *    and rewrite all callers to invoke the clone. The original function
 *    is kept if any caller remains (external callers, indirect callers,
 *    or callers that pass a non-constant). The clones are then eligible
 *    for further constant-folding by the per-function pipeline.
 *
 * Soundness: both passes are exact-by-construction (no SMT needed).
 * DFE only removes functions that no caller can reach (modulo linkage).
 * IPCP only specialises when EVERY caller passes the same constant
 * (verified via CallGraph::argument_constants); when in doubt, it
 * preserves the original.
 */
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "clunk/IR/Module.h"

namespace clunk::search {

struct CrossFnConfig {
    bool enable_dfe = true;       // dead function elimination
    bool enable_ipcp = true;      // interprocedural constant propagation
    size_t ipcp_max_clones_per_fn = 4;  // cap on clones per function
    size_t ipcp_max_total_clones = 32;  // global cap across the module
    size_t ipcp_min_fn_size = 2;        // skip trivial wrappers
    size_t ipcp_max_fn_size = 64;       // skip pathological giant functions
};

class CrossFunctionPasses {
public:
    explicit CrossFunctionPasses(const CrossFnConfig& config = {})
        : config_(config) {}

    // Run all enabled passes on `module` in-place. Returns true iff the
    // module was modified (a function was removed or specialised).
    bool run(ir::Module& module);

    // Individual passes (also exposed for testing).
    bool run_dead_function_elimination(ir::Module& module);
    bool run_ipcp(ir::Module& module);

    struct Stats {
        size_t dfe_removed = 0;
        size_t ipcp_cloned = 0;
        size_t ipcp_callers_rewritten = 0;
    };
    const Stats& stats() const { return stats_; }

private:
    CrossFnConfig config_;
    Stats stats_{};
};

} // namespace clunk::search
