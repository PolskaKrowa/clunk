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
 * Scope (deliberately conservative — inlining must be exact by
 * construction, since the pre-inlining caller cannot be SMT-verified):
 *   - callee is defined in the same module, is not the caller itself,
 *     and has exactly ONE basic block ending in `ret`;
 *   - no calls, invokes, phis, or allocas inside the callee (allocas
 *     would change per-iteration allocation behaviour when the call
 *     site sits in a loop; nested calls would need a call graph);
 *   - not vararg; argument count matches the call's operand count.
 *
 * Callee instructions are cloned with fresh SSA names, argument
 * references substitute to the call operands, and the `ret` value
 * substitutes for the call result everywhere in the caller. Memory
 * operations in the callee are fine — the clone preserves their exact
 * order and count at the call site.
 */
#include <cstddef>
#include <memory>

#include "clunk/IR/Function.h"
#include "clunk/IR/Module.h"

namespace clunk::search {

struct InlinerConfig {
    size_t max_callee_instructions = 32;  // excluding the ret
    size_t max_inlines_per_function = 8;
};

class Inliner {
public:
    explicit Inliner(const InlinerConfig& config = {});

    // Inline eligible call sites of `fn` using bodies from `mod`.
    // Returns the rewritten function, or nullptr if nothing was inlined.
    std::shared_ptr<ir::Function> inline_calls(const ir::Function& fn,
                                               const ir::Module& mod);

    struct Stats {
        size_t call_sites_seen = 0;
        size_t call_sites_inlined = 0;
    };
    const Stats& stats() const { return stats_; }

private:
    InlinerConfig config_;
    Stats stats_{};
};

} // namespace clunk::search
