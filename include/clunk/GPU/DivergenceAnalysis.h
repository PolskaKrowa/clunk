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
 * Clunk GPU DivergenceAnalysis — uniform vs divergent value analysis.
 *
 * A value is "uniform" if it has the same value for every thread in a
 * warp; it is "divergent" if it may differ across threads. The canonical
 * divergent values are threadIdx.{x,y,z}, blockIdx.{x,y,z} (when the
 * grid/block extent along that axis is > 1), and laneid.
 *
 * The analysis is value-propagation:
 *   - Kernel arguments are uniform (every thread sees the same arg).
 *   - Constants are uniform.
 *   - Values whose name suggests a thread-id builtin are divergent.
 *     We recognise: "tid", "threadIdx", "blockIdx", "laneid", "lane_id",
 *     "warpid", "warp_id", "globaltid".
 *   - An instruction whose opcode is Br (conditional) is divergent iff
 *     its condition operand is divergent.
 *   - An instruction with a divergent operand is itself divergent, with
 *     the following exceptions:
 *       * Pointer arithmetic on a uniform base with a divergent index
 *         produces a divergent *pointer* but the pointer is not "thread-
 *         uniform" — it stays divergent.
 *       * ICmp/FCmp inherit divergence from their operands.
 *       * Alloca is divergent iff its size operand is divergent.
 *   - GetElementPtr propagates divergence from any index operand.
 *   - Load: divergent iff its *pointer* operand is divergent (loads
 *     from a uniform pointer return uniform values across the warp).
 *   - Store: divergent iff its pointer operand is divergent.
 *
 * For each conditional branch marked divergent, we identify the
 * immediate post-dominator of its containing block (approximated as
 * the nearest common successor of both branch targets) as the
 * reconvergence point.
 */
#include "clunk/IR/BasicBlock.h"
#include "clunk/IR/Function.h"
#include "clunk/IR/Instruction.h"
#include "clunk/IR/Value.h"

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace clunk::gpu {

struct DivergenceResult {
    // Per-value divergence flag (true = divergent).
    // Keyed by raw Value* pointer identity.
    std::unordered_map<const ir::Value*, bool> is_divergent;

    // Per-block: is this block an identified reconvergence point
    // (immediate post-dominator of some divergent branch)?
    std::unordered_map<const ir::BasicBlock*, bool> is_reconvergence;

    // Per-branch (conditional Br): is the branch divergent?
    std::unordered_map<const ir::Instruction*, bool> divergent_branches;

    size_t divergent_branch_count = 0;
    size_t total_conditional_branches = 0;
    size_t divergent_value_count = 0;
    size_t total_value_count = 0;

    // Fraction of conditional branches that are divergent, in [0, 1].
    double branch_divergence_fraction() const {
        if (total_conditional_branches == 0) return 0.0;
        return static_cast<double>(divergent_branch_count) /
               static_cast<double>(total_conditional_branches);
    }

    // Fraction of all tracked values that are divergent, in [0, 1].
    double value_divergence_fraction() const {
        if (total_value_count == 0) return 0.0;
        return static_cast<double>(divergent_value_count) /
               static_cast<double>(total_value_count);
    }
};

class DivergenceAnalysis {
public:
    DivergenceAnalysis() = default;

    DivergenceResult analyse(const ir::Function& fn);

private:
    // Heuristic: does this value's name suggest it is thread-id derived?
    static bool looks_like_thread_id(const std::string& name);

    // Does the opcode propagate divergence from operands to result?
    // (Most do; some like Load/Store look at the *pointer* operand only.)
    static bool propagates_divergence(ir::Opcode op);
};

} // namespace clunk::gpu
