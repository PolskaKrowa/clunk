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
 * Clunk GPU LivenessAnalysis — backward liveness pass for register-pressure
 * estimation and instruction scheduling.
 *
 * A value is "live" at a program point if it has a use that is reachable
 * from that point without first being redefined. We compute, per
 * instruction, the set of Values live on entry (live_in) and on exit
 * (live_out). The maximum |live_in| over all program points is the
 * function's "register pressure" — the minimum number of registers needed
 * to hold simultaneously-live SSA values.
 *
 * The IR has no use-def chains (no Value::users() and no Use objects),
 * so we walk every instruction's operands to discover uses, and treat
 * the named result of an Instruction as its def.
 */
#include "clunk/IR/BasicBlock.h"
#include "clunk/IR/Function.h"
#include "clunk/IR/Instruction.h"
#include "clunk/IR/Value.h"

#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace clunk::gpu {

struct LiveSets {
    using ValueSet = std::unordered_set<const ir::Value*>;

    // live_in[I]  = values live on entry to instruction I (before I executes)
    // live_out[I] = values live on exit from instruction I (after I executes)
    std::unordered_map<const ir::Instruction*, ValueSet> live_in;
    std::unordered_map<const ir::Instruction*, ValueSet> live_out;

    // Per-block live-in (entry to the block) and live-out (exit from the block)
    std::unordered_map<const ir::BasicBlock*, ValueSet> block_live_in;
    std::unordered_map<const ir::BasicBlock*, ValueSet> block_live_out;

    // Maximum simultaneously-live set size observed at any instruction
    // boundary inside the function. This is the figure register pressure
    // reductions should compare against the per-thread register budget.
    size_t function_max_live = 0;

    // Per-block max simultaneously-live set size (peak |live_in| inside
    // the block, taken at the moment just before each instruction).
    std::unordered_map<const ir::BasicBlock*, size_t> block_max_live;
};

class LivenessAnalysis {
public:
    LivenessAnalysis() = default;

    // Compute liveness for every instruction in the function.
    // Idempotent; safe to call multiple times.
    LiveSets compute(const ir::Function& fn);

private:
    // Collect the set of values "used" (read) by an instruction.
    static void collect_uses(const ir::Instruction& inst,
                              LiveSets::ValueSet& out);

    // Collect the value "defined" (written) by an instruction, if any.
    // Returns nullptr for instructions like Store that produce no value.
    static const ir::Value* collect_def(const ir::Instruction& inst);
};

} // namespace clunk::gpu
