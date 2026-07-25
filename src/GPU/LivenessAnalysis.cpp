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

/*
 * Clunk GPU LivenessAnalysis — backward liveness pass implementation.
 */
#include "clunk/GPU/LivenessAnalysis.h"

#include <algorithm>
#include <unordered_map>
#include <unordered_set>

namespace clunk::gpu {

// ── collect_uses ────────────────────────────────────────────────────────────
//
// Add every operand Value that the instruction reads to the use set.
// We only track operands that have a non-empty name — unnamed constants
// are always "available" (the emitter re-materialises them on the fly),
// so they don't contribute to register pressure in our model.
//
void LivenessAnalysis::collect_uses(const ir::Instruction& inst,
                                     LiveSets::ValueSet& out)
{
    for (const auto& op : inst.operands()) {
        if (!op) continue;
        // Track any operand that is itself a Value produced by an
        // instruction (i.e. has a name). Constants and function arguments
        // are also tracked by name so they appear in liveness — arguments
        // map to .param registers, but their liveness still affects the
        // register-pressure estimate.
        if (op->has_name()) {
            out.insert(op.get());
        }
    }
}

// ── collect_def ─────────────────────────────────────────────────────────────
//
// The "def" of an instruction is its own result Value. Instructions
// that produce no result (Store, Br, Ret, Fence, Switch, etc.) define
// nothing and return nullptr.
//
const ir::Value* LivenessAnalysis::collect_def(const ir::Instruction& inst) {
    // Terminators, stores and fences have no result.
    if (inst.is_terminator()) return nullptr;
    if (inst.opcode() == ir::Opcode::Store) return nullptr;
    if (inst.opcode() == ir::Opcode::Fence) return nullptr;

    // If the instruction has a name, it defines that named value.
    if (inst.has_name()) return &inst;
    return nullptr;
}

// ── compute ─────────────────────────────────────────────────────────────────
//
// Standard backward-dataflow liveness:
//
//   live_out[B] = union of live_in[S] for every successor S of B
//   live_in[B]  = use(B) ∪ (live_out[B] \ def(B))
//
// where use(B)/def(B) are accumulated across all instructions in B.
//
// Then per-instruction liveness is computed by a single backward sweep
// of each block:
//
//   live_out[I] = live_in[I_succ]  (or live_out[B] if I is the last instr)
//   live_in[I]  = use(I) ∪ (live_out[I] \ def(I))
//
// Because the IR is SSA, the standard fixed-point iteration converges
// in at most 2 passes (a single reverse-postorder sweep plus one more
// to propagate across loop back-edges). We do up to 3 passes for safety.
//
LiveSets LivenessAnalysis::compute(const ir::Function& fn) {
    LiveSets result;

    // Make sure predecessor info is up-to-date (successors are computed
    // on the fly from terminators, predecessors are not).
    const_cast<ir::Function&>(fn).compute_predecessors();

    const auto& blocks = fn.blocks();
    if (blocks.empty()) return result;

    // Per-block use and def sets (block-local — backward sweep within block)
    std::unordered_map<const ir::BasicBlock*, LiveSets::ValueSet> block_use;
    std::unordered_map<const ir::BasicBlock*, LiveSets::ValueSet> block_def;

    for (const auto& bb : blocks) {
        if (!bb) continue;
        LiveSets::ValueSet uses, defs;
        for (const auto& inst : bb->instructions()) {
            // For each instruction, uses are read before def — but for
            // block-level use/def (which is what transfer functions need),
            // we use the standard "use is anything read before being
            // defined locally" formulation. In SSA, every value is defined
            // exactly once, so a value used in B is either defined in B
            // (and then it's not in block_use) or defined elsewhere (and
            // then it IS in block_use).
            LiveSets::ValueSet inst_uses;
            collect_uses(*inst, inst_uses);
            const ir::Value* d = collect_def(*inst);

            for (const auto* u : inst_uses) {
                if (defs.count(u) == 0) {
                    uses.insert(u);
                }
            }
            if (d) defs.insert(d);
        }
        block_use[bb.get()] = std::move(uses);
        block_def[bb.get()] = std::move(defs);
    }

    // Fixed-point iteration for block_live_in / block_live_out
    // (live_out[B] = union of live_in[S]; live_in[B] = use ∪ (out - def))
    bool changed = true;
    unsigned iteration = 0;
    const unsigned kMaxIterations = 8;  // generous; SSA converges in ≤3

    while (changed && iteration < kMaxIterations) {
        changed = false;
        ++iteration;

        // Walk blocks in reverse order so that, in acyclic CFGs, one pass
        // suffices. Loop back-edges require another iteration.
        for (auto it = blocks.rbegin(); it != blocks.rend(); ++it) {
            const auto& bb = *it;
            if (!bb) continue;

            // live_out[B] = union of live_in[S]
            LiveSets::ValueSet new_out;
            for (const auto& succ_name : bb->successors()) {
                auto succ = fn.block(succ_name);
                if (!succ) continue;
                auto lit = result.block_live_in.find(succ.get());
                if (lit != result.block_live_in.end()) {
                    for (const auto* v : lit->second) new_out.insert(v);
                }
            }

            // live_in[B] = use[B] ∪ (live_out[B] - def[B])
            LiveSets::ValueSet new_in;
            const auto& uses = block_use[bb.get()];
            const auto& defs = block_def[bb.get()];
            for (const auto* v : uses) new_in.insert(v);
            for (const auto* v : new_out) {
                if (defs.count(v) == 0) new_in.insert(v);
            }

            auto cur_in_it = result.block_live_in.find(bb.get());
            auto cur_out_it = result.block_live_out.find(bb.get());
            if (cur_in_it == result.block_live_in.end() ||
                cur_in_it->second != new_in) {
                result.block_live_in[bb.get()] = std::move(new_in);
                changed = true;
            }
            if (cur_out_it == result.block_live_out.end() ||
                cur_out_it->second != new_out) {
                result.block_live_out[bb.get()] = std::move(new_out);
                changed = true;
            }
        }
    }

    // Now do the per-instruction backward sweep inside each block.
    size_t function_max = 0;
    for (const auto& bb : blocks) {
        if (!bb) continue;
        const auto& instrs = bb->instructions();
        if (instrs.empty()) continue;

        // Start: live_out[terminator] = live_out[B]
        LiveSets::ValueSet cur;
        const auto& bout = result.block_live_out[bb.get()];
        for (const auto* v : bout) cur.insert(v);

        size_t block_max = cur.size();

        // Walk instructions in reverse, propagating liveness backwards.
        for (auto iit = instrs.rbegin(); iit != instrs.rend(); ++iit) {
            const auto& inst = *iit;
            result.live_out[inst.get()] = cur;

            // live_in[I] = use(I) ∪ (live_out[I] - def(I))
            LiveSets::ValueSet new_in;
            collect_uses(*inst, new_in);
            const ir::Value* d = collect_def(*inst);
            for (const auto* v : cur) {
                if (v != d) new_in.insert(v);
            }
            cur = std::move(new_in);
            result.live_in[inst.get()] = cur;

            if (cur.size() > block_max) block_max = cur.size();
        }

        result.block_max_live[bb.get()] = block_max;
        if (block_max > function_max) function_max = block_max;
    }

    result.function_max_live = function_max;
    return result;
}

} // namespace clunk::gpu
