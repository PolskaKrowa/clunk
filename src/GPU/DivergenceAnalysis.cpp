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
 * Clunk GPU DivergenceAnalysis — uniform vs divergent value analysis.
 */
#include "clunk/GPU/DivergenceAnalysis.h"

#include <algorithm>
#include <cctype>
#include <unordered_map>
#include <unordered_set>

namespace clunk::gpu {

// ── looks_like_thread_id ────────────────────────────────────────────────────
//
// Names that indicate a thread-id-derived (and thus divergent) value.
// We do a case-insensitive substring search for the most common CUDA
// builtin tokens: threadIdx, blockIdx, laneid, tidx, tidy, tidz, etc.
//
bool DivergenceAnalysis::looks_like_thread_id(const std::string& name) {
    if (name.empty()) return false;

    std::string s;
    s.reserve(name.size());
    for (char c : name) {
        s.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    }

    // Substrings that strongly suggest a thread-id derivation.
    // "tid" is a common short form for threadIdx.x; we match it as a
    // standalone token or as part of "tid_x" / "tidX" / "globaltid".
    static const char* kDivergentTokens[] = {
        "threadidx", "threadidy", "threadidz",
        "blockidx",  "blockidy",  "blockidz",
        "laneid",    "lane_id",   "laneindex",
        "warpid",    "warp_id",   "warpindex",
        "globaltid", "global_tid",
        "tid",       // catch "tid", "tid_x", "tidx", "globaltid"
        "gtid",      // common shorthand for global thread id
    };
    for (const char* tok : kDivergentTokens) {
        if (s.find(tok) != std::string::npos) return true;
    }
    return false;
}

// ── propagates_divergence ───────────────────────────────────────────────────
//
// Most arithmetic/logic opcodes propagate divergence from operands to
// result. The exceptions are Load (divergence depends only on the pointer
// operand, since a load from a uniform pointer returns a uniform value)
// and the no-result opcodes (Store, Br, Ret, Fence — they have no result
// to propagate to, but they still consume divergent operands).
//
bool DivergenceAnalysis::propagates_divergence(ir::Opcode op) {
    switch (op) {
        case ir::Opcode::Add:  case ir::Opcode::Sub:  case ir::Opcode::Mul:
        case ir::Opcode::UDiv: case ir::Opcode::SDiv:
        case ir::Opcode::URem: case ir::Opcode::SRem:
        case ir::Opcode::FAdd: case ir::Opcode::FSub:
        case ir::Opcode::FMul: case ir::Opcode::FDiv: case ir::Opcode::FRem:
        case ir::Opcode::And:  case ir::Opcode::Or:   case ir::Opcode::Xor:
        case ir::Opcode::Shl:  case ir::Opcode::LShr: case ir::Opcode::AShr:
        case ir::Opcode::ICmp: case ir::Opcode::FCmp:
        case ir::Opcode::Trunc: case ir::Opcode::ZExt: case ir::Opcode::SExt:
        case ir::Opcode::FPTrunc: case ir::Opcode::FPExt:
        case ir::Opcode::FPToUI: case ir::Opcode::FPToSI:
        case ir::Opcode::UIToFP: case ir::Opcode::SIToFP:
        case ir::Opcode::PtrToInt: case ir::Opcode::IntToPtr:
        case ir::Opcode::BitCast:  case ir::Opcode::AddrSpaceCast:
        case ir::Opcode::Select:
        case ir::Opcode::GetElementPtr:
        case ir::Opcode::ExtractValue: case ir::Opcode::InsertValue:
        case ir::Opcode::ExtractElement: case ir::Opcode::InsertElement:
        case ir::Opcode::ShuffleVector:
            return true;
        // Load: divergence follows only the pointer operand (operand 0)
        // — a load from a uniform pointer yields a uniform value.
        case ir::Opcode::Load:
            return false;
        // Alloca is per-thread stack space; we model it as uniform
        // (each thread has its own private stack, so the slot index
        // is the same across threads — divergence is irrelevant).
        case ir::Opcode::Alloca:
            return false;
        default:
            return false;
    }
}

// ── analyse ─────────────────────────────────────────────────────────────────
//
// Worklist algorithm:
//   1. Seed: kernel arguments are uniform; values named like "tid" etc.
//      are divergent; everything else starts uniform.
//   2. Iterate: for each instruction, if any divergent operand
//      propagates to the result, mark the result divergent.
//   3. Repeat until no changes.
//   4. Mark conditional branches whose condition is divergent.
//   5. Find reconvergence points (post-dominators of divergent branches).
//
DivergenceResult DivergenceAnalysis::analyse(const ir::Function& fn) {
    DivergenceResult result;

    // Build a name → BasicBlock map for successor lookups.
    std::unordered_map<std::string, const ir::BasicBlock*> bb_by_name;
    for (const auto& bb : fn.blocks()) {
        if (bb) bb_by_name[bb->name()] = bb.get();
    }

    // Seed: arguments are uniform (every thread sees the same arg).
    // We don't explicitly store uniform values in the map — only divergent
    // ones — to make the iteration cheaper. `is_divergent` defaults to
    // false on map miss.
    auto is_div = [&](const ir::Value* v) -> bool {
        if (!v) return false;
        auto it = result.is_divergent.find(v);
        return it != result.is_divergent.end() && it->second;
    };

    // Seed: values whose names look like thread-id builtins.
    for (const auto& bb : fn.blocks()) {
        if (!bb) continue;
        for (const auto& inst : bb->instructions()) {
            if (inst->has_name() && looks_like_thread_id(inst->name())) {
                result.is_divergent[inst.get()] = true;
            }
            // Also seed operands that look like thread-ids (e.g. references
            // to named values "tid" loaded from arguments).
            for (const auto& op : inst->operands()) {
                if (op && op->has_name() && looks_like_thread_id(op->name())) {
                    result.is_divergent[op.get()] = true;
                }
            }
        }
    }

    // Iterate to fixpoint.
    bool changed = true;
    unsigned iter = 0;
    const unsigned kMaxIter = 16;

    while (changed && iter < kMaxIter) {
        changed = false;
        ++iter;

        for (const auto& bb : fn.blocks()) {
            if (!bb) continue;
            for (const auto& inst : bb->instructions()) {
                if (!inst->has_name()) continue;  // no result, no propagation

                // Already known divergent — skip.
                if (is_div(inst.get())) continue;

                bool divergent = false;

                if (inst->opcode() == ir::Opcode::Load) {
                    // Load divergence comes from the pointer operand only.
                    if (inst->num_operands() >= 1) {
                        divergent = is_div(inst->operand(0).get());
                    }
                } else if (inst->opcode() == ir::Opcode::Alloca) {
                    // Alloca is uniform (per-thread stack slots are
                    // identically addressed across the warp).
                    divergent = false;
                } else if (propagates_divergence(inst->opcode())) {
                    for (const auto& op : inst->operands()) {
                        if (is_div(op.get())) {
                            divergent = true;
                            break;
                        }
                    }
                } else {
                    // For terminators / stores we don't propagate to a
                    // result (there isn't one).
                    divergent = false;
                }

                if (divergent) {
                    result.is_divergent[inst.get()] = true;
                    changed = true;
                }
            }
        }
    }

    // Count divergent values
    for (const auto& [v, div] : result.is_divergent) {
        (void)v;
        if (div) ++result.divergent_value_count;
    }

    // Walk conditional branches and mark divergence.
    for (const auto& bb : fn.blocks()) {
        if (!bb) continue;
        for (const auto& inst : bb->instructions()) {
            if (inst->opcode() != ir::Opcode::Br) continue;
            if (inst->num_operands() == 0) continue;  // unconditional
            ++result.total_conditional_branches;
            bool bd = is_div(inst->operand(0).get());
            result.divergent_branches[inst.get()] = bd;
            if (bd) ++result.divergent_branch_count;
        }
    }

    // Identify reconvergence points: for each divergent branch, the
    // immediate post-dominator of its containing block is the
    // reconvergence point. We approximate the IPDOM as the nearest
    // common successor of both branch targets — for the simple
    // "if-then-else-merge" pattern that's the merge block.
    //
    // Algorithm: for each divergent branch in block B with targets
    // T and F, find the set of blocks reachable from T and from F
    // (BFS), and the first block reachable from BOTH is the
    // reconvergence point. Mark it in is_reconvergence.
    auto reachable_from = [&](const ir::BasicBlock* start) {
        std::unordered_set<const ir::BasicBlock*> seen;
        std::vector<const ir::BasicBlock*> queue;
        queue.push_back(start);
        seen.insert(start);
        while (!queue.empty()) {
            const ir::BasicBlock* b = queue.back();
            queue.pop_back();
            for (const auto& s : b->successors()) {
                auto it = bb_by_name.find(s);
                if (it == bb_by_name.end()) continue;
                if (seen.insert(it->second).second) {
                    queue.push_back(it->second);
                }
            }
        }
        return seen;
    };

    for (const auto& bb : fn.blocks()) {
        if (!bb) continue;
        for (const auto& inst : bb->instructions()) {
            if (inst->opcode() != ir::Opcode::Br) continue;
            if (inst->num_operands() == 0) continue;
            auto it = result.divergent_branches.find(inst.get());
            if (it == result.divergent_branches.end() || !it->second) continue;

            const auto& md = inst->metadata();
            auto true_it = md.find("true_bb");
            auto false_it = md.find("false_bb");
            if (true_it == md.end() || false_it == md.end()) continue;

            auto tbb_it = bb_by_name.find(true_it->second);
            auto fbb_it = bb_by_name.find(false_it->second);
            if (tbb_it == bb_by_name.end() || fbb_it == bb_by_name.end()) continue;

            auto t_reach = reachable_from(tbb_it->second);
            // BFS from F; the first node we encounter that is also in
            // t_reach is the IPDOM (modulo BFS ordering — for the
            // if-then-else-merge pattern this is the merge block).
            std::vector<const ir::BasicBlock*> queue;
            std::unordered_set<const ir::BasicBlock*> seen;
            queue.push_back(fbb_it->second);
            seen.insert(fbb_it->second);
            while (!queue.empty()) {
                const ir::BasicBlock* b = queue.back();
                queue.pop_back();
                if (t_reach.count(b) && b != bb.get()) {
                    result.is_reconvergence[b] = true;
                    break;
                }
                for (const auto& s : b->successors()) {
                    auto it2 = bb_by_name.find(s);
                    if (it2 == bb_by_name.end()) continue;
                    if (seen.insert(it2->second).second) {
                        queue.push_back(it2->second);
                    }
                }
            }
        }
    }

    // Total tracked value count (named instructions + named operands).
    std::unordered_set<const ir::Value*> all_values;
    for (const auto& bb : fn.blocks()) {
        if (!bb) continue;
        for (const auto& inst : bb->instructions()) {
            if (inst->has_name()) all_values.insert(inst.get());
            for (const auto& op : inst->operands()) {
                if (op && op->has_name()) all_values.insert(op.get());
            }
        }
    }
    result.total_value_count = all_values.size();

    return result;
}

} // namespace clunk::gpu
