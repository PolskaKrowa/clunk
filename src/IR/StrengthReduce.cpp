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
 * Clunk StrengthReduce — implementation.
 * See include/clunk/IR/StrengthReduce.h for the design/scope notes.
 */
#include "clunk/IR/StrengthReduce.h"

#include <unordered_map>

#include "clunk/Analysis/KnownBits.h"
#include "clunk/Analysis/PowersOfTwo.h"
#include "clunk/IR/BasicBlock.h"
#include "clunk/IR/Instruction.h"
#include "clunk/IR/Value.h"

namespace clunk::ir {

namespace {

bool is_pow2_or_zero(analysis::Pow2Fact fact) {
    return fact == analysis::Pow2Fact::PowerOfTwo ||
           fact == analysis::Pow2Fact::ZeroOrPowerOfTwo;
}

// Re-resolve every named-value operand across `fn` to the CURRENT
// instruction object with that name. Same idea (and same independent
// re-implementation, per this codebase's convention — see
// DataflowPrune.cpp's copy of the same helper) as needed after
// rebuilding a function block-by-block with a rolling `binding` map:
// other instructions may still hold a shared_ptr to an OLD object that's
// still name-correct but no longer the canonical def.
void remap_operands_by_name(Function& fn) {
    std::unordered_map<std::string, std::shared_ptr<Value>> defs;
    for (auto& block : fn.blocks()) {
        if (!block) continue;
        for (auto& inst : block->instructions()) {
            if (inst && inst->has_name()) defs[inst->name()] = inst;
        }
    }
    for (auto& block : fn.blocks()) {
        if (!block) continue;
        for (auto& inst : block->instructions()) {
            if (!inst) continue;
            for (size_t i = 0; i < inst->num_operands(); ++i) {
                auto op = inst->operand(i);
                if (!op || !op->has_name()) continue;
                auto it = defs.find(op->name());
                if (it != defs.end() && it->second != op) {
                    inst->set_operand(i, it->second);
                }
            }
        }
    }
}

} // namespace

std::shared_ptr<Function> simplify_pow2_strength_reduce(const Function& fn,
                                                          StrengthReduceStats* stats) {
    auto env = analysis::analyse_known_bits(fn);
    if (env.empty()) return nullptr;

    // Quick pre-check: any URem whose divisor is provably a nonzero
    // power of two? Avoids paying for a deep copy + rebuild otherwise.
    bool any = false;
    for (auto& block : fn.blocks()) {
        if (!block) continue;
        for (auto& inst : block->instructions()) {
            if (!inst || inst->opcode() != Opcode::URem || inst->num_operands() != 2) continue;
            if (!inst->has_name()) continue;
            if (is_pow2_or_zero(analysis::operand_pow2_fact(inst->operand(1), env))) {
                any = true;
                break;
            }
        }
        if (any) break;
    }
    if (!any) return nullptr;

    auto work = std::make_shared<Function>(fn.name(), fn.function_type(), fn.linkage());
    for (auto& arg : fn.arguments()) work->add_argument(arg.type, arg.name, arg.attrs);
    for (auto& [k, v] : fn.attributes()) work->set_attribute(k, v);

    size_t rewritten = 0;
    unsigned tmp_id = 0;

    // binding: name -> replacement Value, for operand resolution as we
    // rebuild (same pattern as DataflowPrune.cpp's simplify_known_bits).
    std::unordered_map<std::string, std::shared_ptr<Value>> binding;
    auto resolve = [&](const std::shared_ptr<Value>& v) -> std::shared_ptr<Value> {
        if (v && v->has_name()) {
            auto it = binding.find(v->name());
            if (it != binding.end()) return it->second;
        }
        return v;
    };

    for (auto& block : fn.blocks()) {
        if (!block) continue;
        auto& new_bb = work->add_block(block->name());
        new_bb.clear_predecessors();

        for (auto& inst : block->instructions()) {
            if (!inst) continue;

            if (inst->has_name() && inst->opcode() == Opcode::URem &&
                inst->num_operands() == 2 &&
                is_pow2_or_zero(analysis::operand_pow2_fact(inst->operand(1), env))) {
                auto ity = std::dynamic_pointer_cast<IntegerType>(inst->type());
                if (!ity && inst->type()) {
                    ity = std::make_shared<IntegerType>(
                        static_cast<unsigned>(inst->type()->bit_width()));
                }
                if (ity) {
                    // x urem y  ->  x and (y - 1). See header comment for
                    // the soundness argument (exact for y a power of two
                    // AND exact for y == 0, since this project's SMT
                    // verifier's bvurem-by-zero semantics equal x, same
                    // as x and 0xFFFF...F). `y` is resolved first so a
                    // divisor already folded/renamed earlier this same
                    // pass is picked up correctly.
                    auto dividend = resolve(inst->operand(0));
                    auto divisor = resolve(inst->operand(1));
                    auto one = std::make_shared<ConstantInt>(ity, int64_t(1));

                    auto mask = std::make_shared<Instruction>(
                        Opcode::Sub, inst->type(),
                        inst->name() + ".pow2mask" + std::to_string(tmp_id++));
                    mask->add_operand(divisor);
                    mask->add_operand(one);
                    new_bb.add_instruction(mask);

                    auto andi = std::make_shared<Instruction>(
                        Opcode::And, inst->type(), inst->name());
                    andi->add_operand(dividend);
                    andi->add_operand(mask);
                    new_bb.add_instruction(andi);

                    ++rewritten;
                    continue;  // don't emit the original urem
                }
            }

            // Rebuild the instruction verbatim with resolved operands.
            auto new_inst = std::make_shared<Instruction>(inst->opcode(), inst->type(), inst->name());
            for (auto& op : inst->operands()) new_inst->add_operand(resolve(op));
            for (auto& [k, v] : inst->metadata()) new_inst->set_metadata(k, v);
            new_inst->binop_flags() = inst->binop_flags();
            if (inst->alignment()) new_inst->set_alignment(inst->alignment().value());
            new_inst->set_volatile(inst->is_volatile());
            new_bb.add_instruction(new_inst);
        }
    }

    if (rewritten == 0) return nullptr;
    remap_operands_by_name(*work);
    work->compute_predecessors();
    if (stats) stats->urem_to_and += rewritten;
    return work;
}

} // namespace clunk::ir