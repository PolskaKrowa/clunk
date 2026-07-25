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
 * Clunk IR Clone — canonical deep-copy helper for ir::Function.
 *
 * The single canonical deep-copy implementation used by search engines
 * and PatternLibrary::apply. The clone is a structural deep copy: a
 * brand-new Function object with fresh BasicBlock / Instruction shared_ptrs.
 * The operand shared_ptrs are copied by value (shallow) — that is safe for
 * our mutation model because operands are immutable Values (ConstantInt /
 * argument refs) and we rebuild instructions rather than mutate them in
 * place. For in-place mutation that needs undo, see
 * clunk/Search/MutationScope.h.
 */
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include "clunk/IR/Function.h"
#include "clunk/IR/BasicBlock.h"
#include "clunk/IR/Instruction.h"

namespace clunk::ir {

// Deep-copy a Function. Returns a fresh shared_ptr<Function> with the same
// name, type, linkage, arguments, attributes, blocks, and instructions.
// Metadata, binop flags, alignment, and volatile markers are preserved.
inline std::shared_ptr<Function> deep_copy_function(const Function& src) {
    auto copy = std::make_shared<Function>(src.name(), src.function_type(), src.linkage());

    // Copy arguments (type + name + attrs). Attrs matter: the memory
    // optimiser's alias oracle reads `noalias` from Argument::attrs, so
    // losing them in a clone would silently weaken (never unsound, but
    // needlessly conservative) every downstream alias query.
    for (auto& arg : src.arguments()) {
        copy->add_argument(arg.type, arg.name, arg.attrs);
    }

    // Copy function attributes
    for (auto& [k, v] : src.attributes()) {
        copy->set_attribute(k, v);
    }

    // Copy basic blocks and their instructions
    for (auto& block : src.blocks()) {
        if (!block) continue;
        auto& new_bb = copy->add_block(block->name());
        new_bb.clear_predecessors();
        for (auto& inst : block->instructions()) {
            if (!inst) continue;
            auto new_inst = std::make_shared<Instruction>(
                inst->opcode(), inst->type(), inst->name());
            // Copy operands (shared_ptr copy — Values are logically
            // immutable; instruction-result operands are remapped below).
            for (auto& op : inst->operands()) {
                new_inst->add_operand(op);
            }
            // Copy metadata
            for (auto& [k, v] : inst->metadata()) {
                new_inst->set_metadata(k, v);
            }
            // Copy binop flags
            new_inst->binop_flags() = inst->binop_flags();
            // Copy alignment
            if (inst->alignment()) {
                new_inst->set_alignment(inst->alignment().value());
            }
            // Copy volatile
            new_inst->set_volatile(inst->is_volatile());
            new_bb.add_instruction(new_inst);
        }
    }

    // Remap instruction-result operands onto the CLONED instructions.
    // Without this every copied operand kept pointing into the SOURCE
    // function, which broke any consumer that resolves values by object
    // identity — most notably the Interpreter (Context keys bindings by
    // pointer), which returned nullopt for every cloned function with a
    // def-use chain longer than one instruction. Two passes so phis can
    // reference definitions that appear later (loop back-edges).
    {
        std::unordered_map<std::string, std::shared_ptr<Value>> defs;
        for (auto& block : copy->blocks()) {
            for (auto& inst : block->instructions()) {
                if (inst && inst->has_name()) defs[inst->name()] = inst;
            }
        }
        for (auto& block : copy->blocks()) {
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

    return copy;
}

// Convenience: clone via a shared_ptr source.
inline std::shared_ptr<Function> deep_copy_function(const std::shared_ptr<Function>& src) {
    if (!src) return nullptr;
    return deep_copy_function(*src);
}

// ── validate_function ─────────────────────────────────────────────────────
// Walks the function in block order, tracking defined SSA value names
// (instruction results + function arguments). Every operand that is a
// named value must be defined BEFORE its use (by an earlier instruction
// in the same block, or by a function argument, or — for PHI nodes — by
// an instruction in a predecessor block; we approximate the PHI case by
// allowing any name previously seen anywhere, which is a conservative
// over-approximation that still catches cases where SSA names from
// another function are referenced but never defined.
//
// Returns true iff the function is well-formed (no dangling references).
// Used by EvolutionarySearch::crossover and PatternLibrary::apply to
// reject invalid children.
inline bool validate_function(const Function& fn) {
    std::unordered_set<std::string> defined;
    // Block names are control-flow targets, not SSA values. In the canonical
    // representation branch labels live in instruction metadata, but some
    // paths carry them as string-named operands (see BasicBlock::successors);
    // either way a terminator operand that names a block is a label, not a
    // value use, and must not be checked against `defined`.
    std::unordered_set<std::string> block_names;
    for (auto& block : fn.blocks()) {
        if (block) block_names.insert(block->name());
    }
    // Function arguments are defined at function entry.
    for (auto& arg : fn.arguments()) {
        if (!arg.name.empty()) defined.insert(arg.name);
    }

    // Names defined ANYWHERE in the function — phis on loop headers
    // legitimately reference back-edge values defined later in block
    // order, so their operands are checked against this set instead of
    // the walked `defined` set.
    std::unordered_set<std::string> defined_anywhere;
    for (auto& block : fn.blocks()) {
        if (!block) continue;
        for (auto& inst : block->instructions()) {
            if (inst && inst->has_name()) defined_anywhere.insert(inst->name());
        }
    }
    for (auto& arg : fn.arguments()) {
        if (!arg.name.empty()) defined_anywhere.insert(arg.name);
    }

    for (auto& block : fn.blocks()) {
        if (!block) continue;
        for (auto& inst : block->instructions()) {
            if (!inst) continue;
            // Phi operands: defined anywhere is enough (see above).
            if (inst->opcode() == Opcode::Phi) {
                for (auto& op : inst->operands()) {
                    if (op && op->has_name() && !block_names.count(op->name()) &&
                        !defined_anywhere.count(op->name())) {
                        return false;
                    }
                }
                if (inst->has_name()) defined.insert(inst->name());
                continue;
            }
            // Check operands are defined (skip constants, which have no name).
            for (auto& op : inst->operands()) {
                if (!op) continue;
                if (!op->has_name()) continue;  // constant or unnamed
                // For terminators, skip label operands (block names) but
                // still validate genuine VALUE operands — the returned value
                // of a `ret`, a branch condition, a switch selector. An
                // undefined operand on a terminator must be rejected here to
                // prevent invalid IR from propagating through scoring and
                // verification.
                if (inst->is_terminator() && block_names.count(op->name())) {
                    continue;
                }
                if (defined.find(op->name()) == defined.end()) {
                    return false;  // use before def
                }
            }
            // Define this instruction's result name (if any).
            if (inst->has_name()) {
                defined.insert(inst->name());
            }
        }
    }
    return true;
}

} // namespace clunk::ir
