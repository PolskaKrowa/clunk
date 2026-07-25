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
 * Clunk IR BasicBlock — a sequence of instructions ending with a terminator.
 */
#include <memory>
#include <string>
#include <vector>
#include "clunk/IR/Instruction.h"

namespace clunk::ir {

class BasicBlock {
public:
    explicit BasicBlock(const std::string& name) : name_(name) {}

    const std::string& name() const { return name_; }

    // Instruction management
    void add_instruction(std::shared_ptr<Instruction> inst) {
        instrs_.push_back(inst);
    }

    void insert_instruction(size_t pos, std::shared_ptr<Instruction> inst) {
        if (pos <= instrs_.size()) {
            instrs_.insert(instrs_.begin() + pos, inst);
        }
    }

    void remove_instruction(size_t pos) {
        if (pos < instrs_.size()) {
            instrs_.erase(instrs_.begin() + pos);
        }
    }

    void replace_instruction(size_t pos, std::shared_ptr<Instruction> inst) {
        if (pos < instrs_.size()) {
            instrs_[pos] = inst;
        }
    }

    const std::vector<std::shared_ptr<Instruction>>& instructions() const { return instrs_; }
    std::vector<std::shared_ptr<Instruction>>& instructions() { return instrs_; }

    size_t size() const { return instrs_.size(); }
    bool empty() const { return instrs_.empty(); }

    std::shared_ptr<Instruction> instruction(size_t i) const { return instrs_.at(i); }

    // Get the terminator instruction (last instruction if it is a terminator)
    std::shared_ptr<Instruction> terminator() const {
        if (instrs_.empty()) return nullptr;
        auto& last = instrs_.back();
        return last->is_terminator() ? last : nullptr;
    }

    // Check if this block is well-formed (ends with a terminator)
    bool is_well_formed() const {
        return !instrs_.empty() && instrs_.back()->is_terminator();
    }

    // Successor blocks (from branch instructions)
    std::vector<std::string> successors() const;

    // Predecessor blocks (populated by Function)
    std::vector<std::string> predecessors() const { return preds_; }
    void add_predecessor(const std::string& bb_name) { preds_.push_back(bb_name); }
    void clear_predecessors() { preds_.clear(); }

    std::string to_string() const;

private:
    std::string name_;
    std::vector<std::shared_ptr<Instruction>> instrs_;
    std::vector<std::string> preds_;
};

} // namespace clunk::ir
