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
 * Clunk IR Builder — convenience for constructing IR programmatically.
 */
#include <memory>
#include <string>
#include "clunk/IR/Type.h"
#include "clunk/IR/Value.h"
#include "clunk/IR/Instruction.h"
#include "clunk/IR/BasicBlock.h"
#include "clunk/IR/Function.h"
#include "clunk/IR/Module.h"

namespace clunk::ir {

class IRBuilder {
public:
    explicit IRBuilder(TypeContext& ctx) : ctx_(ctx), current_bb_(nullptr) {}

    // Position
    void set_insert_point(BasicBlock* bb) { current_bb_ = bb; }
    BasicBlock* insert_point() const { return current_bb_; }

    // ── Terminator instructions ─────────────────────────────────────────
    std::shared_ptr<Value> create_ret(std::shared_ptr<Value> val) {
        auto inst = inst::make_ret(val);
        current_bb_->add_instruction(inst);
        return inst;
    }

    std::shared_ptr<Value> create_ret_void() {
        auto inst = inst::make_ret_void();
        current_bb_->add_instruction(inst);
        return inst;
    }

    std::shared_ptr<Value> create_br(const std::string& dest) {
        auto inst = inst::make_br_uncond(dest);
        current_bb_->add_instruction(inst);
        return inst;
    }

    std::shared_ptr<Value> create_cond_br(std::shared_ptr<Value> cond,
                                           const std::string& true_bb,
                                           const std::string& false_bb) {
        auto inst = inst::make_br(cond, true_bb, false_bb);
        current_bb_->add_instruction(inst);
        return inst;
    }

    // ── Binary integer operations ───────────────────────────────────────
    std::shared_ptr<Value> create_add(std::shared_ptr<Value> lhs, std::shared_ptr<Value> rhs,
                                       const std::string& name = "") {
        auto inst = inst::make_add(lhs, rhs, name);
        current_bb_->add_instruction(inst);
        return inst;
    }

    std::shared_ptr<Value> create_sub(std::shared_ptr<Value> lhs, std::shared_ptr<Value> rhs,
                                       const std::string& name = "") {
        auto inst = inst::make_sub(lhs, rhs, name);
        current_bb_->add_instruction(inst);
        return inst;
    }

    std::shared_ptr<Value> create_mul(std::shared_ptr<Value> lhs, std::shared_ptr<Value> rhs,
                                       const std::string& name = "") {
        auto inst = inst::make_mul(lhs, rhs, name);
        current_bb_->add_instruction(inst);
        return inst;
    }

    // ── Comparison ──────────────────────────────────────────────────────
    std::shared_ptr<Value> create_icmp(CmpPredicate pred,
                                        std::shared_ptr<Value> lhs, std::shared_ptr<Value> rhs,
                                        const std::string& name = "") {
        auto inst = inst::make_icmp(pred, lhs, rhs, name);
        current_bb_->add_instruction(inst);
        return inst;
    }

    // ── Memory operations ───────────────────────────────────────────────
    std::shared_ptr<Value> create_alloca(std::shared_ptr<Type> ty,
                                          const std::string& name = "", unsigned align = 0) {
        auto inst = inst::make_alloca(ty, name, align);
        current_bb_->add_instruction(inst);
        return inst;
    }

    std::shared_ptr<Value> create_load(std::shared_ptr<Value> ptr,
                                        const std::string& name = "", unsigned align = 0) {
        auto inst = inst::make_load(ptr, name, align);
        current_bb_->add_instruction(inst);
        return inst;
    }

    std::shared_ptr<Value> create_store(std::shared_ptr<Value> val, std::shared_ptr<Value> ptr,
                                         unsigned align = 0) {
        auto inst = inst::make_store(val, ptr, align);
        current_bb_->add_instruction(inst);
        return inst;
    }

    // ── Other ───────────────────────────────────────────────────────────
    std::shared_ptr<Value> create_phi(std::shared_ptr<Type> ty,
                                       const std::string& name = "") {
        auto inst = inst::make_phi(ty, name);
        current_bb_->add_instruction(inst);
        return inst;
    }

    std::shared_ptr<Value> create_call(std::shared_ptr<Type> ret_type,
                                        const std::string& callee,
                                        std::vector<std::shared_ptr<Value>> args,
                                        const std::string& name = "") {
        auto inst = inst::make_call(ret_type, callee, std::move(args), name);
        current_bb_->add_instruction(inst);
        return inst;
    }

    std::shared_ptr<Value> create_select(std::shared_ptr<Value> cond,
                                          std::shared_ptr<Value> true_val,
                                          std::shared_ptr<Value> false_val,
                                          const std::string& name = "") {
        auto inst = inst::make_select(cond, true_val, false_val, name);
        current_bb_->add_instruction(inst);
        return inst;
    }

    // ── Constants ───────────────────────────────────────────────────────
    std::shared_ptr<ConstantInt> get_int32(int32_t val) {
        return ConstantInt::get(ctx_, val, 32);
    }
    std::shared_ptr<ConstantInt> get_int64(int64_t val) {
        return ConstantInt::get(ctx_, val, 64);
    }
    std::shared_ptr<ConstantInt> get_int1(bool val) {
        return ConstantInt::get(ctx_, val ? 1 : 0, 1);
    }

private:
    TypeContext& ctx_;
    BasicBlock* current_bb_;
};

} // namespace clunk::ir
