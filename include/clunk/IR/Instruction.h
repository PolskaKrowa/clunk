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
 * Clunk IR Instruction — SSA instructions modelled after LLVM IR.
 */
#include <memory>
#include <string>
#include <vector>
#include <optional>
#include <unordered_map>
#include "clunk/IR/Type.h"
#include "clunk/IR/Value.h"

namespace clunk::ir {

// ── Opcodes ─────────────────────────────────────────────────────────────
enum class Opcode : uint16_t {
    // Terminator instructions
    Ret, Br, Switch, Invoke, Resume, Unreachable,

    // Binary operations (integer)
    Add, Sub, Mul, UDiv, SDiv, URem, SRem,

    // Binary operations (floating point)
    FAdd, FSub, FMul, FDiv, FRem,

    // Bitwise operations
    And, Or, Xor, Shl, LShr, AShr,

    // Memory operations
    Alloca, Load, Store, GetElementPtr, Fence,

    // Conversion operations
    Trunc, ZExt, SExt, FPTrunc, FPExt,
    FPToUI, FPToSI, UIToFP, SIToFP,
    PtrToInt, IntToPtr, BitCast, AddrSpaceCast,

    // Other
    ICmp, FCmp, Phi, Select, Call,
    VAArg, ExtractValue, InsertValue,
    ExtractElement, InsertElement, ShuffleVector,
    LandingPad, SwitchInst
};

// ── Comparison predicates ───────────────────────────────────────────────
enum class CmpPredicate : uint8_t {
    // Integer
    EQ  = 32, NE  = 33,
    UGT = 34, UGE = 35, ULT = 36, ULE = 37,
    SGT = 38, SGE = 39, SLT = 40, SLE = 41,

    // Floating point
    FFalse = 0, FOEQ = 1, FOGT = 2, FOGE = 3,
    FOLT = 4, FOLE = 5, FONE = 6, FORD = 7,
    FUNO = 8, FUEQ = 9, FUGT = 10, FUGE = 11,
    FULT = 12, FULE = 13, FUNE = 14, FTrue = 15
};

// ── Binary operation flags ──────────────────────────────────────────────
struct BinOpFlags {
    bool nuw = false; // No Unsigned Wrap
    bool nsw = false; // No Signed Wrap
    bool exact = false; // No rounding/truncation

    std::string to_string() const {
        std::string s;
        if (nuw) s += " nuw";
        if (nsw) s += " nsw";
        if (exact) s += " exact";
        return s;
    }
};

// ── Instruction class ───────────────────────────────────────────────────
class Instruction final : public Value {
public:
    Instruction(Opcode op, std::shared_ptr<Type> type,
                const std::string& name = "")
        : Value(type, name), opcode_(op) {}

    Opcode opcode() const { return opcode_; }
    bool is_terminator() const;
    bool is_binary_op() const;
    bool is_memory_op() const;
    bool is_cast() const;
    bool is_cmp() const;

    // Operands
    void add_operand(std::shared_ptr<Value> v) { operands_.push_back(v); }
    const std::vector<std::shared_ptr<Value>>& operands() const { return operands_; }
    std::shared_ptr<Value> operand(size_t i) const { return operands_.at(i); }
    size_t num_operands() const { return operands_.size(); }
    void set_operand(size_t i, std::shared_ptr<Value> v) { operands_.at(i) = v; }

    // Metadata
    void set_metadata(const std::string& key, const std::string& val) {
        metadata_[key] = val;
    }
    const std::unordered_map<std::string, std::string>& metadata() const { return metadata_; }

    // Flags for binary operations
    BinOpFlags& binop_flags() { return flags_; }
    const BinOpFlags& binop_flags() const { return flags_; }

    // Alignment for memory ops
    void set_alignment(unsigned align) { alignment_ = align; }
    std::optional<unsigned> alignment() const { return alignment_; }

    // Volatile for memory ops
    void set_volatile(bool v) { volatile_ = v; }
    bool is_volatile() const { return volatile_; }

    // Opcode name string
    static const char* opcode_name(Opcode op);
    std::string to_string() const override;

    // When an Instruction is used as an operand, print just its SSA name
    // (e.g. "%r"), NOT its full definition — otherwise operand rendering
    // recurses into the defining instruction.
    std::string print_as_operand() const override {
        if (has_name()) return "%" + name_;
        return "<unnamed>";
    }

private:
    Opcode opcode_;
    std::vector<std::shared_ptr<Value>> operands_;
    std::unordered_map<std::string, std::string> metadata_;
    BinOpFlags flags_;
    std::optional<unsigned> alignment_;
    bool volatile_ = false;
};

// ── Helper: build specific instructions ─────────────────────────────────
namespace inst {

std::shared_ptr<Instruction> make_ret(std::shared_ptr<Value> val);
std::shared_ptr<Instruction> make_ret_void();
std::shared_ptr<Instruction> make_br(std::shared_ptr<Value> cond,
                                      const std::string& true_bb,
                                      const std::string& false_bb);
std::shared_ptr<Instruction> make_br_uncond(const std::string& dest);
std::shared_ptr<Instruction> make_add(std::shared_ptr<Value> lhs, std::shared_ptr<Value> rhs,
                                       const std::string& name = "", BinOpFlags flags = {});
std::shared_ptr<Instruction> make_sub(std::shared_ptr<Value> lhs, std::shared_ptr<Value> rhs,
                                       const std::string& name = "", BinOpFlags flags = {});
std::shared_ptr<Instruction> make_mul(std::shared_ptr<Value> lhs, std::shared_ptr<Value> rhs,
                                       const std::string& name = "", BinOpFlags flags = {});
std::shared_ptr<Instruction> make_icmp(CmpPredicate pred,
                                        std::shared_ptr<Value> lhs, std::shared_ptr<Value> rhs,
                                        const std::string& name = "");
std::shared_ptr<Instruction> make_alloca(std::shared_ptr<Type> ty,
                                          const std::string& name = "", unsigned align = 0);
std::shared_ptr<Instruction> make_load(std::shared_ptr<Type> loaded_type,
                                        std::shared_ptr<Value> ptr,
                                        const std::string& name = "", unsigned align = 0);
std::shared_ptr<Instruction> make_load(std::shared_ptr<Value> ptr,
                                        const std::string& name = "", unsigned align = 0);
std::shared_ptr<Instruction> make_store(std::shared_ptr<Value> val, std::shared_ptr<Value> ptr,
                                         unsigned align = 0);
std::shared_ptr<Instruction> make_phi(std::shared_ptr<Type> ty,
                                       const std::string& name = "");
std::shared_ptr<Instruction> make_call(std::shared_ptr<Type> ret_type,
                                        const std::string& callee,
                                        std::vector<std::shared_ptr<Value>> args,
                                        const std::string& name = "");
std::shared_ptr<Instruction> make_select(std::shared_ptr<Value> cond,
                                          std::shared_ptr<Value> true_val,
                                          std::shared_ptr<Value> false_val,
                                          const std::string& name = "");
std::shared_ptr<Instruction> make_gep(std::shared_ptr<Type> result_type,
                                       std::shared_ptr<Value> ptr,
                                       std::vector<std::shared_ptr<Value>> indices,
                                       const std::string& name = "");

// ── Vector element operations ────────────────────────────────────────────
// `vec` must be VectorType-typed; the result type is its element type.
std::shared_ptr<Instruction> make_extractelement(std::shared_ptr<Value> vec,
                                                  std::shared_ptr<Value> index,
                                                  const std::string& name = "");
std::shared_ptr<Instruction> make_insertelement(std::shared_ptr<Value> vec,
                                                 std::shared_ptr<Value> elem,
                                                 std::shared_ptr<Value> index,
                                                 const std::string& name = "");
// `mask` is a ConstantVector of i32 lane indices (result lane count =
// mask lane count; index i selects lane i of `lhs` for i < N, lane i-N of
// `rhs` otherwise; -1 = undef lane, LLVM semantics).
std::shared_ptr<Instruction> make_shufflevector(std::shared_ptr<Value> lhs,
                                                 std::shared_ptr<Value> rhs,
                                                 std::shared_ptr<Value> mask,
                                                 const std::string& name = "");
// A generic vector-or-scalar binop with an explicit opcode (add/sub/mul/
// and/or/xor/shl/lshr/ashr/...): result type = lhs type.
std::shared_ptr<Instruction> make_binop(Opcode op,
                                         std::shared_ptr<Value> lhs,
                                         std::shared_ptr<Value> rhs,
                                         const std::string& name = "",
                                         BinOpFlags flags = {});

} // namespace inst

} // namespace clunk::ir
