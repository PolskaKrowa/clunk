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
 * Clunk IR Instruction — implementation of SSA instruction methods and
 * the inst::make_* factory functions.
 */
#include "clunk/IR/Instruction.h"

#include <sstream>

namespace clunk::ir {

// ── Instruction classification ──────────────────────────────────────────────

bool Instruction::is_terminator() const {
    switch (opcode_) {
    case Opcode::Ret:
    case Opcode::Br:
    case Opcode::Switch:
    case Opcode::Invoke:
    case Opcode::Resume:
    case Opcode::Unreachable:
        return true;
    default:
        return false;
    }
}

bool Instruction::is_binary_op() const {
    switch (opcode_) {
    // Integer binary
    case Opcode::Add:
    case Opcode::Sub:
    case Opcode::Mul:
    case Opcode::UDiv:
    case Opcode::SDiv:
    case Opcode::URem:
    case Opcode::SRem:
    // Floating-point binary
    case Opcode::FAdd:
    case Opcode::FSub:
    case Opcode::FMul:
    case Opcode::FDiv:
    case Opcode::FRem:
    // Bitwise / shift
    case Opcode::And:
    case Opcode::Or:
    case Opcode::Xor:
    case Opcode::Shl:
    case Opcode::LShr:
    case Opcode::AShr:
        return true;
    default:
        return false;
    }
}

bool Instruction::is_memory_op() const {
    switch (opcode_) {
    case Opcode::Alloca:
    case Opcode::Load:
    case Opcode::Store:
    case Opcode::GetElementPtr:
    case Opcode::Fence:
        return true;
    default:
        return false;
    }
}

bool Instruction::is_cast() const {
    switch (opcode_) {
    case Opcode::Trunc:
    case Opcode::ZExt:
    case Opcode::SExt:
    case Opcode::FPTrunc:
    case Opcode::FPExt:
    case Opcode::FPToUI:
    case Opcode::FPToSI:
    case Opcode::UIToFP:
    case Opcode::SIToFP:
    case Opcode::PtrToInt:
    case Opcode::IntToPtr:
    case Opcode::BitCast:
    case Opcode::AddrSpaceCast:
        return true;
    default:
        return false;
    }
}

bool Instruction::is_cmp() const {
    return opcode_ == Opcode::ICmp || opcode_ == Opcode::FCmp;
}

// ── Opcode name ─────────────────────────────────────────────────────────────

const char* Instruction::opcode_name(Opcode op) {
    switch (op) {
    // Terminators
    case Opcode::Ret:         return "ret";
    case Opcode::Br:          return "br";
    case Opcode::Switch:      return "switch";
    case Opcode::Invoke:      return "invoke";
    case Opcode::Resume:      return "resume";
    case Opcode::Unreachable: return "unreachable";
    // Integer binary
    case Opcode::Add:  return "add";
    case Opcode::Sub:  return "sub";
    case Opcode::Mul:  return "mul";
    case Opcode::UDiv: return "udiv";
    case Opcode::SDiv: return "sdiv";
    case Opcode::URem: return "urem";
    case Opcode::SRem: return "srem";
    // Floating-point binary
    case Opcode::FAdd: return "fadd";
    case Opcode::FSub: return "fsub";
    case Opcode::FMul: return "fmul";
    case Opcode::FDiv: return "fdiv";
    case Opcode::FRem: return "frem";
    // Bitwise / shift
    case Opcode::And:  return "and";
    case Opcode::Or:   return "or";
    case Opcode::Xor:  return "xor";
    case Opcode::Shl:  return "shl";
    case Opcode::LShr: return "lshr";
    case Opcode::AShr: return "ashr";
    // Memory
    case Opcode::Alloca:       return "alloca";
    case Opcode::Load:         return "load";
    case Opcode::Store:        return "store";
    case Opcode::GetElementPtr: return "getelementptr";
    case Opcode::Fence:        return "fence";
    // Casts
    case Opcode::Trunc:         return "trunc";
    case Opcode::ZExt:          return "zext";
    case Opcode::SExt:          return "sext";
    case Opcode::FPTrunc:       return "fptrunc";
    case Opcode::FPExt:         return "fpext";
    case Opcode::FPToUI:        return "fptoui";
    case Opcode::FPToSI:        return "fptosi";
    case Opcode::UIToFP:        return "uitofp";
    case Opcode::SIToFP:        return "sitofp";
    case Opcode::PtrToInt:      return "ptrtoint";
    case Opcode::IntToPtr:      return "inttoptr";
    case Opcode::BitCast:       return "bitcast";
    case Opcode::AddrSpaceCast: return "addrspacecast";
    // Other
    case Opcode::ICmp:           return "icmp";
    case Opcode::FCmp:           return "fcmp";
    case Opcode::Phi:            return "phi";
    case Opcode::Select:         return "select";
    case Opcode::Call:           return "call";
    case Opcode::VAArg:          return "va_arg";
    case Opcode::ExtractValue:   return "extractvalue";
    case Opcode::InsertValue:    return "insertvalue";
    case Opcode::ExtractElement: return "extractelement";
    case Opcode::InsertElement:  return "insertelement";
    case Opcode::ShuffleVector:  return "shufflevector";
    case Opcode::LandingPad:     return "landingpad";
    case Opcode::SwitchInst:     return "switch";
    }
    return "<unknown>";
}

// ── Comparison predicate name helper ────────────────────────────────────────

namespace {

const char* cmp_predicate_name(CmpPredicate pred) {
    switch (pred) {
    // Integer predicates
    case CmpPredicate::EQ:  return "eq";
    case CmpPredicate::NE:  return "ne";
    case CmpPredicate::UGT: return "ugt";
    case CmpPredicate::UGE: return "uge";
    case CmpPredicate::ULT: return "ult";
    case CmpPredicate::ULE: return "ule";
    case CmpPredicate::SGT: return "sgt";
    case CmpPredicate::SGE: return "sge";
    case CmpPredicate::SLT: return "slt";
    case CmpPredicate::SLE: return "sle";
    // Floating-point predicates
    case CmpPredicate::FFalse: return "false";
    case CmpPredicate::FOEQ:   return "oeq";
    case CmpPredicate::FOGT:   return "ogt";
    case CmpPredicate::FOGE:   return "oge";
    case CmpPredicate::FOLT:   return "olt";
    case CmpPredicate::FOLE:   return "ole";
    case CmpPredicate::FONE:   return "one";
    case CmpPredicate::FORD:   return "ord";
    case CmpPredicate::FUNO:   return "uno";
    case CmpPredicate::FUEQ:   return "ueq";
    case CmpPredicate::FUGT:   return "ugt";
    case CmpPredicate::FUGE:   return "uge";
    case CmpPredicate::FULT:   return "ult";
    case CmpPredicate::FULE:   return "ule";
    case CmpPredicate::FUNE:   return "une";
    case CmpPredicate::FTrue:  return "true";
    }
    return "<unknown>";
}

// Retrieve the CmpPredicate stored in metadata under key "pred".
CmpPredicate get_cmp_predicate(const Instruction& inst) {
    auto it = inst.metadata().find("pred");
    if (it != inst.metadata().end()) {
        return static_cast<CmpPredicate>(std::stoul(it->second));
    }
    return CmpPredicate::EQ; // fallback
}

// Helper: format an alignment suffix like ", align 8" if alignment is set.
std::string align_suffix(const Instruction& inst) {
    if (inst.alignment().has_value() && inst.alignment().value() != 0) {
        return ", align " + std::to_string(inst.alignment().value());
    }
    return "";
}

} // anonymous namespace

// ── Instruction::to_string() ────────────────────────────────────────────────

std::string Instruction::to_string() const {
    std::string s;
    s.reserve(64);

    // -- Terminators --
    if (opcode_ == Opcode::Ret) {
        if (num_operands() == 0) {
            s.append("ret void");
        } else {
            s.append("ret ");
            s.append(type_->to_string());
            s += ' ';
            s.append(operands_[0]->print_as_operand());
        }
        return s;
    }

    if (opcode_ == Opcode::Br) {
        if (num_operands() == 0) {
            // Unconditional branch — target stored in metadata
            auto it = metadata_.find("dest_bb");
            s.append("br label %");
            s.append(it != metadata_.end() ? it->second : std::string("<unknown>"));
        } else {
            // Conditional branch
            auto it_t = metadata_.find("true_bb");
            auto it_f = metadata_.find("false_bb");
            s.append("br ");
            s.append(operands_[0]->type()->to_string());
            s += ' ';
            s.append(operands_[0]->print_as_operand());
            s.append(", label %");
            s.append(it_t != metadata_.end() ? it_t->second : std::string("<unknown>"));
            s.append(", label %");
            s.append(it_f != metadata_.end() ? it_f->second : std::string("<unknown>"));
        }
        return s;
    }

    if (opcode_ == Opcode::Unreachable) {
        return "unreachable";
    }

    if (opcode_ == Opcode::Switch) {
        s.append("switch ");
        s.append(operands_[0]->type()->to_string());
        s += ' ';
        s.append(operands_[0]->print_as_operand());
        s.append(", label %");
        s.append(metadata_.count("default_bb") ? metadata_.at("default_bb") : std::string("<unknown>"));
        s.append(" [");
        // remaining operands encode cases; simplified representation
        s.append(" ]");
        return s;
    }

    if (opcode_ == Opcode::Resume) {
        s.append("resume ");
        s.append(type_->to_string());
        s += ' ';
        s.append(operands_[0]->print_as_operand());
        return s;
    }

    if (opcode_ == Opcode::Invoke) {
        auto it = metadata_.find("callee");
        s += '%';
        s.append(name_);
        s.append(" = invoke ");
        s.append(type_->to_string());
        s.append(" @");
        s.append(it != metadata_.end() ? it->second : std::string("<unknown>"));
        s += '(';
        for (size_t i = 0; i < num_operands(); ++i) {
            if (i > 0) s.append(", ");
            s.append(operands_[i]->type()->to_string());
            s += ' ';
            s.append(operands_[i]->print_as_operand());
        }
        s += ')';
        return s;
    }

    // -- Binary operations --
    if (is_binary_op()) {
        s += '%';
        s.append(name_);
        s.append(" = ");
        s.append(opcode_name(opcode_));
        s.append(flags_.to_string());
        s += ' ';
        s.append(type_->to_string());
        s += ' ';
        s.append(operands_[0]->print_as_operand());
        s.append(", ");
        s.append(operands_[1]->print_as_operand());
        return s;
    }

    // -- Comparison operations --
    if (opcode_ == Opcode::ICmp) {
        auto pred = get_cmp_predicate(*this);
        s += '%';
        s.append(name_);
        s.append(" = icmp ");
        s.append(cmp_predicate_name(pred));
        s += ' ';
        s.append(operands_[0]->type()->to_string());
        s += ' ';
        s.append(operands_[0]->print_as_operand());
        s.append(", ");
        s.append(operands_[1]->print_as_operand());
        return s;
    }

    if (opcode_ == Opcode::FCmp) {
        auto pred = get_cmp_predicate(*this);
        s += '%';
        s.append(name_);
        s.append(" = fcmp ");
        s.append(cmp_predicate_name(pred));
        s += ' ';
        s.append(operands_[0]->type()->to_string());
        s += ' ';
        s.append(operands_[0]->print_as_operand());
        s.append(", ");
        s.append(operands_[1]->print_as_operand());
        return s;
    }

    // -- Cast operations --
    if (is_cast()) {
        s += '%';
        s.append(name_);
        s.append(" = ");
        s.append(opcode_name(opcode_));
        s += ' ';
        s.append(operands_[0]->type()->to_string());
        s += ' ';
        s.append(operands_[0]->print_as_operand());
        s.append(" to ");
        s.append(type_->to_string());
        return s;
    }

    // -- Memory operations --
    if (opcode_ == Opcode::Alloca) {
        // Result type is a pointer; allocated type is the pointee.
        auto* ptr_ty = static_cast<const PointerType*>(type_.get());
        s += '%';
        s.append(name_);
        s.append(" = alloca ");
        s.append(ptr_ty->pointee()->to_string());
        s.append(align_suffix(*this));
        return s;
    }

    if (opcode_ == Opcode::Load) {
        s += '%';
        s.append(name_);
        s.append(" = load ");
        s.append(type_->to_string());
        s.append(", ");
        s.append(operands_[0]->type()->to_string());
        s += ' ';
        s.append(operands_[0]->print_as_operand());
        s.append(align_suffix(*this));
        if (volatile_) s.append(", volatile");
        return s;
    }

    if (opcode_ == Opcode::Store) {
        s.append("store ");
        s.append(operands_[0]->type()->to_string());
        s += ' ';
        s.append(operands_[0]->print_as_operand());
        s.append(", ");
        s.append(operands_[1]->type()->to_string());
        s += ' ';
        s.append(operands_[1]->print_as_operand());
        s.append(align_suffix(*this));
        if (volatile_) s.append(", volatile");
        return s;
    }

    if (opcode_ == Opcode::GetElementPtr) {
        // operand 0 = pointer, rest = indices
        auto* ptr_ty = static_cast<const PointerType*>(operands_[0]->type().get());
        s += '%';
        s.append(name_);
        s.append(" = getelementptr ");
        s.append(ptr_ty->pointee()->to_string());
        s.append(", ");
        s.append(operands_[0]->type()->to_string());
        s += ' ';
        s.append(operands_[0]->print_as_operand());
        for (size_t i = 1; i < num_operands(); ++i) {
            s.append(", ");
            s.append(operands_[i]->type()->to_string());
            s += ' ';
            s.append(operands_[i]->print_as_operand());
        }
        return s;
    }

    if (opcode_ == Opcode::Fence) {
        s.append("fence");
        auto it = metadata_.find("ordering");
        if (it != metadata_.end()) {
            s += ' ';
            s.append(it->second);
        }
        return s;
    }

    // -- Phi --
    if (opcode_ == Opcode::Phi) {
        s += '%';
        s.append(name_);
        s.append(" = phi ");
        s.append(type_->to_string());
        // Block names stored in metadata "phi_blocks" as comma-separated list.
        // Operands are the incoming values in matching order.
        auto it = metadata_.find("phi_blocks");
        std::vector<std::string> blocks;
        if (it != metadata_.end()) {
            std::istringstream bs(it->second);
            std::string b;
            while (std::getline(bs, b, ',')) {
                blocks.push_back(b);
            }
        }
        for (size_t i = 0; i < num_operands(); ++i) {
            if (i > 0) s += ',';
            s.append(" [ ");
            s.append(operands_[i]->print_as_operand());
            s.append(", %");
            s.append(i < blocks.size() ? blocks[i] : std::string("<unknown>"));
            s.append(" ]");
        }
        return s;
    }

    // -- Select --
    if (opcode_ == Opcode::Select) {
        s += '%';
        s.append(name_);
        s.append(" = select ");
        s.append(operands_[0]->type()->to_string());
        s += ' ';
        s.append(operands_[0]->print_as_operand());
        s.append(", ");
        s.append(operands_[1]->type()->to_string());
        s += ' ';
        s.append(operands_[1]->print_as_operand());
        s.append(", ");
        s.append(operands_[2]->type()->to_string());
        s += ' ';
        s.append(operands_[2]->print_as_operand());
        return s;
    }

    // -- Call --
    if (opcode_ == Opcode::Call) {
        auto it = metadata_.find("callee");
        s += '%';
        s.append(name_);
        s.append(" = call ");
        s.append(type_->to_string());
        s.append(" @");
        s.append(it != metadata_.end() ? it->second : std::string("<unknown>"));
        s += '(';
        for (size_t i = 0; i < num_operands(); ++i) {
            if (i > 0) s.append(", ");
            s.append(operands_[i]->type()->to_string());
            s += ' ';
            s.append(operands_[i]->print_as_operand());
        }
        s += ')';
        return s;
    }

    // -- VAArg --
    if (opcode_ == Opcode::VAArg) {
        s += '%';
        s.append(name_);
        s.append(" = va_arg ");
        s.append(operands_[0]->type()->to_string());
        s += ' ';
        s.append(operands_[0]->print_as_operand());
        s.append(", ");
        s.append(type_->to_string());
        return s;
    }

    // -- ExtractValue / InsertValue --
    if (opcode_ == Opcode::ExtractValue) {
        s += '%';
        s.append(name_);
        s.append(" = extractvalue ");
        s.append(operands_[0]->type()->to_string());
        s += ' ';
        s.append(operands_[0]->print_as_operand());
        auto it = metadata_.find("indices");
        if (it != metadata_.end()) {
            s.append(", ");
            s.append(it->second);
        }
        return s;
    }

    if (opcode_ == Opcode::InsertValue) {
        s += '%';
        s.append(name_);
        s.append(" = insertvalue ");
        s.append(operands_[0]->type()->to_string());
        s += ' ';
        s.append(operands_[0]->print_as_operand());
        s.append(", ");
        s.append(operands_[1]->type()->to_string());
        s += ' ';
        s.append(operands_[1]->print_as_operand());
        auto it = metadata_.find("indices");
        if (it != metadata_.end()) {
            s.append(", ");
            s.append(it->second);
        }
        return s;
    }

    // -- ExtractElement / InsertElement --
    if (opcode_ == Opcode::ExtractElement) {
        s += '%';
        s.append(name_);
        s.append(" = extractelement ");
        s.append(operands_[0]->type()->to_string());
        s += ' ';
        s.append(operands_[0]->print_as_operand());
        s.append(", ");
        s.append(operands_[1]->type()->to_string());
        s += ' ';
        s.append(operands_[1]->print_as_operand());
        return s;
    }

    if (opcode_ == Opcode::InsertElement) {
        s += '%';
        s.append(name_);
        s.append(" = insertelement ");
        s.append(operands_[0]->type()->to_string());
        s += ' ';
        s.append(operands_[0]->print_as_operand());
        s.append(", ");
        s.append(operands_[1]->type()->to_string());
        s += ' ';
        s.append(operands_[1]->print_as_operand());
        s.append(", ");
        s.append(operands_[2]->type()->to_string());
        s += ' ';
        s.append(operands_[2]->print_as_operand());
        return s;
    }

    // -- ShuffleVector --
    if (opcode_ == Opcode::ShuffleVector) {
        s += '%';
        s.append(name_);
        s.append(" = shufflevector ");
        s.append(operands_[0]->type()->to_string());
        s += ' ';
        s.append(operands_[0]->print_as_operand());
        s.append(", ");
        s.append(operands_[1]->type()->to_string());
        s += ' ';
        s.append(operands_[1]->print_as_operand());
        s.append(", ");
        s.append(operands_[2]->type()->to_string());
        s += ' ';
        s.append(operands_[2]->print_as_operand());
        return s;
    }

    // -- LandingPad --
    if (opcode_ == Opcode::LandingPad) {
        s += '%';
        s.append(name_);
        s.append(" = landingpad ");
        s.append(type_->to_string());
        return s;
    }

    // -- SwitchInst (detailed switch) --
    if (opcode_ == Opcode::SwitchInst) {
        s += '%';
        s.append(name_);
        s.append(" = switch ");
        s.append(operands_[0]->type()->to_string());
        s += ' ';
        s.append(operands_[0]->print_as_operand());
        return s;
    }

    // -- Fallback for any unhandled opcode --
    s += '%';
    s.append(name_);
    s.append(" = ");
    s.append(opcode_name(opcode_));
    s.append(" ...");
    return s;
}

// ── Factory functions ───────────────────────────────────────────────────────
namespace inst {

std::shared_ptr<Instruction> make_ret(std::shared_ptr<Value> val) {
    auto inst = std::make_shared<Instruction>(Opcode::Ret, val->type());
    inst->add_operand(val);
    return inst;
}

std::shared_ptr<Instruction> make_ret_void() {
    static const auto void_ty = std::make_shared<VoidType>();
    return std::make_shared<Instruction>(Opcode::Ret, void_ty);
}

std::shared_ptr<Instruction> make_br(std::shared_ptr<Value> cond,
                                      const std::string& true_bb,
                                      const std::string& false_bb) {
    auto inst = std::make_shared<Instruction>(Opcode::Br, cond->type());
    inst->add_operand(cond);
    inst->set_metadata("true_bb", true_bb);
    inst->set_metadata("false_bb", false_bb);
    return inst;
}

std::shared_ptr<Instruction> make_br_uncond(const std::string& dest) {
    static const auto void_ty = std::make_shared<VoidType>();
    auto inst = std::make_shared<Instruction>(Opcode::Br, void_ty);
    inst->set_metadata("dest_bb", dest);
    return inst;
}

std::shared_ptr<Instruction> make_add(std::shared_ptr<Value> lhs,
                                       std::shared_ptr<Value> rhs,
                                       const std::string& name,
                                       BinOpFlags flags) {
    auto inst = std::make_shared<Instruction>(Opcode::Add, lhs->type(), name);
    inst->add_operand(lhs);
    inst->add_operand(rhs);
    inst->binop_flags() = flags;
    return inst;
}

std::shared_ptr<Instruction> make_sub(std::shared_ptr<Value> lhs,
                                       std::shared_ptr<Value> rhs,
                                       const std::string& name,
                                       BinOpFlags flags) {
    auto inst = std::make_shared<Instruction>(Opcode::Sub, lhs->type(), name);
    inst->add_operand(lhs);
    inst->add_operand(rhs);
    inst->binop_flags() = flags;
    return inst;
}

std::shared_ptr<Instruction> make_mul(std::shared_ptr<Value> lhs,
                                       std::shared_ptr<Value> rhs,
                                       const std::string& name,
                                       BinOpFlags flags) {
    auto inst = std::make_shared<Instruction>(Opcode::Mul, lhs->type(), name);
    inst->add_operand(lhs);
    inst->add_operand(rhs);
    inst->binop_flags() = flags;
    return inst;
}

std::shared_ptr<Instruction> make_icmp(CmpPredicate pred,
                                        std::shared_ptr<Value> lhs,
                                        std::shared_ptr<Value> rhs,
                                        const std::string& name) {
    // icmp produces an i1 result
    static const auto i1_ty = std::make_shared<IntegerType>(1);
    auto inst = std::make_shared<Instruction>(Opcode::ICmp, i1_ty, name);
    inst->add_operand(lhs);
    inst->add_operand(rhs);
    inst->set_metadata("pred", std::to_string(static_cast<unsigned>(pred)));
    return inst;
}

std::shared_ptr<Instruction> make_alloca(std::shared_ptr<Type> ty,
                                          const std::string& name,
                                          unsigned align) {
    auto ptr_ty = std::make_shared<PointerType>(ty);
    auto inst = std::make_shared<Instruction>(Opcode::Alloca, ptr_ty, name);
    if (align != 0) inst->set_alignment(align);
    return inst;
}

std::shared_ptr<Instruction> make_load(std::shared_ptr<Type> loaded_type,
                                        std::shared_ptr<Value> ptr,
                                        const std::string& name,
                                        unsigned align) {
    // Use the explicitly provided loaded_type
    auto inst = std::make_shared<Instruction>(Opcode::Load, loaded_type, name);
    inst->add_operand(ptr);
    if (align != 0) inst->set_alignment(align);
    return inst;
}

std::shared_ptr<Instruction> make_load(std::shared_ptr<Value> ptr,
                                        const std::string& name,
                                        unsigned align) {
    // Infer loaded type from pointer's pointee type (with safety check)
    std::shared_ptr<Type> result_ty;
    if (ptr && ptr->type() && ptr->type()->is_pointer()) {
        auto& ptr_ty = static_cast<const PointerType&>(*ptr->type());
        result_ty = ptr_ty.pointee();
    }
    if (!result_ty) {
        // Fallback: use i32 as a guess and add a warning via metadata
        static const auto i32_fallback = std::make_shared<IntegerType>(32);
        result_ty = i32_fallback;
    }
    auto inst = std::make_shared<Instruction>(Opcode::Load, result_ty, name);
    inst->add_operand(ptr);
    if (align != 0) inst->set_alignment(align);
    return inst;
}

std::shared_ptr<Instruction> make_store(std::shared_ptr<Value> val,
                                         std::shared_ptr<Value> ptr,
                                         unsigned align) {
    static const auto void_ty = std::make_shared<VoidType>();
    auto inst = std::make_shared<Instruction>(Opcode::Store, void_ty);
    inst->add_operand(val);
    inst->add_operand(ptr);
    if (align != 0) inst->set_alignment(align);
    return inst;
}

std::shared_ptr<Instruction> make_phi(std::shared_ptr<Type> ty,
                                       const std::string& name) {
    auto inst = std::make_shared<Instruction>(Opcode::Phi, ty, name);
    // Incoming values and blocks are added later via add_operand()
    // and set_metadata("phi_blocks", "bb1,bb2,...").
    return inst;
}

std::shared_ptr<Instruction> make_call(std::shared_ptr<Type> ret_type,
                                        const std::string& callee,
                                        std::vector<std::shared_ptr<Value>> args,
                                        const std::string& name) {
    auto inst = std::make_shared<Instruction>(Opcode::Call, ret_type, name);
    inst->set_metadata("callee", callee);
    for (auto& a : args) {
        inst->add_operand(a);
    }
    return inst;
}

std::shared_ptr<Instruction> make_select(std::shared_ptr<Value> cond,
                                          std::shared_ptr<Value> true_val,
                                          std::shared_ptr<Value> false_val,
                                          const std::string& name) {
    auto inst = std::make_shared<Instruction>(Opcode::Select, true_val->type(), name);
    inst->add_operand(cond);      // operand 0
    inst->add_operand(true_val);  // operand 1
    inst->add_operand(false_val); // operand 2
    return inst;
}

std::shared_ptr<Instruction> make_gep(std::shared_ptr<Type> result_type,
                                       std::shared_ptr<Value> ptr,
                                       std::vector<std::shared_ptr<Value>> indices,
                                       const std::string& name) {
    auto inst = std::make_shared<Instruction>(Opcode::GetElementPtr, result_type, name);
    inst->add_operand(ptr); // operand 0
    for (auto& idx : indices) {
        inst->add_operand(idx);
    }
    return inst;
}

std::shared_ptr<Instruction> make_extractelement(std::shared_ptr<Value> vec,
                                                  std::shared_ptr<Value> index,
                                                  const std::string& name) {
    std::shared_ptr<Type> elem_ty;
    if (vec && vec->type() && vec->type()->is_vector()) {
        elem_ty = static_cast<const VectorType&>(*vec->type()).element_type();
    } else {
        static const auto i32_fallback = std::make_shared<IntegerType>(32);
        elem_ty = i32_fallback;
    }
    auto inst = std::make_shared<Instruction>(Opcode::ExtractElement, elem_ty, name);
    inst->add_operand(vec);
    inst->add_operand(index);
    return inst;
}

std::shared_ptr<Instruction> make_insertelement(std::shared_ptr<Value> vec,
                                                 std::shared_ptr<Value> elem,
                                                 std::shared_ptr<Value> index,
                                                 const std::string& name) {
    auto inst = std::make_shared<Instruction>(Opcode::InsertElement, vec->type(), name);
    inst->add_operand(vec);
    inst->add_operand(elem);
    inst->add_operand(index);
    return inst;
}

std::shared_ptr<Instruction> make_shufflevector(std::shared_ptr<Value> lhs,
                                                 std::shared_ptr<Value> rhs,
                                                 std::shared_ptr<Value> mask,
                                                 const std::string& name) {
    // Result: vector with the source element type and the mask's lane count.
    std::shared_ptr<Type> result_ty = lhs ? lhs->type() : nullptr;
    if (lhs && lhs->type() && lhs->type()->is_vector() &&
        mask && mask->type() && mask->type()->is_vector()) {
        auto elem = static_cast<const VectorType&>(*lhs->type()).element_type();
        auto lanes = static_cast<const VectorType&>(*mask->type()).count();
        result_ty = std::make_shared<VectorType>(lanes, elem);
    }
    auto inst = std::make_shared<Instruction>(Opcode::ShuffleVector, result_ty, name);
    inst->add_operand(lhs);
    inst->add_operand(rhs);
    inst->add_operand(mask);
    return inst;
}

std::shared_ptr<Instruction> make_binop(Opcode op,
                                         std::shared_ptr<Value> lhs,
                                         std::shared_ptr<Value> rhs,
                                         const std::string& name,
                                         BinOpFlags flags) {
    auto inst = std::make_shared<Instruction>(op, lhs->type(), name);
    inst->add_operand(lhs);
    inst->add_operand(rhs);
    inst->binop_flags() = flags;
    return inst;
}

} // namespace inst
} // namespace clunk::ir
