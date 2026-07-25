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
 * Clunk GPU PTXEmitter — ir::Function → PTX text lowering.
 */
#include "clunk/GPU/PTXEmitter.h"

#include <algorithm>
#include <sstream>
#include <unordered_map>

namespace clunk::gpu {

// ── Constructor ─────────────────────────────────────────────────────────────

PTXEmitter::PTXEmitter(GpuArch arch) : arch_(arch) {}

// ── Type classification ─────────────────────────────────────────────────────

PTXEmitter::RegClass PTXEmitter::classify(const ir::Type* ty) const {
    if (!ty) return {"u32", "%r"};
    if (ty->is_integer()) {
        unsigned bits = static_cast<const ir::IntegerType*>(ty)->bits();
        if (bits == 1) return {"pred", "%p"};
        if (bits <= 32) return {"u32", "%r"};
        return {"u64", "%rd"};
    }
    if (ty->is_float())  return {"f32", "%f"};
    if (ty->is_double()) return {"f64", "%fd"};
    if (ty->is_pointer()) return {"u64", "%rd"};
    return {"u32", "%r"};  // fallback
}

unsigned PTXEmitter::type_bytes(const ir::Type* ty) const {
    if (!ty) return 4;
    if (ty->is_integer()) {
        unsigned bits = static_cast<const ir::IntegerType*>(ty)->bits();
        if (bits == 1) return 1;
        return (bits + 7) / 8;
    }
    if (ty->is_float())  return 4;
    if (ty->is_double()) return 8;
    if (ty->is_pointer()) return 8;
    return 4;
}

const char* PTXEmitter::ptx_address_space(unsigned addrspace) {
    switch (addrspace) {
        case 0: return "global";
        case 1: return "global";   // LLR addrspace(1) ≈ global on CUDA
        case 2: return "shared";
        case 3: return "local";
        case 4: return "const";
        default: return "global";
    }
}

// ── Register allocation ─────────────────────────────────────────────────────

std::string PTXEmitter::alloc(RegClass cls) {
    if (cls.decl == "u32")       return "%r"   + std::to_string(u32_count_++);
    if (cls.decl == "u64")       return "%rd"  + std::to_string(u64_count_++);
    if (cls.decl == "f32")       return "%f"   + std::to_string(f32_count_++);
    if (cls.decl == "f64")       return "%fd"  + std::to_string(f64_count_++);
    if (cls.decl == "pred")      return "%p"   + std::to_string(pred_count_++);
    return "%r0";
}

std::string PTXEmitter::alloc_for(const std::string& ssa_name,
                                    const ir::Type* ty) {
    auto it = reg_by_name_.find(ssa_name);
    if (it != reg_by_name_.end()) return it->second;
    auto r = alloc(classify(ty));
    reg_by_name_[ssa_name] = r;
    return r;
}

std::string PTXEmitter::lookup(const std::string& ssa_name) const {
    auto it = reg_by_name_.find(ssa_name);
    return it != reg_by_name_.end() ? it->second : std::string("%r0");
}

// ── Operand rendering ───────────────────────────────────────────────────────

std::string PTXEmitter::render_operand(const ir::Value& v) const {
    // ConstantInt → render its integer value as a decimal literal
    if (auto* ci = dynamic_cast<const ir::ConstantInt*>(&v)) {
        return std::to_string(ci->value());
    }
    // ConstantFP → render as a decimal literal (PTX accepts float immediates)
    if (auto* cf = dynamic_cast<const ir::ConstantFP*>(&v)) {
        std::ostringstream oss;
        oss.precision(9);
        oss << cf->value();
        return oss.str();
    }
    // ConstantPointerNull → 0
    if (dynamic_cast<const ir::ConstantPointerNull*>(&v)) {
        return "0";
    }
    // UndefValue / PoisonValue → 0 (treat as zero)
    if (dynamic_cast<const ir::UndefValue*>(&v) ||
        dynamic_cast<const ir::PoisonValue*>(&v)) {
        return "0";
    }
    // Named value (argument or instruction result) → its register
    if (v.has_name()) {
        auto it = reg_by_name_.find(v.name());
        if (it != reg_by_name_.end()) return it->second;
        // Unknown name — emit 0 and let PTX assembler catch it (we should
        // never get here if alloc_for was called for every defined value).
        return std::string("0 /*unknown:") + v.name() + "*/";
    }
    return "0";
}

// ── emit (top-level) ────────────────────────────────────────────────────────

std::string PTXEmitter::emit(const ir::Function& fn) {
    // Reset state for this emission
    total_regs_ = 0;
    u32_count_ = u64_count_ = f32_count_ = f64_count_ = pred_count_ = 0;
    reg_by_name_.clear();

    ArchLimits limits = get_arch_limits(arch_);

    std::string out;

    auto append = [&out](const std::string& s) { out += s; };
    auto append_line = [&out](const std::string& s) { out += s; out += "\n"; };

    // ── PTX header ──────────────────────────────────────────────────────────
    append_line(".version " + std::to_string(limits.ptx_isa_version_major) +
                "." + std::to_string(limits.ptx_isa_version_minor));
    append_line(".target " + std::string(limits.name));
    append_line(".address_size 64");
    append("");

    // ── Kernel signature ────────────────────────────────────────────────────
    append(".visible .entry " + fn.name() + "(");
    bool first = true;
    for (const auto& arg : fn.arguments()) {
        append(first ? "\n    " : ",\n    ");
        first = false;

        RegClass cls = classify(arg.type.get());
        append(".param ." + cls.decl + " arg_" + arg.name);
    }
    append_line("\n) {");

    // ── Allocate registers for arguments (load .param into .reg) ────────────
    //
    // Each argument is copied from its .param slot into a virtual register
    // at function entry, so the rest of the body can refer to it by name.
    std::string arg_loads;
    for (const auto& arg : fn.arguments()) {
        RegClass cls = classify(arg.type.get());
        std::string reg = alloc_for(arg.name, arg.type.get());
        if (cls.decl == "pred") {
            arg_loads += "    ld.param.u8 " + reg + ", [arg_" +
                         arg.name + "];\n";
        } else if (cls.decl == "u32" || cls.decl == "u64" ||
                   cls.decl == "f32" || cls.decl == "f64") {
            arg_loads += "    ld.param." + cls.decl + " " + reg +
                         ", [arg_" + arg.name + "];\n";
        }
    }

    // ── Function body ───────────────────────────────────────────────────────
    std::string body;
    for (const auto& bb : fn.blocks()) {
        if (!bb) continue;

        // Block label. PTX labels are bare identifiers followed by ':'.
        // Prefix with $L_ to avoid collisions with registers (%name) and
        // PTX keywords.
        body += "$L_" + bb->name() + ":\n";

        for (const auto& inst : bb->instructions()) {
            // Pre-allocate a register for the instruction's result so that
            // render_operand() can find it if the instruction is later used
            // (in SSA this is fine — the result name is unique).
            if (inst->has_name() && !inst->is_terminator() &&
                inst->opcode() != ir::Opcode::Store &&
                inst->opcode() != ir::Opcode::Fence) {
                alloc_for(inst->name(), inst->type().get());
            }
            emit_instruction(body, *inst);
        }
    }

    // ── Emit register declarations and body ─────────────────────────────────
    emit_reg_decls(out);
    out += arg_loads;
    out += body;

    // Ensure every kernel ends with a ret (PTX requires it).
    append_line("    ret;");
    append_line("}");

    total_regs_ = u32_count_ + u64_count_ + f32_count_ +
                  f64_count_ + pred_count_;
    return out;
}

// ── emit_reg_decls ──────────────────────────────────────────────────────────

void PTXEmitter::emit_reg_decls(std::string& out) const {
    if (u32_count_ > 0) {
        out += "    .reg .u32 %r<" + std::to_string(u32_count_) + ">;\n";
    }
    if (u64_count_ > 0) {
        out += "    .reg .u64 %rd<" + std::to_string(u64_count_) + ">;\n";
    }
    if (f32_count_ > 0) {
        out += "    .reg .f32 %f<" + std::to_string(f32_count_) + ">;\n";
    }
    if (f64_count_ > 0) {
        out += "    .reg .f64 %fd<" + std::to_string(f64_count_) + ">;\n";
    }
    if (pred_count_ > 0) {
        out += "    .reg .pred %p<" + std::to_string(pred_count_) + ">;\n";
    }
    out += "\n";
}

// ── emit_instruction ────────────────────────────────────────────────────────

void PTXEmitter::emit_instruction(std::string& out,
                                    const ir::Instruction& inst)
{
    using Op = ir::Opcode;
    switch (inst.opcode()) {
        // Integer binary
        case Op::Add:  emit_binop(out, inst, "add.u32"); return;
        case Op::Sub:  emit_binop(out, inst, "sub.u32"); return;
        case Op::Mul:  emit_binop(out, inst, "mul.lo.u32"); return;
        case Op::UDiv: emit_binop(out, inst, "div.u32"); return;
        case Op::SDiv: emit_binop(out, inst, "div.s32"); return;
        case Op::URem: emit_binop(out, inst, "rem.u32"); return;
        case Op::SRem: emit_binop(out, inst, "rem.s32"); return;
        // Bitwise
        case Op::And:  emit_binop(out, inst, "and.b32"); return;
        case Op::Or:   emit_binop(out, inst, "or.b32"); return;
        case Op::Xor:  emit_binop(out, inst, "xor.b32"); return;
        case Op::Shl:  emit_binop(out, inst, "shl.b32"); return;
        case Op::LShr: emit_binop(out, inst, "shr.b32"); return;
        case Op::AShr: emit_binop(out, inst, "shr.s32"); return;
        // Floating-point binary
        case Op::FAdd: emit_fp_binop(out, inst, "add.f32"); return;
        case Op::FSub: emit_fp_binop(out, inst, "sub.f32"); return;
        case Op::FMul: emit_fp_binop(out, inst, "mul.f32"); return;
        case Op::FDiv: emit_fp_binop(out, inst, "div.rn.f32"); return;
        case Op::FRem: {
            // PTX has no frem instruction — would need a libcall.
            // Emit a comment and a nop so the kernel still assembles.
            out += "    // frem not directly supported; libcall required\n";
            out += "    nop;\n";
            return;
        }
        // Comparisons
        case Op::ICmp: emit_icmp(out, inst); return;
        case Op::FCmp: {
            // Simplified: treat all fcmp as setp with the same predicate
            // (real implementation would decode the FP predicate).
            out += "    // fcmp lowered approximately\n";
            emit_icmp(out, inst);  // reuse integer setp lowering
            return;
        }
        // Memory
        case Op::Load:  emit_load(out, inst); return;
        case Op::Store: emit_store(out, inst); return;
        case Op::GetElementPtr: emit_gep(out, inst); return;
        case Op::Alloca: emit_alloca(out, inst); return;
        // Control flow
        case Op::Br:  emit_branch(out, inst); return;
        case Op::Ret: emit_return(out, inst); return;
        // Conversions
        case Op::Trunc: case Op::ZExt: case Op::SExt:
        case Op::FPTrunc: case Op::FPExt:
        case Op::FPToUI: case Op::FPToSI:
        case Op::UIToFP: case Op::SIToFP:
        case Op::PtrToInt: case Op::IntToPtr:
        case Op::BitCast: case Op::AddrSpaceCast: {
            // Lower as a mov (with cvt for typed conversions — simplified).
            std::string dst = lookup(inst.name());
            if (inst.num_operands() >= 1) {
                std::string src = render_operand(*inst.operand(0));
                out += "    mov.u32 " + dst + ", " + src + ";\n";
            }
            return;
        }
        // Select → selp
        case Op::Select: {
            std::string dst = lookup(inst.name());
            if (inst.num_operands() >= 3) {
                std::string cond = render_operand(*inst.operand(0));
                std::string tv   = render_operand(*inst.operand(1));
                std::string fv   = render_operand(*inst.operand(2));
                // selp: dst = (pred) ? tv : fv
                out += "    selp.u32 " + dst + ", " + tv + ", " + fv +
                       ", " + cond + ";\n";
            }
            return;
        }
        // Phi → mov (placeholder; real lowering requires insertion of
        // moves at predecessor exits).
        case Op::Phi: {
            std::string dst = lookup(inst.name());
            if (inst.num_operands() >= 1) {
                std::string src = render_operand(*inst.operand(0));
                out += "    mov.u32 " + dst + ", " + src + ";\n";
            }
            return;
        }
        // Call → comment (real implementation would emit prototype + call)
        case Op::Call: {
            out += "    // call not yet supported by PTXEmitter\n";
            out += "    nop;\n";
            return;
        }
        // Fence → membar
        case Op::Fence: {
            out += "    membar.gl;\n";
            return;
        }
        case Op::Unreachable: {
            out += "    trap;\n";
            return;
        }
        // Switch / Invoke / Resume — not yet supported
        case Op::Switch: case Op::SwitchInst:
        case Op::Invoke: case Op::Resume:
        case Op::VAArg:
        case Op::ExtractValue: case Op::InsertValue:
        case Op::ExtractElement: case Op::InsertElement:
        case Op::ShuffleVector:
        case Op::LandingPad: {
            out += "    // unsupported opcode: " +
                   std::string(ir::Instruction::opcode_name(inst.opcode())) +
                   "\n";
            out += "    nop;\n";
            return;
        }
    }
    out += "    // unknown opcode\n    nop;\n";
}

// ── emit_binop ──────────────────────────────────────────────────────────────

void PTXEmitter::emit_binop(std::string& out, const ir::Instruction& inst,
                              const std::string& ptx_op)
{
    if (inst.num_operands() < 2) return;
    std::string dst = lookup(inst.name());
    std::string lhs = render_operand(*inst.operand(0));
    std::string rhs = render_operand(*inst.operand(1));
    out += "    " + ptx_op + " " + dst + ", " + lhs + ", " + rhs + ";\n";
}

// ── emit_fp_binop ───────────────────────────────────────────────────────────

void PTXEmitter::emit_fp_binop(std::string& out, const ir::Instruction& inst,
                                 const std::string& ptx_op)
{
    if (inst.num_operands() < 2) return;
    std::string dst = lookup(inst.name());
    std::string lhs = render_operand(*inst.operand(0));
    std::string rhs = render_operand(*inst.operand(1));
    out += "    " + ptx_op + " " + dst + ", " + lhs + ", " + rhs + ";\n";
}

// ── emit_icmp ───────────────────────────────────────────────────────────────

void PTXEmitter::emit_icmp(std::string& out, const ir::Instruction& inst) {
    if (inst.num_operands() < 2) return;
    std::string dst = lookup(inst.name());

    // Decode the comparison predicate from metadata "pred"
    unsigned pred_val = 32;  // default EQ
    auto it = inst.metadata().find("pred");
    if (it != inst.metadata().end()) {
        try { pred_val = static_cast<unsigned>(std::stoul(it->second)); }
        catch (...) { pred_val = 32; }
    }
    auto pred = static_cast<ir::CmpPredicate>(pred_val);

    // Map CmpPredicate → PTX setp suffix
    const char* setp_suffix = "eq.u32";
    bool is_signed = false;
    switch (pred) {
        case ir::CmpPredicate::EQ:  setp_suffix = "eq.u32"; break;
        case ir::CmpPredicate::NE:  setp_suffix = "ne.u32"; break;
        case ir::CmpPredicate::UGT: setp_suffix = "hi.u32"; break;
        case ir::CmpPredicate::UGE: setp_suffix = "hs.u32"; break;
        case ir::CmpPredicate::ULT: setp_suffix = "lo.u32"; break;
        case ir::CmpPredicate::ULE: setp_suffix = "ls.u32"; break;
        case ir::CmpPredicate::SGT: setp_suffix = "gt.s32"; is_signed = true; break;
        case ir::CmpPredicate::SGE: setp_suffix = "ge.s32"; is_signed = true; break;
        case ir::CmpPredicate::SLT: setp_suffix = "lt.s32"; is_signed = true; break;
        case ir::CmpPredicate::SLE: setp_suffix = "le.s32"; is_signed = true; break;
        default: setp_suffix = "eq.u32"; break;
    }
    (void)is_signed;

    std::string lhs = render_operand(*inst.operand(0));
    std::string rhs = render_operand(*inst.operand(1));
    out += "    setp." + std::string(setp_suffix) + " " + dst +
           ", " + lhs + ", " + rhs + ";\n";
}

// ── emit_load ───────────────────────────────────────────────────────────────

void PTXEmitter::emit_load(std::string& out, const ir::Instruction& inst) {
    if (inst.num_operands() < 1) return;
    std::string dst = lookup(inst.name());
    std::string ptr = render_operand(*inst.operand(0));

    // Determine address space from the pointer operand's type
    const char* space = "global";
    if (auto* pty = dynamic_cast<const ir::PointerType*>(inst.operand(0)->type().get())) {
        space = ptx_address_space(pty->address_space());
    }
    unsigned bytes = type_bytes(inst.type().get());
    if (bytes == 0) bytes = 4;

    out += "    ld." + std::string(space) + ".b" + std::to_string(bytes) +
           " " + dst + ", [" + ptr + "];\n";
}

// ── emit_store ──────────────────────────────────────────────────────────────

void PTXEmitter::emit_store(std::string& out, const ir::Instruction& inst) {
    if (inst.num_operands() < 2) return;
    std::string val = render_operand(*inst.operand(0));
    std::string ptr = render_operand(*inst.operand(1));

    const char* space = "global";
    if (auto* pty = dynamic_cast<const ir::PointerType*>(inst.operand(1)->type().get())) {
        space = ptx_address_space(pty->address_space());
    }
    unsigned bytes = type_bytes(inst.operand(0)->type().get());
    if (bytes == 0) bytes = 4;

    out += "    st." + std::string(space) + ".b" + std::to_string(bytes) +
           " [" + ptr + "], " + val + ";\n";
}

// ── emit_gep ────────────────────────────────────────────────────────────────
//
// Lower a GEP to a sequence of mul + add. We model the simple case
// `getelementptr T, T* %ptr, i32 %idx` as `%ptr + %idx * sizeof(T)`.
// Multi-index GEPs are decomposed into a sequence of strides, but the
// stride is only correct when the index advances by exactly one element
// size at each step (true for flat arrays; incorrect for nested arrays
// and structs — a known limitation).
//
void PTXEmitter::emit_gep(std::string& out, const ir::Instruction& inst) {
    if (inst.num_operands() < 2) {
        // Just a pointer copy
        if (inst.has_name()) {
            std::string dst = lookup(inst.name());
            if (inst.num_operands() == 1) {
                out += "    mov.u64 " + dst + ", " +
                       render_operand(*inst.operand(0)) + ";\n";
            }
        }
        return;
    }

    std::string dst = lookup(inst.name());
    std::string ptr = render_operand(*inst.operand(0));

    // Determine element size from the pointer's pointee type.
    unsigned elem_size = 4;  // default i32
    if (auto* pty = dynamic_cast<const ir::PointerType*>(inst.operand(0)->type().get())) {
        if (pty->pointee()) {
            unsigned b = type_bytes(pty->pointee().get());
            if (b > 0) elem_size = b;
        }
    }

    // Walk indices: for each index, scale by element size and add to a
    // running pointer. For simplicity, we treat every index as a flat
    // byte-offset multiplier (i.e. assume nested GEPs reduce to a
    // single stride — true for the common 1-D array pattern in our
    // examples, false in general).
    std::string accum = ptr;
    for (size_t i = 1; i < inst.num_operands(); ++i) {
        std::string idx = render_operand(*inst.operand(i));
        std::string scaled = alloc({"u32", "%r"});
        std::string scaled_ext = alloc({"u64", "%rd"});

        // scaled = idx * elem_size
        out += "    mul.lo.u32 " + scaled + ", " + idx + ", " +
               std::to_string(elem_size) + ";\n";
        // scaled_ext = (u64)scaled
        out += "    cvt.u64.u32 " + scaled_ext + ", " + scaled + ";\n";
        // accum = accum + scaled_ext
        std::string next = alloc({"u64", "%rd"});
        out += "    add.u64 " + next + ", " + accum + ", " + scaled_ext + ";\n";
        accum = next;
    }
    out += "    mov.u64 " + dst + ", " + accum + ";\n";
}

// ── emit_branch ─────────────────────────────────────────────────────────────

void PTXEmitter::emit_branch(std::string& out, const ir::Instruction& inst) {
    const auto& md = inst.metadata();
    if (inst.num_operands() == 0) {
        // Unconditional
        auto it = md.find("dest_bb");
        if (it != md.end()) {
            out += "    bra $L_" + it->second + ";\n";
        }
        return;
    }
    // Conditional: @pred bra true; bra false
    std::string cond = render_operand(*inst.operand(0));
    auto t_it = md.find("true_bb");
    auto f_it = md.find("false_bb");
    if (t_it != md.end()) {
        out += "    @" + cond + " bra $L_" + t_it->second + ";\n";
    }
    if (f_it != md.end()) {
        out += "    bra $L_" + f_it->second + ";\n";
    } else if (t_it != md.end()) {
        // No false branch — fall through to next block (emit a nop placeholder).
        out += "    nop;\n";
    }
}

// ── emit_return ─────────────────────────────────────────────────────────────

void PTXEmitter::emit_return(std::string& out, const ir::Instruction& inst) {
    // The kernel-level `ret;` is emitted by emit() at function end, so
    // any intermediate ret instructions just become unconditional
    // branches to the function exit (we approximate this by emitting
    // ret — PTX allows multiple ret; though it's unusual).
    (void)inst;
    out += "    ret;\n";
}

// ── emit_alloca ─────────────────────────────────────────────────────────────

void PTXEmitter::emit_alloca(std::string& out, const ir::Instruction& inst) {
    if (!inst.has_name()) return;
    std::string dst = lookup(inst.name());

    // Alloca lowers to a .local variable declaration. Since PTX requires
    // declarations at function scope (not inside basic blocks), we emit a
    // runtime "address-of" pattern: get the address of a pre-declared
    // .local array.
    //
    // For simplicity here we emit a comment and a mov from a placeholder
    // local symbol; a full implementation would collect all allocas and
    // hoist their declarations to the top of the function.
    unsigned bytes = type_bytes(inst.type().get());
    out += "    // alloca lowered to .local .align " +
           std::to_string(bytes) + " .b8 " + dst + "_slot[0];\n";
    out += "    mov.u64 " + dst + ", 0; // address-of " + dst + "_slot\n";
}

} // namespace clunk::gpu
