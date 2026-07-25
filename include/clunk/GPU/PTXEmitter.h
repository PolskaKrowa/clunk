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
 * Clunk GPU PTXEmitter — lowers an ir::Function to NVIDIA PTX text.
 *
 * This is the first real PTX emitter in clunk: prior to this the "GPU
 * optimiser" only annotated the IR with metadata strings and never
 * produced any PTX. The emitter covers the integer arithmetic, control
 * flow, memory, and comparison opcodes most kernels need:
 *
 *   add / sub / mul / udiv / sdiv / urem / srem
 *   and / or  / xor / shl  / lshr / ashr
 *   fadd / fsub / fmul / fdiv
 *   icmp → setp.{eq,ne,lt,le,gt,ge}.{u32,s32}
 *   br (cond) → @P bra / bra
 *   br (uncond) → bra
 *   ret → ret
 *   load / store → ld.<space>.b<width> / st.<space>.b<width>
 *   getelementptr → mad / add (pointer arithmetic)
 *   alloca → .local declaration
 *
 * Register allocation is a naive sequential scheme: every named SSA
 * value receives its own virtual register of the appropriate PTX class
 * (.u32, .u64, .f32, .f64, .pred). This is correct but high-pressure;
 * a linear-scan allocator is a follow-up. Spilling to local memory is
 * not yet performed.
 *
 * The emitter is arch-aware: the `.target` directive is set from the
 * GpuArch passed to the constructor (default SM_80).
 */
#include "clunk/GPU/ArchLimits.h"
#include "clunk/IR/Function.h"
#include "clunk/IR/Instruction.h"
#include "clunk/IR/Type.h"
#include "clunk/IR/Value.h"

#include <string>
#include <unordered_map>
#include <vector>

namespace clunk::gpu {

class PTXEmitter final {
public:
    explicit PTXEmitter(GpuArch arch = GpuArch::SM_80);

    // Emit PTX text for the function. Returns the full kernel source
    // (header + signature + body). If the function cannot be lowered
    // (e.g. unsupported opcode), the offending instruction is emitted
    // as a `// unsupported: <opcode>` comment and lowered to a `nop;`.
    std::string emit(const ir::Function& fn);

    void set_arch(GpuArch arch) { arch_ = arch; }
    GpuArch arch() const { return arch_; }

    // Total number of virtual registers allocated during the last emit.
    size_t num_registers() const { return total_regs_; }

private:
    GpuArch arch_;
    size_t total_regs_ = 0;

    // Per-class counters for sequential allocation
    unsigned u32_count_ = 0;  // %r0..%rN
    unsigned u64_count_ = 0;  // %rd0..%rdN
    unsigned f32_count_ = 0;  // %f0..%fN
    unsigned f64_count_ = 0;  // %fd0..%fdN
    unsigned pred_count_ = 0; // %p0..%pN

    // SSA value name → PTX register name (e.g. "sum" → "%r0")
    std::unordered_map<std::string, std::string> reg_by_name_;

    // PTX register class info for an IR type
    struct RegClass {
        std::string decl;    // "u32", "u64", "f32", "f64", "pred"
        std::string prefix;  // "%r", "%rd", "%f", "%fd", "%p"
    };

    RegClass classify(const ir::Type* ty) const;

    // Allocate a fresh register of the given class
    std::string alloc(RegClass cls);

    // Allocate (or look up existing) register for a named SSA value
    std::string alloc_for(const std::string& ssa_name, const ir::Type* ty);
    std::string lookup(const std::string& ssa_name) const;

    // Render a Value as an operand string. Constants become literals,
    // named values become their PTX register name, undef/null become "0".
    std::string render_operand(const ir::Value& v) const;

    // PTX load/store width suffix for a type (8, 16, 32, 64)
    unsigned type_bytes(const ir::Type* ty) const;

    // PTX address space name for a pointer-type address-space number
    static const char* ptx_address_space(unsigned addrspace);

    // Emit a single instruction; appends to `out`.
    void emit_instruction(std::string& out, const ir::Instruction& inst);

    // Helpers for specific opcode families
    void emit_binop(std::string& out, const ir::Instruction& inst,
                    const std::string& ptx_op);
    void emit_fp_binop(std::string& out, const ir::Instruction& inst,
                       const std::string& ptx_op);
    void emit_icmp(std::string& out, const ir::Instruction& inst);
    void emit_load(std::string& out, const ir::Instruction& inst);
    void emit_store(std::string& out, const ir::Instruction& inst);
    void emit_gep(std::string& out, const ir::Instruction& inst);
    void emit_branch(std::string& out, const ir::Instruction& inst);
    void emit_return(std::string& out, const ir::Instruction& inst);
    void emit_alloca(std::string& out, const ir::Instruction& inst);

    // Emit `.reg` declarations for all allocated registers of a class
    void emit_reg_decls(std::string& out) const;
};

} // namespace clunk::gpu
