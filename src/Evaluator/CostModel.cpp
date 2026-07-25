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
 * Clunk CostModel — implementation of the per-architecture cost tables.
 *
 * The tables below are coarse approximations based on public
 * architecture manuals (Intel SDM, ARM Cortex-A78 TRM, NVIDIA PTX
 * ISA). They are intentionally simple: the goal is to capture the
 * *relative* cost differences between opcodes and between
 * architectures, not to predict absolute cycle counts.
 */
#include "clunk/Evaluator/CostModel.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <string>

namespace clunk::evaluator {

namespace {

// Case-insensitive "starts with" helper used by the arch-name factory.
bool iequals_prefix(const std::string& s, const std::string& prefix) {
    if (s.size() < prefix.size()) return false;
    for (size_t i = 0; i < prefix.size(); ++i) {
        if (std::tolower(static_cast<unsigned char>(s[i])) !=
            std::tolower(static_cast<unsigned char>(prefix[i]))) {
            return false;
        }
    }
    return true;
}

// True iff `v` is a strictly positive power of two (1, 2, 4, 8, ...).
// Treats `v` as unsigned (so 1<<63 is still a power of two).
bool is_pow2_u64(uint64_t v) {
    return v != 0 && (v & (v - 1)) == 0;
}

// Helper: does `operands` contain a constant operand satisfying `pred`?
// Scans all entries; returns true on the first match.
bool any_operand_is(const std::vector<OperandValueInfo>& operands,
                     bool (*pred)(const OperandValueInfo&)) {
    for (const auto& o : operands) {
        if (pred(o)) return true;
    }
    return false;
}

bool operand_is_zero(const OperandValueInfo& o) {
    return o.is_constant && o.is_zero;
}
bool operand_is_one(const OperandValueInfo& o) {
    return o.is_constant && o.is_one;
}
bool operand_is_pow2(const OperandValueInfo& o) {
    return o.is_constant && o.is_power_of_two && !o.is_zero;
}

} // namespace

// ── Generic model ─────────────────────────────────────────────────────────

OpCost GenericCostModel::cost(ir::Opcode op) const {
    OpCost c{1.0, 1.0, 0, 1};
    switch (op) {
        // Integer arithmetic
        case ir::Opcode::Add:
        case ir::Opcode::Sub:
        case ir::Opcode::And:
        case ir::Opcode::Or:
        case ir::Opcode::Xor:
            c.latency_cycles = 1.0; c.throughput_cycles = 0.5; c.uops = 1;
            break;
        case ir::Opcode::Mul:
            c.latency_cycles = 3.0; c.throughput_cycles = 1.0; c.uops = 1;
            break;
        case ir::Opcode::UDiv:
        case ir::Opcode::SDiv:
        case ir::Opcode::URem:
        case ir::Opcode::SRem:
            c.latency_cycles = 20.0; c.throughput_cycles = 20.0; c.uops = 4;
            break;
        case ir::Opcode::Shl:
        case ir::Opcode::LShr:
        case ir::Opcode::AShr:
            c.latency_cycles = 1.0; c.throughput_cycles = 1.0; c.uops = 1;
            break;
        // Floating point
        case ir::Opcode::FAdd:
        case ir::Opcode::FSub:
            c.latency_cycles = 4.0; c.throughput_cycles = 0.5; c.uops = 1;
            break;
        case ir::Opcode::FMul:
            c.latency_cycles = 5.0; c.throughput_cycles = 0.5; c.uops = 1;
            break;
        case ir::Opcode::FDiv:
        case ir::Opcode::FRem:
            c.latency_cycles = 18.0; c.throughput_cycles = 18.0; c.uops = 4;
            break;
        // Memory
        case ir::Opcode::Alloca:
            c.latency_cycles = 0.0; c.throughput_cycles = 1.0; c.uops = 0;
            break;
        case ir::Opcode::Load:
            c.latency_cycles = 4.0; c.throughput_cycles = 0.5; c.uops = 2;
            break;
        case ir::Opcode::Store:
            c.latency_cycles = 1.0; c.throughput_cycles = 1.0; c.uops = 1;
            break;
        case ir::Opcode::GetElementPtr:
        case ir::Opcode::BitCast:
        case ir::Opcode::PtrToInt:
        case ir::Opcode::IntToPtr:
            c.latency_cycles = 1.0; c.throughput_cycles = 1.0; c.uops = 1;
            break;
        // Compare/select
        case ir::Opcode::ICmp:
        case ir::Opcode::FCmp:
            c.latency_cycles = 1.0; c.throughput_cycles = 1.0; c.uops = 1;
            break;
        case ir::Opcode::Select:
            c.latency_cycles = 2.0; c.throughput_cycles = 1.0; c.uops = 1;
            break;
        // Casts
        case ir::Opcode::Trunc:
        case ir::Opcode::ZExt:
        case ir::Opcode::SExt:
            c.latency_cycles = 1.0; c.throughput_cycles = 1.0; c.uops = 1;
            break;
        // Control flow
        case ir::Opcode::Br:
        case ir::Opcode::Switch:
        case ir::Opcode::SwitchInst:
            c.latency_cycles = 1.0; c.throughput_cycles = 1.0; c.uops = 1;
            break;
        case ir::Opcode::Ret:
            c.latency_cycles = 1.0; c.throughput_cycles = 1.0; c.uops = 1;
            break;
        case ir::Opcode::Call:
            c.latency_cycles = 20.0; c.throughput_cycles = 20.0; c.uops = 4;
            break;
        // Vector element ops (GPR<->SIMD domain crossing dominates).
        case ir::Opcode::ExtractElement:
        case ir::Opcode::InsertElement:
            c.latency_cycles = 3.0; c.throughput_cycles = 1.0; c.uops = 1;
            break;
        case ir::Opcode::ShuffleVector:
            c.latency_cycles = 1.0; c.throughput_cycles = 1.0; c.uops = 1;
            break;
        // Phi is zero-cost in hardware (renamed at register-alloc).
        case ir::Opcode::Phi:
            c.latency_cycles = 0.0; c.throughput_cycles = 0.0; c.uops = 0;
            break;
        default:
            c.latency_cycles = 1.0; c.throughput_cycles = 1.0; c.uops = 1;
            break;
    }
    return c;
}

MemoryLatencies GenericCostModel::memory() const { return MemoryLatencies{}; }

// ── x86_64 (Skylake-class) ────────────────────────────────────────────────
//
// Port mask conventions (Intel):
//   bit 0 → port 0 (ALU, mul, vector int)
//   bit 1 → port 1 (ALU, vector int)
//   bit 2 → port 5 (ALU, vector int, branches)
//   bit 3 → port 6 (ALU, branches)
//   bit 4 → port 2 (load)
//   bit 5 → port 3 (load)
//   bit 6 → port 4 (store-data)
//   bit 7 → port 7 (store-address)

OpCost X86_64CostModel::cost(ir::Opcode op) const {
    // Defaults derived from Generic; we override latency/ports/uops.
    OpCost c = GenericCostModel{}.cost(op);
    switch (op) {
        case ir::Opcode::Add:
        case ir::Opcode::Sub:
        case ir::Opcode::And:
        case ir::Opcode::Or:
        case ir::Opcode::Xor:
            c.latency_cycles = 1.0; c.throughput_cycles = 0.25;
            c.port_mask = 0b00010111; c.uops = 1; // ports 0,1,5,6
            break;
        case ir::Opcode::Mul:
            c.latency_cycles = 3.0; c.throughput_cycles = 1.0;
            c.port_mask = 0b00000011; c.uops = 3; // ports 0,1
            break;
        case ir::Opcode::UDiv:
        case ir::Opcode::SDiv:
        case ir::Opcode::URem:
        case ir::Opcode::SRem:
            c.latency_cycles = 25.0; c.throughput_cycles = 25.0;
            c.port_mask = 0b00000001; c.uops = 10; // port 0 only
            break;
        case ir::Opcode::Shl:
        case ir::Opcode::LShr:
        case ir::Opcode::AShr:
            c.latency_cycles = 1.0; c.throughput_cycles = 0.5;
            c.port_mask = 0b00010111; c.uops = 1; // ports 0,1,5,6
            break;
        case ir::Opcode::Load:
            c.latency_cycles = 5.0; c.throughput_cycles = 0.5;
            c.port_mask = 0b00110000; c.uops = 2; // ports 2,3
            break;
        case ir::Opcode::Store:
            c.latency_cycles = 1.0; c.throughput_cycles = 1.0;
            c.port_mask = 0b11000000; c.uops = 2; // ports 4,7
            break;
        case ir::Opcode::Br:
        case ir::Opcode::Switch:
        case ir::Opcode::SwitchInst:
            c.latency_cycles = 1.0; c.throughput_cycles = 1.0;
            c.port_mask = 0b01000100; c.uops = 2; // ports 0,6 + branch unit
            break;
        case ir::Opcode::ICmp:
        case ir::Opcode::FCmp:
            c.latency_cycles = 1.0; c.throughput_cycles = 0.5;
            c.port_mask = 0b00010111; c.uops = 1;
            break;
        case ir::Opcode::Call:
            c.latency_cycles = 30.0; c.throughput_cycles = 30.0;
            c.port_mask = 0b11111111; c.uops = 8;
            break;
        case ir::Opcode::ExtractElement:
        case ir::Opcode::InsertElement:
            // pextrd/pinsrd: cross the SIMD/GPR domain (port 0 + port 5).
            c.latency_cycles = 3.0; c.throughput_cycles = 1.0;
            c.port_mask = 0b00000101; c.uops = 2;
            break;
        case ir::Opcode::ShuffleVector:
            // pshufd/shufps-class: single uop on port 5.
            c.latency_cycles = 1.0; c.throughput_cycles = 1.0;
            c.port_mask = 0b00000100; c.uops = 1;
            break;
        default:
            break;
    }
    return c;
}

MemoryLatencies X86_64CostModel::memory() const {
    MemoryLatencies m;
    m.l1_load    = 4.0;
    m.l2_load    = 12.0;
    m.l3_load    = 36.0;
    m.dram_load  = 200.0;
    m.gpu_global = 400.0;
    m.gpu_shared = 30.0;
    m.gpu_local  = 100.0;
    m.store_base = 4.0;  // LFB + store-forward latency
    return m;
}

// ── X86_64CostModel operand-aware, per-kind cost ──────────────────────────
//
// Applies the following TTI-style discounts on top of the flat cost(op)
// table:
//
//   - Mul  with a constant power-of-two operand  → shift cost (1.0 cyc
//     latency, 0.5 throughput, 1 uop) instead of IMUL (3.0/1.0/3).
//   - UDiv / SDiv with a constant power-of-two    → shift cost (1.0/0.5/1)
//     instead of IDIV (25.0/25.0/10). URem/SRem additionally need an
//     AND mask so they cost ~2 uops but still 1.0 latency.
//   - Add / Sub / Xor / Or / And with constant 0  → 0.0 (the op is a
//     no-op; either the input is forwarded unchanged or the result is
//     constant-folded).
//   - Mul with constant 1                          → 0.0 (identity).
//   - Shl / LShr / AShr with constant 0 shift     → 0.0 (no-op).
//
// For CostKind::CodeSize we report the uops count of the resulting
// instruction (so a foldable mul-by-8 contributes 1, not 3). For
// CostKind::Latency we report the latency of the resulting instruction.
// For CostKind::ReciprocalThroughput we report its throughput_cycles.
double X86_64CostModel::cost(ir::Opcode op, CostKind kind,
                              const std::vector<OperandValueInfo>& operands) const {
    // First, resolve the *effective* OpCost after operand-aware discounts.
    OpCost c = this->cost(op);  // base table
    bool folded_to_shift = false;
    bool folded_to_noop  = false;

    switch (op) {
        case ir::Opcode::Mul:
            if (any_operand_is(operands, operand_is_one)) {
                folded_to_noop = true;
            } else if (any_operand_is(operands, operand_is_pow2)) {
                // LEA-foldable to a shift: Shl on x86_64 has 1.0 cyc
                // latency, 0.5 throughput, 1 uop.
                OpCost shift = this->cost(ir::Opcode::Shl);
                c = shift;
                folded_to_shift = true;
            }
            break;
        case ir::Opcode::UDiv:
        case ir::Opcode::SDiv:
            if (any_operand_is(operands, operand_is_pow2)) {
                OpCost shift = this->cost(ir::Opcode::LShr);
                c = shift;
                folded_to_shift = true;
            }
            break;
        case ir::Opcode::URem:
        case ir::Opcode::SRem:
            if (any_operand_is(operands, operand_is_pow2)) {
                // x & (pow2 - 1): an AND plus the shift-subtraction, but
                // realistically one AND op on x86. Charge as And.
                OpCost and_op = this->cost(ir::Opcode::And);
                c = and_op;
                folded_to_shift = true;  // conceptually a "strength-reduce"
            }
            break;
        case ir::Opcode::Add:
        case ir::Opcode::Sub:
        case ir::Opcode::Xor:
        case ir::Opcode::Or:
        case ir::Opcode::And:
            if (any_operand_is(operands, operand_is_zero)) {
                folded_to_noop = true;
            }
            break;
        case ir::Opcode::Shl:
        case ir::Opcode::LShr:
        case ir::Opcode::AShr:
            if (any_operand_is(operands, operand_is_zero)) {
                folded_to_noop = true;
            }
            break;
        default:
            break;
    }

    if (folded_to_noop) {
        return 0.0;  // No-op is free under every CostKind.
    }

    (void)folded_to_shift;  // currently a marker; the values in `c` already
                            // reflect the post-fold instruction's cost.

    switch (kind) {
        case CostKind::Latency:               return c.latency_cycles;
        case CostKind::CodeSize:              return static_cast<double>(c.uops);
        case CostKind::ReciprocalThroughput:
        default:                              return c.throughput_cycles;
    }
}

// ── AArch64 (Cortex-A78-class) ────────────────────────────────────────────
// Ports (simplified):
//   bit 0 → ALU pipe 0
//   bit 1 → ALU pipe 1
//   bit 2 → LS pipe
//   bit 3 → FP / vector pipe

OpCost AArch64CostModel::cost(ir::Opcode op) const {
    OpCost c = GenericCostModel{}.cost(op);
    switch (op) {
        case ir::Opcode::Add:
        case ir::Opcode::Sub:
        case ir::Opcode::And:
        case ir::Opcode::Or:
        case ir::Opcode::Xor:
        case ir::Opcode::Shl:
        case ir::Opcode::LShr:
        case ir::Opcode::AShr:
            c.latency_cycles = 1.0; c.throughput_cycles = 0.5;
            c.port_mask = 0b00000011; c.uops = 1;
            break;
        case ir::Opcode::Mul:
            c.latency_cycles = 3.0; c.throughput_cycles = 1.0;
            c.port_mask = 0b00001000; c.uops = 1; // FP/MAC pipe
            break;
        case ir::Opcode::UDiv:
        case ir::Opcode::SDiv:
        case ir::Opcode::URem:
        case ir::Opcode::SRem:
            c.latency_cycles = 15.0; c.throughput_cycles = 15.0;
            c.port_mask = 0b00001000; c.uops = 2;
            break;
        case ir::Opcode::Load:
            c.latency_cycles = 4.0; c.throughput_cycles = 1.0;
            c.port_mask = 0b00000100; c.uops = 1;
            break;
        case ir::Opcode::Store:
            c.latency_cycles = 1.0; c.throughput_cycles = 1.0;
            c.port_mask = 0b00000100; c.uops = 1;
            break;
        case ir::Opcode::Br:
        case ir::Opcode::Switch:
        case ir::Opcode::SwitchInst:
            c.latency_cycles = 1.0; c.throughput_cycles = 1.0;
            c.port_mask = 0b00000001; c.uops = 1; // BR branch unit
            break;
        case ir::Opcode::Call:
            c.latency_cycles = 15.0; c.throughput_cycles = 15.0;
            c.port_mask = 0b00000111; c.uops = 3;
            break;
        case ir::Opcode::FAdd:
        case ir::Opcode::FSub:
            c.latency_cycles = 4.0; c.throughput_cycles = 0.5;
            c.port_mask = 0b00001000; c.uops = 1;
            break;
        case ir::Opcode::FMul:
            c.latency_cycles = 4.0; c.throughput_cycles = 0.5;
            c.port_mask = 0b00001000; c.uops = 1;
            break;
        case ir::Opcode::FDiv:
            c.latency_cycles = 12.0; c.throughput_cycles = 12.0;
            c.port_mask = 0b00001000; c.uops = 2;
            break;
        default:
            break;
    }
    return c;
}

MemoryLatencies AArch64CostModel::memory() const {
    MemoryLatencies m;
    m.l1_load    = 3.0;
    m.l2_load    = 11.0;
    m.l3_load    = 40.0;
    m.dram_load  = 180.0;
    m.gpu_global = 400.0;
    m.gpu_shared = 30.0;
    m.gpu_local  = 100.0;
    m.store_base = 3.0;
    return m;
}

// ── PTX (SM_80) ───────────────────────────────────────────────────────────
//
// PTX is a virtual ISA; the "cost" here is a coarse model of
// single-warp scalar execution on an SM (one lane per cycle).
// Real GPUs schedule 32 lanes per warp, but for a structural
// comparison the relative ordering is what matters.

OpCost PTXCostModel::cost(ir::Opcode op) const {
    OpCost c = GenericCostModel{}.cost(op);
    switch (op) {
        case ir::Opcode::Add:
        case ir::Opcode::Sub:
        case ir::Opcode::And:
        case ir::Opcode::Or:
        case ir::Opcode::Xor:
            c.latency_cycles = 4.0; c.throughput_cycles = 1.0;
            c.port_mask = 0b00000011; c.uops = 1; // INT32 pipe
            break;
        case ir::Opcode::Mul:
            c.latency_cycles = 6.0; c.throughput_cycles = 1.0;
            c.port_mask = 0b00000100; c.uops = 1; // IMAD pipe
            break;
        case ir::Opcode::UDiv:
        case ir::Opcode::SDiv:
        case ir::Opcode::URem:
        case ir::Opcode::SRem:
            c.latency_cycles = 80.0; c.throughput_cycles = 80.0;
            c.port_mask = 0b00000100; c.uops = 8; // microcoded
            break;
        case ir::Opcode::Shl:
        case ir::Opcode::LShr:
        case ir::Opcode::AShr:
            c.latency_cycles = 4.0; c.throughput_cycles = 1.0;
            c.port_mask = 0b00000011; c.uops = 1;
            break;
        case ir::Opcode::Load:
            c.latency_cycles = 100.0; c.throughput_cycles = 1.0;
            c.port_mask = 0b00001000; c.uops = 2; // LSU
            break;
        case ir::Opcode::Store:
            c.latency_cycles = 1.0; c.throughput_cycles = 1.0;
            c.port_mask = 0b00001000; c.uops = 2; // LSU
            break;
        case ir::Opcode::Br:
        case ir::Opcode::Switch:
        case ir::Opcode::SwitchInst:
            c.latency_cycles = 4.0; c.throughput_cycles = 4.0;
            c.port_mask = 0b00000001; c.uops = 2; // warp divergence
            break;
        case ir::Opcode::Call:
            c.latency_cycles = 50.0; c.throughput_cycles = 50.0;
            c.port_mask = 0b00001111; c.uops = 8;
            break;
        case ir::Opcode::FAdd:
        case ir::Opcode::FSub:
            c.latency_cycles = 4.0; c.throughput_cycles = 1.0;
            c.port_mask = 0b00000100; c.uops = 1; // FMA pipe
            break;
        case ir::Opcode::FMul:
            c.latency_cycles = 6.0; c.throughput_cycles = 1.0;
            c.port_mask = 0b00000100; c.uops = 1;
            break;
        case ir::Opcode::FDiv:
            c.latency_cycles = 30.0; c.throughput_cycles = 4.0;
            c.port_mask = 0b00000100; c.uops = 4; // SFU
            break;
        default:
            break;
    }
    return c;
}

MemoryLatencies PTXCostModel::memory() const {
    MemoryLatencies m;
    m.l1_load    = 30.0;   // L1 / shared
    m.l2_load    = 100.0;  // L2
    m.l3_load    = 100.0;
    m.dram_load  = 400.0;  // HBM
    m.gpu_global = 400.0;
    m.gpu_shared = 30.0;
    m.gpu_local  = 100.0;
    m.store_base = 1.0;
    return m;
}

// ── Default per-kind, operand-aware cost ────────────────────────────────────
//
// Falls back to the flat cost(op) table. Arch-specific overrides can
// replace this with richer logic (see X86_64CostModel::cost below).
//
// CostKind mapping:
//   ReciprocalThroughput → OpCost::throughput_cycles
//   Latency               → OpCost::latency_cycles
//   CodeSize              → static_cast<double>(OpCost::uops)
//
// (uops is a reasonable CodeSize proxy at the IR level: the post- lowering
// bytes emitted for an instruction is roughly proportional to its uop
// count for out-of-order x86 cores.)
double CostModel::cost(ir::Opcode op, CostKind kind,
                        const std::vector<OperandValueInfo>& /*operands*/) const {
    OpCost c = this->cost(op);  // virtual dispatch — picks arch-specific table
    switch (kind) {
        case CostKind::Latency:               return c.latency_cycles;
        case CostKind::CodeSize:              return static_cast<double>(c.uops);
        case CostKind::ReciprocalThroughput:
        default:                              return c.throughput_cycles;
    }
}

// ── Derive OperandValueInfo from an IR value ───────────────────────────────
OperandValueInfo CostModel::operand_info(const ir::Value& v) {
    OperandValueInfo info;
    // bitwidth (0 if unknown / non-integer)
    if (v.type()) {
        size_t bw = v.type()->bit_width();
        info.bitwidth = (bw > 0) ? static_cast<unsigned>(bw) : 0u;
    }
    // ConstantInt?
    if (auto* ci = dynamic_cast<const ir::ConstantInt*>(&v)) {
        info.is_constant = true;
        info.constant_value = ci->value();
        int64_t val = ci->value();
        info.is_zero = (val == 0);
        info.is_one  = (val == 1);
        info.is_all_ones = (val == -1);
        // is_power_of_two: treat the bit pattern as unsigned.
        uint64_t uval = static_cast<uint64_t>(val);
        info.is_power_of_two = is_pow2_u64(uval);
    }
    return info;
}

// ── Factory ───────────────────────────────────────────────────────────────

std::shared_ptr<const CostModel> make_cost_model(Arch arch) {
    switch (arch) {
        case Arch::X86_64:  return std::make_shared<X86_64CostModel>();
        case Arch::AArch64: return std::make_shared<AArch64CostModel>();
        case Arch::PTX:     return std::make_shared<PTXCostModel>();
        case Arch::Generic:
        default:            return std::make_shared<GenericCostModel>();
    }
}

std::shared_ptr<const CostModel> make_cost_model_for_name(const std::string& name) {
    if (iequals_prefix(name, "x86") || iequals_prefix(name, "amd64") ||
        iequals_prefix(name, "intel")) {
        return make_cost_model(Arch::X86_64);
    }
    if (iequals_prefix(name, "aarch64") || iequals_prefix(name, "arm64") ||
        iequals_prefix(name, "arm")) {
        return make_cost_model(Arch::AArch64);
    }
    if (iequals_prefix(name, "ptx") || iequals_prefix(name, "nvptx") ||
        iequals_prefix(name, "sm_") || iequals_prefix(name, "cuda")) {
        return make_cost_model(Arch::PTX);
    }
    return make_cost_model(Arch::Generic);
}

// ── UOpsCostModel ──────────────────────────────────────────────────────────
//
// Per-arch uOps tables (Agner Fog-style approximations). These return
// the number of fused-domain micro-ops each instruction would issue.
//
// For x86_64 (Skylake-class):
//   - Most simple ALU ops (Add/Sub/And/Or/Xor/Shl/...) = 1 uop
//   - Mul                                                  = 3 uops
//   - UDiv/SDiv/URem/SRem                                  = 10-20 uops (microcoded)
//   - Load                                                 = 2 uops
//   - Store                                                = 2 uops
//   - ICmp/FCmp                                            = 1 uop
//   - Select                                               = 1 uop
//   - Br/Switch                                            = 1-2 uops
//   - Call                                                 = 8 uops (rough)
//   - Phi                                                  = 0 uops (renamed)
//   - Casts (Trunc/ZExt/SExt/BitCast/...)                  = 1 uop
//   - Alloca                                               = 0 uops (stack-pointer adjust)
//
// For other arches we fall back to the per-arch CostModel's existing
// uops field (which is also an approximation but consistent with the
// latency/throughput tables already in use).
//
namespace {

OpCost uops_cost_x86_64(ir::Opcode op) {
    OpCost c{};  // zero-init
    c.latency_cycles = 1.0;
    c.throughput_cycles = 1.0;
    c.port_mask = 0;
    c.uops = 1;
    switch (op) {
        case ir::Opcode::Add:
        case ir::Opcode::Sub:
        case ir::Opcode::And:
        case ir::Opcode::Or:
        case ir::Opcode::Xor:
        case ir::Opcode::Shl:
        case ir::Opcode::LShr:
        case ir::Opcode::AShr:
            c.uops = 1; c.latency_cycles = 1.0; c.throughput_cycles = 0.25;
            break;
        case ir::Opcode::Mul:
            c.uops = 3; c.latency_cycles = 3.0; c.throughput_cycles = 1.0;
            break;
        case ir::Opcode::UDiv:
        case ir::Opcode::SDiv:
        case ir::Opcode::URem:
        case ir::Opcode::SRem:
            c.uops = 20; c.latency_cycles = 25.0; c.throughput_cycles = 25.0;
            break;
        case ir::Opcode::Load:
            c.uops = 2; c.latency_cycles = 5.0; c.throughput_cycles = 0.5;
            break;
        case ir::Opcode::Store:
            c.uops = 2; c.latency_cycles = 1.0; c.throughput_cycles = 1.0;
            break;
        case ir::Opcode::ICmp:
        case ir::Opcode::FCmp:
            c.uops = 1; c.latency_cycles = 1.0; c.throughput_cycles = 0.5;
            break;
        case ir::Opcode::Select:
            c.uops = 1; c.latency_cycles = 2.0; c.throughput_cycles = 1.0;
            break;
        case ir::Opcode::Br:
        case ir::Opcode::Switch:
        case ir::Opcode::SwitchInst:
            c.uops = 2; c.latency_cycles = 1.0; c.throughput_cycles = 1.0;
            break;
        case ir::Opcode::Ret:
            c.uops = 1; c.latency_cycles = 1.0; c.throughput_cycles = 1.0;
            break;
        case ir::Opcode::Call:
            c.uops = 8; c.latency_cycles = 30.0; c.throughput_cycles = 30.0;
            break;
        case ir::Opcode::Alloca:
            c.uops = 0; c.latency_cycles = 0.0; c.throughput_cycles = 1.0;
            break;
        case ir::Opcode::Phi:
            c.uops = 0; c.latency_cycles = 0.0; c.throughput_cycles = 0.0;
            break;
        // FP — approximated as same uop count as the corresponding int op.
        case ir::Opcode::FAdd:
        case ir::Opcode::FSub:
            c.uops = 1; c.latency_cycles = 4.0; c.throughput_cycles = 0.5;
            break;
        case ir::Opcode::FMul:
            c.uops = 1; c.latency_cycles = 4.0; c.throughput_cycles = 0.5;
            break;
        case ir::Opcode::FDiv:
        case ir::Opcode::FRem:
            c.uops = 4; c.latency_cycles = 18.0; c.throughput_cycles = 18.0;
            break;
        default:
            c.uops = 1; c.latency_cycles = 1.0; c.throughput_cycles = 1.0;
            break;
    }
    return c;
}

} // anonymous namespace

UOpsCostModel::UOpsCostModel(Arch arch)
    : arch_(arch), delegate_(make_cost_model(arch)) {}

OpCost UOpsCostModel::cost(ir::Opcode op) const {
    if (arch_ == Arch::X86_64) {
        return uops_cost_x86_64(op);
    }
    // Other arches: reuse the per-arch CostModel's uops field (already
    // an approximation; consistent with the latency/throughput tables).
    return delegate_->cost(op);
}

MemoryLatencies UOpsCostModel::memory() const { return delegate_->memory(); }
unsigned UOpsCostModel::issue_width() const { return delegate_->issue_width(); }
unsigned UOpsCostModel::num_ports() const { return delegate_->num_ports(); }
const char* UOpsCostModel::arch_name() const {
    static const char x86[]   = "uops-x86_64";
    static const char arm[]   = "uops-aarch64";
    static const char ptx[]   = "uops-ptx";
    static const char gen[]   = "uops-generic";
    switch (arch_) {
        case Arch::X86_64:  return x86;
        case Arch::AArch64: return arm;
        case Arch::PTX:     return ptx;
        case Arch::Generic:
        default:            return gen;
    }
}

double UOpsCostModel::cost(ir::Opcode op, CostKind kind,
                            const std::vector<OperandValueInfo>& operands) const {
    OpCost c = this->cost(op);  // arch-specific uops table
    // Operand-aware discounts (same shape as X86_64CostModel): a mul by a
    // power-of-two becomes a shift (1 uop), a div by a power-of-two becomes
    // a shift (1 uop), add/sub/xor/or/and by zero is 0 uops, mul by one is
    // 0 uops, shift by zero is 0 uops.
    auto apply_discounts = [&]() -> double {
        switch (op) {
            case ir::Opcode::Mul:
                if (any_operand_is(operands, operand_is_one))  return 0.0;
                if (any_operand_is(operands, operand_is_pow2)) {
                    // Reduced to a shift — 1 uop on x86-class arches.
                    return 1.0;
                }
                return static_cast<double>(c.uops);
            case ir::Opcode::UDiv:
            case ir::Opcode::SDiv:
            case ir::Opcode::URem:
            case ir::Opcode::SRem:
                if (any_operand_is(operands, operand_is_pow2)) {
                    // Reduced to a shift (and mask for rem) — 1-2 uops.
                    return (op == ir::Opcode::URem || op == ir::Opcode::SRem) ? 2.0 : 1.0;
                }
                return static_cast<double>(c.uops);
            case ir::Opcode::Add:
            case ir::Opcode::Sub:
            case ir::Opcode::Xor:
            case ir::Opcode::Or:
            case ir::Opcode::And:
                if (any_operand_is(operands, operand_is_zero))  return 0.0;
                return static_cast<double>(c.uops);
            case ir::Opcode::Shl:
            case ir::Opcode::LShr:
            case ir::Opcode::AShr:
                if (any_operand_is(operands, operand_is_zero))  return 0.0;
                return static_cast<double>(c.uops);
            default:
                return static_cast<double>(c.uops);
        }
    };

    switch (kind) {
        case CostKind::Latency:
            // uOps model doesn't predict latency well — fall back to the
            // delegate's latency table. Apply the same operand-aware
            // discounts though (a foldable mul-by-pow2 has shift latency).
            // Reuse the discounted uop count as a latency proxy only when
            // it's strictly lower than the delegate's latency.
            {
                double delegate_lat = delegate_->cost(op, CostKind::Latency, operands);
                double discounted_uops = apply_discounts();
                // Heuristic: a no-op (0 uops) has 0 latency; a shift-folded
                // mul has shift latency (1.0). Otherwise keep delegate latency.
                if (discounted_uops == 0.0)  return 0.0;
                if (op == ir::Opcode::Mul && any_operand_is(operands, operand_is_pow2)) {
                    return 1.0;  // shift latency
                }
                if ((op == ir::Opcode::UDiv || op == ir::Opcode::SDiv) &&
                    any_operand_is(operands, operand_is_pow2)) {
                    return 1.0;  // shift latency
                }
                return delegate_lat;
            }
        case CostKind::CodeSize:
        case CostKind::ReciprocalThroughput:
        default:
            return apply_discounts();
    }
}

std::shared_ptr<const CostModel> make_uops_cost_model(Arch arch) {
    return std::make_shared<UOpsCostModel>(arch);
}

std::shared_ptr<const CostModel> make_uops_cost_model_for_name(const std::string& arch_name) {
    if (iequals_prefix(arch_name, "x86") || iequals_prefix(arch_name, "amd64") ||
        iequals_prefix(arch_name, "intel")) {
        return make_uops_cost_model(Arch::X86_64);
    }
    if (iequals_prefix(arch_name, "aarch64") || iequals_prefix(arch_name, "arm64") ||
        iequals_prefix(arch_name, "arm")) {
        return make_uops_cost_model(Arch::AArch64);
    }
    if (iequals_prefix(arch_name, "ptx") || iequals_prefix(arch_name, "nvptx") ||
        iequals_prefix(arch_name, "sm_") || iequals_prefix(arch_name, "cuda")) {
        return make_uops_cost_model(Arch::PTX);
    }
    return make_uops_cost_model(Arch::Generic);
}

} // namespace clunk::evaluator
