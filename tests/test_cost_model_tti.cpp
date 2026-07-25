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
 * Clunk CostModel TTI Tests — I5 / R4.A + R4.G + R3.C.
 *
 * Tests the TTI-style InstructionCost API (CostKind + OperandValueInfo),
 * the operand-aware x86_64 cost discounts, the GPU occupancy cliff
 * penalty, and the UOpsCostModel two-tier final ranker.
 *
 * Uses the same CHECK(cond, msg) macro pattern as test_evaluator.cpp.
 */
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "clunk/IR/Type.h"
#include "clunk/IR/Value.h"
#include "clunk/IR/Instruction.h"
#include "clunk/IR/BasicBlock.h"
#include "clunk/IR/Function.h"
#include "clunk/IR/Module.h"
#include "clunk/IR/IRBuilder.h"
#include "clunk/Evaluator/CostModel.h"
#include "clunk/GPU/ArchLimits.h"
#include "clunk/GPU/OccupancyModel.h"

static int g_pass = 0, g_fail = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { std::cerr << "FAIL: " << msg << " at " << __FILE__ << ":" << __LINE__ << "\n"; g_fail++; } \
    else { g_pass++; } \
} while(0)

using namespace clunk::ir;
using namespace clunk::evaluator;
using namespace clunk::gpu;

// ─────────────────────────────────────────────────────────────────────────────
//  Helpers
// ─────────────────────────────────────────────────────────────────────────────

// Build an OperandValueInfo for a constant power-of-two (e.g. 8 = 2^3).
static OperandValueInfo make_pow2_operand(int64_t val, unsigned bw = 32) {
    OperandValueInfo o;
    o.is_constant = true;
    o.constant_value = val;
    o.bitwidth = bw;
    o.is_zero = (val == 0);
    o.is_one = (val == 1);
    o.is_all_ones = (val == -1);
    uint64_t u = static_cast<uint64_t>(val);
    o.is_power_of_two = (u != 0 && (u & (u - 1)) == 0);
    return o;
}

static OperandValueInfo make_const_operand(int64_t val, unsigned bw = 32) {
    return make_pow2_operand(val, bw);  // sets is_power_of_two correctly
}

// ─────────────────────────────────────────────────────────────────────────────
//  Test 1: CostKind::ReciprocalThroughput default matches cost(op)
// ─────────────────────────────────────────────────────────────────────────────
//
// The new TTI-style cost(op, CostKind) overload must return the same
// ReciprocalThroughput as the legacy OpCost::throughput_cycles field.
//
static void test_cost_kind_reciprocal_throughput_default() {
    auto x86 = make_cost_model(Arch::X86_64);
    auto arm = make_cost_model(Arch::AArch64);
    auto ptx = make_cost_model(Arch::PTX);
    auto gen = make_cost_model(Arch::Generic);

    const Opcode ops[] = {Opcode::Add, Opcode::Mul, Opcode::UDiv,
                          Opcode::Load, Opcode::Store, Opcode::Br,
                          Opcode::ICmp, Opcode::Call, Opcode::Phi};

    for (auto* model : {x86.get(), arm.get(), ptx.get(), gen.get()}) {
        for (Opcode op : ops) {
            OpCost legacy = model->cost(op);
            double tti = model->cost(op, CostKind::ReciprocalThroughput);
            CHECK(std::fabs(tti - legacy.throughput_cycles) < 1e-9,
                  "ReciprocalThroughput default matches cost(op).throughput_cycles");
        }
    }

    // Also check Latency and CodeSize defaults.
    for (auto* model : {x86.get(), arm.get(), ptx.get(), gen.get()}) {
        OpCost legacy = model->cost(Opcode::Mul);
        double lat = model->cost(Opcode::Mul, CostKind::Latency);
        double csz = model->cost(Opcode::Mul, CostKind::CodeSize);
        CHECK(std::fabs(lat - legacy.latency_cycles) < 1e-9,
              "Latency default matches cost(op).latency_cycles");
        CHECK(std::fabs(csz - static_cast<double>(legacy.uops)) < 1e-9,
              "CodeSize default matches cost(op).uops");
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Test 2: Operand-aware cost — Mul by power-of-two is shift cost
// ─────────────────────────────────────────────────────────────────────────────
//
// On x86_64, `mul x, 8` is LEA-foldable to a shift. The operand-aware
// cost(op, kind, operands) override must return the shift's cost, not
// IMUL's cost (3.0 cycles latency, 1.0 throughput, 3 uops).
//
static void test_operand_aware_cost_mul_power_of_two() {
    auto x86 = make_cost_model(Arch::X86_64);

    // Baseline: Mul without operand context.
    OpCost mul_base = x86->cost(Opcode::Mul);
    OpCost shift_base = x86->cost(Opcode::Shl);

    std::vector<OperandValueInfo> operands = {
        make_pow2_operand(8),  // x * 8 — power of two
    };

    double mul_pow2_throughput = x86->cost(Opcode::Mul,
                                            CostKind::ReciprocalThroughput,
                                            operands);
    double mul_pow2_latency = x86->cost(Opcode::Mul,
                                         CostKind::Latency,
                                         operands);
    double mul_pow2_codesize = x86->cost(Opcode::Mul,
                                          CostKind::CodeSize,
                                          operands);

    // The discounted throughput must be the shift's throughput, not mul's.
    CHECK(std::fabs(mul_pow2_throughput - shift_base.throughput_cycles) < 1e-9,
          "mul x,8 throughput == shift throughput (not mul throughput)");
    CHECK(std::fabs(mul_pow2_latency - shift_base.latency_cycles) < 1e-9,
          "mul x,8 latency == shift latency (not mul latency)");
    CHECK(mul_pow2_codesize < static_cast<double>(mul_base.uops),
          "mul x,8 CodeSize < mul CodeSize");

    // Mul by 1 is a no-op — must return 0 under every CostKind.
    std::vector<OperandValueInfo> one_operand = { make_const_operand(1) };
    CHECK(x86->cost(Opcode::Mul, CostKind::ReciprocalThroughput, one_operand) == 0.0,
          "mul x,1 is a no-op (throughput = 0)");
    CHECK(x86->cost(Opcode::Mul, CostKind::Latency, one_operand) == 0.0,
          "mul x,1 is a no-op (latency = 0)");
    CHECK(x86->cost(Opcode::Mul, CostKind::CodeSize, one_operand) == 0.0,
          "mul x,1 is a no-op (codesize = 0)");

    // Mul by a non-power-of-two constant (e.g. 7) must NOT be discounted.
    std::vector<OperandValueInfo> non_pow2 = { make_const_operand(7) };
    double mul7 = x86->cost(Opcode::Mul, CostKind::ReciprocalThroughput, non_pow2);
    CHECK(std::fabs(mul7 - mul_base.throughput_cycles) < 1e-9,
          "mul x,7 is NOT folded (no pow2 operand)");
}

// ─────────────────────────────────────────────────────────────────────────────
//  Test 3: Operand-aware cost — UDiv/SDiv by power-of-two is shift cost
// ─────────────────────────────────────────────────────────────────────────────
//
// On x86_64, `udiv x, 8` lowers to `shr x, 3`. The operand-aware cost
// override must return the shift's cost, not IDIV's cost
// (25.0 cycles latency, 25.0 throughput, 10 uops).
//
static void test_operand_aware_cost_div_power_of_two() {
    auto x86 = make_cost_model(Arch::X86_64);

    OpCost div_base = x86->cost(Opcode::UDiv);
    OpCost shift_base = x86->cost(Opcode::LShr);

    std::vector<OperandValueInfo> operands = { make_pow2_operand(8) };

    double div_pow2_throughput = x86->cost(Opcode::UDiv,
                                            CostKind::ReciprocalThroughput,
                                            operands);
    double div_pow2_latency = x86->cost(Opcode::UDiv,
                                         CostKind::Latency,
                                         operands);
    double div_pow2_codesize = x86->cost(Opcode::UDiv,
                                          CostKind::CodeSize,
                                          operands);

    CHECK(std::fabs(div_pow2_throughput - shift_base.throughput_cycles) < 1e-9,
          "udiv x,8 throughput == shift throughput (not div throughput)");
    CHECK(std::fabs(div_pow2_latency - shift_base.latency_cycles) < 1e-9,
          "udiv x,8 latency == shift latency (not div latency)");
    CHECK(div_pow2_codesize < static_cast<double>(div_base.uops),
          "udiv x,8 CodeSize < div CodeSize");

    // SDiv by power-of-two is also strength-reduced (with sign-fixup, but
    // the cost model treats it like a shift).
    double sdiv_pow2 = x86->cost(Opcode::SDiv,
                                  CostKind::ReciprocalThroughput, operands);
    CHECK(sdiv_pow2 < div_base.throughput_cycles,
          "sdiv x,8 throughput < div throughput");

    // Add/Sub/Xor/Or/And with a zero operand → 0 (no-op).
    std::vector<OperandValueInfo> zero_operand = { make_const_operand(0) };
    for (Opcode op : {Opcode::Add, Opcode::Sub, Opcode::Xor,
                      Opcode::Or, Opcode::And}) {
        CHECK(x86->cost(op, CostKind::ReciprocalThroughput, zero_operand) == 0.0,
              "add/sub/xor/or/and x,0 is a no-op");
    }

    // Shl/LShr/AShr with a zero shift amount → 0 (no-op).
    for (Opcode op : {Opcode::Shl, Opcode::LShr, Opcode::AShr}) {
        CHECK(x86->cost(op, CostKind::ReciprocalThroughput, zero_operand) == 0.0,
              "shl/lshr/ashr x,0 is a no-op");
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Test 3b: CostModel::operand_info() derives OperandValueInfo from ir::Value
// ─────────────────────────────────────────────────────────────────────────────
static void test_operand_info_from_value() {
    Module mod;
    TypeContext& ctx = mod.type_context();

    auto c8  = ConstantInt::get(ctx, 8, 32);
    auto c0  = ConstantInt::get(ctx, 0, 32);
    auto c1  = ConstantInt::get(ctx, 1, 32);
    auto cm1 = ConstantInt::get(ctx, -1, 32);
    auto c7  = ConstantInt::get(ctx, 7, 32);

    OperandValueInfo i8  = CostModel::operand_info(*c8);
    OperandValueInfo i0  = CostModel::operand_info(*c0);
    OperandValueInfo i1  = CostModel::operand_info(*c1);
    OperandValueInfo im1 = CostModel::operand_info(*cm1);
    OperandValueInfo i7  = CostModel::operand_info(*c7);

    CHECK(i8.is_constant && i8.constant_value == 8 && i8.is_power_of_two,
          "operand_info(8) → is_constant, pow2");
    CHECK(i8.bitwidth == 32, "operand_info(8) → bitwidth=32");
    CHECK(i0.is_constant && i0.is_zero && !i0.is_power_of_two,
          "operand_info(0) → is_zero, not pow2");
    CHECK(i1.is_constant && i1.is_one && i1.is_power_of_two,
          "operand_info(1) → is_one AND pow2");
    CHECK(im1.is_constant && im1.is_all_ones,
          "operand_info(-1) → is_all_ones");
    CHECK(i7.is_constant && !i7.is_power_of_two,
          "operand_info(7) → constant, NOT pow2");

    // Non-constant: an Instruction is a Value too — its operand_info
    // should report is_constant=false but a populated bitwidth.
    auto dummy_inst = std::make_shared<Instruction>(Opcode::Add, ctx.int32(), "tmp");
    OperandValueInfo ii = CostModel::operand_info(*dummy_inst);
    CHECK(!ii.is_constant, "operand_info(non-constant Instruction) → !is_constant");
    CHECK(ii.bitwidth == 32, "operand_info(i32 Instruction) → bitwidth=32");
}

// ─────────────────────────────────────────────────────────────────────────────
//  Test 4: SM_80, 32 regs/thread, 256 threads/block, 0 smem
//           → occupancy_ratio ~ 1.0 (full occupancy)
// ─────────────────────────────────────────────────────────────────────────────
//
// The spec's headline example mentioned 0.75 for 32 regs on SM_80, but
// the actual NVIDIA occupancy calc gives 100% for 32 regs/thread (no
// rounding, no register pressure limit). 32*32 = 1024 regs/warp = 8192
// regs/block, 65536/8192 = 8 blocks/SM, 8*8 = 64 warps = 100% of max.
//
// We assert on the real value (~1.0) and verify the new occupancy_ratio
// field is consistent with occupancy_pct.
//
static void test_occupancy_sm80_32_regs() {
    OccupancyInfo info = compute_occupancy(GpuArch::SM_80,
                                            256,   // threads_per_block
                                            32,    // regs_per_thread
                                            0);    // shared_mem_per_block
    CHECK(info.blocks_per_sm == 8, "blocks_per_sm=8 for 32 regs, 256 tpb");
    CHECK(info.achieved_warps_per_sm == 64, "achieved_warps=64");
    CHECK(info.occupancy_pct > 99.9 && info.occupancy_pct <= 100.0,
          "occupancy_pct ~100 for 32 regs, 256 tpb");
    CHECK(std::fabs(info.occupancy_ratio - 1.0) < 1e-9,
          "occupancy_ratio == 1.0 (full occupancy)");

    // New fields populated correctly.
    CHECK(info.threads_per_sm == 8 * 256, "threads_per_sm = blocks * tpb");
    CHECK(info.shared_mem_per_block == 0, "shared_mem_per_block echoed");
    CHECK(info.registers_per_sm == 65536, "registers_per_sm echoed");
    CHECK(info.max_threads_per_sm == 2048, "max_threads_per_sm echoed");
    CHECK(info.max_blocks_per_sm == 32, "max_blocks_per_sm echoed");
}

// ─────────────────────────────────────────────────────────────────────────────
//  Test 5: SM_80, 34 regs/thread, 256 threads/block, 0 smem
//           → occupancy_ratio ~ 0.75 (the cliff!)
// ─────────────────────────────────────────────────────────────────────────────
//
// The cliff: 33 or 34 regs/thread rounds up to 36 (granularity=4) on
// SM_80. 36*32 = 1152 → rounded up to 1280 regs/warp. 1280*8 = 10240
// regs/block. 65536/10240 = 6 blocks/SM (down from 8). 6*8 = 48 warps
// = 75% occupancy. So 32 → 1.0 and 34 → 0.75 is a real cliff — a 25%
// occupancy drop from a single additional register.
//
static void test_occupancy_sm80_34_regs() {
    OccupancyInfo info32 = compute_occupancy(GpuArch::SM_80, 256, 32, 0);
    OccupancyInfo info34 = compute_occupancy(GpuArch::SM_80, 256, 34, 0);

    CHECK(info34.regs_per_thread_rounded == 36,
          "34 regs/thread rounds up to 36 (granularity=4)");
    CHECK(info34.blocks_per_sm == 6,
          "blocks_per_sm drops to 6 (was 8 at 32 regs)");
    CHECK(info34.achieved_warps_per_sm == 48,
          "achieved_warps drops to 48 (was 64 at 32 regs)");
    CHECK(info34.occupancy_ratio < info32.occupancy_ratio,
          "occupancy_ratio at 34 regs < occupancy_ratio at 32 regs (cliff!)");
    CHECK(std::fabs(info34.occupancy_ratio - 0.75) < 1e-9,
          "occupancy_ratio == 0.75 at 34 regs (the cliff)");

    // 32 → 33 should also cross the cliff (33 rounds up to 36).
    OccupancyInfo info33 = compute_occupancy(GpuArch::SM_80, 256, 33, 0);
    CHECK(std::fabs(info33.occupancy_ratio - 0.75) < 1e-9,
          "33 regs → also 0.75 (rounds to 36)");

    // Bigger cliff: 41 regs (rounds to 44) → 1536 regs/warp → 5 blocks/SM
    // → 40 warps → 62.5%.
    OccupancyInfo info41 = compute_occupancy(GpuArch::SM_80, 256, 41, 0);
    CHECK(std::fabs(info41.occupancy_ratio - 0.625) < 1e-9,
          "41 regs → 0.625 (next cliff at granularity-round to 44)");
}

// ─────────────────────────────────────────────────────────────────────────────
//  Test 5b: occupancy_penalty — cliff crossing is penalised
// ─────────────────────────────────────────────────────────────────────────────
static void test_occupancy_penalty_cliff() {
    // Going from 32 → 33 regs (256 tpb, 0 smem) on SM_80 crosses the cliff
    // (1.0 → 0.75). The penalty should be ~25.0 (= 0.25 * 100).
    double penalty_32_to_33 = occupancy_penalty(GpuArch::SM_80,
                                                  32 /*baseline_regs*/,
                                                  33 /*candidate_regs*/,
                                                  256, 0);
    CHECK(penalty_32_to_33 > 24.9 && penalty_32_to_33 < 25.1,
          "occupancy_penalty(32→33) ~= 25.0 (1.0 → 0.75 cliff)");

    // Going from 30 → 31 regs should NOT cross a cliff on SM_80 — both
    // round to 32 and 32 (no round-up). Penalty should be 0.
    double penalty_30_to_31 = occupancy_penalty(GpuArch::SM_80,
                                                  30, 31, 256, 0);
    CHECK(penalty_30_to_31 == 0.0,
          "occupancy_penalty(30→31) == 0 (no cliff crossed)");

    // Going from 33 → 34 regs is also penalty 0 (both round to 36).
    double penalty_33_to_34 = occupancy_penalty(GpuArch::SM_80,
                                                  33, 34, 256, 0);
    CHECK(penalty_33_to_34 == 0.0,
          "occupancy_penalty(33→34) == 0 (no cliff crossed)");

    // Improvement (33 → 32 regs) — penalty is 0 (not negative).
    double penalty_33_to_32 = occupancy_penalty(GpuArch::SM_80,
                                                  33, 32, 256, 0);
    CHECK(penalty_33_to_32 == 0.0,
          "occupancy_penalty(33→32) == 0 (improvement, no penalty)");

    // 4-arg overload: compares against (regs - 1).
    // At 33 regs the previous step (32) gave ratio 1.0; current 0.75 → penalty 25.
    double penalty4 = occupancy_penalty(GpuArch::SM_80, 33, 256, 0);
    CHECK(penalty4 > 24.9 && penalty4 < 25.1,
          "4-arg occupancy_penalty(33) ~= 25.0");
}

// ─────────────────────────────────────────────────────────────────────────────
//  Test 6: UOpsCostModel — Mul on x86_64 is 3 uops (>= 3.0)
// ─────────────────────────────────────────────────────────────────────────────
static void test_uops_cost_model_mul_is_3_or_4_uops() {
    UOpsCostModel uops_x86(Arch::X86_64);

    // The legacy cost(op) returns OpCost; the uops field should be >= 3.
    OpCost mul_cost = uops_x86.cost(Opcode::Mul);
    CHECK(static_cast<double>(mul_cost.uops) >= 3.0,
          "UOpsCostModel(x86_64).cost(Mul).uops >= 3");

    // The new per-kind API: CodeSize returns the uops count.
    double mul_uops = uops_x86.cost(Opcode::Mul, CostKind::CodeSize);
    CHECK(mul_uops >= 3.0,
          "UOpsCostModel(x86_64).cost(Mul, CodeSize) >= 3.0");

    // ReciprocalThroughput for Mul should also reflect the uops count
    // (the uOps model uses uops as the throughput proxy for non-trivial
    // ops; for Mul we return the raw uops count).
    double mul_rt = uops_x86.cost(Opcode::Mul, CostKind::ReciprocalThroughput);
    CHECK(mul_rt >= 3.0,
          "UOpsCostModel(x86_64).cost(Mul, ReciprocalThroughput) >= 3.0");

    // Add should be 1 uop; Div should be much higher (>= 10).
    CHECK(uops_x86.cost(Opcode::Add, CostKind::CodeSize) == 1.0,
          "UOpsCostModel(x86_64).cost(Add, CodeSize) == 1.0");
    CHECK(uops_x86.cost(Opcode::UDiv, CostKind::CodeSize) >= 10.0,
          "UOpsCostModel(x86_64).cost(UDiv, CodeSize) >= 10.0");

    // Phi is 0 uops.
    CHECK(uops_x86.cost(Opcode::Phi, CostKind::CodeSize) == 0.0,
          "UOpsCostModel(x86_64).cost(Phi, CodeSize) == 0.0");

    // The uOps model also applies operand-aware discounts: mul x,8 → 1 uop.
    std::vector<OperandValueInfo> operands = { make_pow2_operand(8) };
    double mul8_uops = uops_x86.cost(Opcode::Mul, CostKind::CodeSize, operands);
    CHECK(mul8_uops == 1.0,
          "UOpsCostModel(x86_64).cost(Mul, CodeSize, {pow2=8}) == 1.0 (shift-fold)");

    // And mul x,1 → 0 uops.
    std::vector<OperandValueInfo> one_op = { make_const_operand(1) };
    double mul1_uops = uops_x86.cost(Opcode::Mul, CostKind::CodeSize, one_op);
    CHECK(mul1_uops == 0.0,
          "UOpsCostModel(x86_64).cost(Mul, CodeSize, {1}) == 0.0 (no-op)");
}

// ─────────────────────────────────────────────────────────────────────────────
//  Test 7: Factory + CheapIRCostModel wrapper
// ─────────────────────────────────────────────────────────────────────────────
static void test_factories_and_cheap_tier() {
    // make_uops_cost_model_for_name dispatch.
    auto u1 = make_uops_cost_model_for_name("x86_64");
    auto u2 = make_uops_cost_model_for_name("aarch64");
    auto u3 = make_uops_cost_model_for_name("sm_80");
    auto u4 = make_uops_cost_model_for_name("unknown_arch");
    CHECK(u1 && u2 && u3 && u4, "uops factory returns non-null");
    CHECK(u1->arch() == Arch::X86_64, "uops x86_64 name → X86_64");
    CHECK(u2->arch() == Arch::AArch64, "uops aarch64 name → AArch64");
    CHECK(u3->arch() == Arch::PTX, "uops sm_80 name → PTX");
    CHECK(u4->arch() == Arch::Generic, "uops unknown name → Generic");

    // CheapIRCostModel wraps a delegate and returns identical numbers.
    auto x86 = make_cost_model(Arch::X86_64);
    CheapIRCostModel cheap(x86);
    CHECK(cheap.arch() == x86->arch(), "CheapIRCostModel.arch() delegates");
    CHECK(cheap.arch_name() == std::string("x86_64"),
          "CheapIRCostModel.arch_name() delegates");
    OpCost cheap_mul = cheap.cost(Opcode::Mul);
    OpCost x86_mul = x86->cost(Opcode::Mul);
    CHECK(cheap_mul.uops == x86_mul.uops &&
          std::fabs(cheap_mul.latency_cycles - x86_mul.latency_cycles) < 1e-9,
          "CheapIRCostModel.cost(op) == delegate.cost(op)");

    // Operand-aware cost also delegates.
    std::vector<OperandValueInfo> operands = { make_pow2_operand(8) };
    double cheap_discount = cheap.cost(Opcode::Mul,
                                        CostKind::ReciprocalThroughput,
                                        operands);
    double x86_discount = x86->cost(Opcode::Mul,
                                     CostKind::ReciprocalThroughput,
                                     operands);
    CHECK(std::fabs(cheap_discount - x86_discount) < 1e-9,
          "CheapIRCostModel.cost(op, kind, operands) == delegate");
}

// ─────────────────────────────────────────────────────────────────────────────
//  Test 8: Backwards-compat — existing cost(op) signature still works
// ─────────────────────────────────────────────────────────────────────────────
static void test_backwards_compat_cost_op() {
    // Make sure the legacy OpCost cost(Opcode) signature is unchanged
    // for every arch — the new TTI additions must be purely additive.
    auto x86 = make_cost_model(Arch::X86_64);
    auto arm = make_cost_model(Arch::AArch64);
    auto ptx = make_cost_model(Arch::PTX);
    auto gen = make_cost_model(Arch::Generic);

    // Spot-check a few known values from the existing tables.
    OpCost x86_mul = x86->cost(Opcode::Mul);
    CHECK(x86_mul.uops == 3, "x86_64 Mul uops == 3 (unchanged)");
    CHECK(std::fabs(x86_mul.latency_cycles - 3.0) < 1e-9,
          "x86_64 Mul latency == 3.0 (unchanged)");

    OpCost ptx_load = ptx->cost(Opcode::Load);
    CHECK(std::fabs(ptx_load.latency_cycles - 100.0) < 1e-9,
          "ptx Load latency == 100.0 (unchanged)");

    OpCost gen_phi = gen->cost(Opcode::Phi);
    CHECK(gen_phi.uops == 0,
          "generic Phi uops == 0 (unchanged)");

    OpCost arm_add = arm->cost(Opcode::Add);
    CHECK(std::fabs(arm_add.throughput_cycles - 0.5) < 1e-9,
          "aarch64 Add throughput == 0.5 (unchanged)");

    // The convenience 2-arg cost(op, operands) overload calls the 3-arg
    // overload with ReciprocalThroughput.
    std::vector<OperandValueInfo> empty_operands;
    double via_2arg = x86->cost(Opcode::Mul, empty_operands);
    double via_3arg = x86->cost(Opcode::Mul, CostKind::ReciprocalThroughput,
                                  empty_operands);
    CHECK(std::fabs(via_2arg - via_3arg) < 1e-9,
          "2-arg cost(op, operands) dispatches to 3-arg with ReciprocalThroughput");
}

// ─────────────────────────────────────────────────────────────────────────────
//  Main
// ─────────────────────────────────────────────────────────────────────────────

int main() {
    std::cout << "=== I5 CostModel TTI tests ===" << std::endl;

    std::cout << "  R4.A: CostKind::ReciprocalThroughput default..." << std::endl;
    test_cost_kind_reciprocal_throughput_default();

    std::cout << "  R4.A: Operand-aware mul x,pow2 → shift..." << std::endl;
    test_operand_aware_cost_mul_power_of_two();

    std::cout << "  R4.A: Operand-aware udiv x,pow2 → shift..." << std::endl;
    test_operand_aware_cost_div_power_of_two();

    std::cout << "  R4.A: operand_info() derives from ir::Value..." << std::endl;
    test_operand_info_from_value();

    std::cout << "  R4.G: SM_80 32 regs → occupancy_ratio ~1.0..." << std::endl;
    test_occupancy_sm80_32_regs();

    std::cout << "  R4.G: SM_80 34 regs → occupancy_ratio ~0.75 (cliff)..." << std::endl;
    test_occupancy_sm80_34_regs();

    std::cout << "  R4.G: occupancy_penalty cliff behaviour..." << std::endl;
    test_occupancy_penalty_cliff();

    std::cout << "  R3.C: UOpsCostModel Mul >= 3 uops..." << std::endl;
    test_uops_cost_model_mul_is_3_or_4_uops();

    std::cout << "  R3.C: factories + CheapIRCostModel..." << std::endl;
    test_factories_and_cheap_tier();

    std::cout << "  Backwards-compat: legacy cost(op) unchanged..." << std::endl;
    test_backwards_compat_cost_op();

    std::cout << "\n=== Results: " << g_pass << " passed, "
              << g_fail << " failed ===" << std::endl;
    return g_fail > 0 ? 1 : 0;
}
