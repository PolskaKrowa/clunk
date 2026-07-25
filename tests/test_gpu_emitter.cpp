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
 * Clunk GPU Emitter Tests — PTXEmitter + scheduling + CUDADriver.
 *
 * Verifies:
 *   1. PTXEmitter produces valid-looking PTX for examples/simple_add.ll,
 *      with the correct .target directive, .address_size, .version header,
 *      integer arithmetic (add.u32), and a ret instruction.
 *   2. PTXEmitter honours set_arch(): emitting with SM_90 yields a
 *      ".target sm_90" directive.
 *   3. PTXOptimizer::optimise() preserves GEP→Load data dependencies
 *      .
 *   4. PTXOptimizer::optimise() does NOT mutate the input function
 *      .
 *   5. PTXOptimizer populates the gpu.ptx_text_present attribute.
 *   6. CUDADriver::has_gpu() returns false in CPU-only environments.
 */
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "clunk/GPU/ArchLimits.h"
#include "clunk/GPU/CUDADriver.h"
#include "clunk/GPU/PTXEmitter.h"
#include "clunk/GPU/PTXOptimizer.h"
#include "clunk/IR/BasicBlock.h"
#include "clunk/IR/Function.h"
#include "clunk/IR/Instruction.h"
#include "clunk/IR/Module.h"
#include "clunk/IR/Type.h"
#include "clunk/IR/Value.h"
#include "clunk/Parser/IRParser.h"
#include "clunk/Pattern/PatternLibrary.h"

static int g_pass = 0, g_fail = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { std::cerr << "FAIL: " << msg << " at " << __FILE__ << ":" << __LINE__ << "\n"; g_fail++; } \
    else { g_pass++; } \
} while(0)

using namespace clunk::gpu;
using namespace clunk::ir;
using namespace clunk;

// ── Test 1: PTXEmitter on simple_add.ll ────────────────────────────────────
//
//   define i32 @simple_add(i32 %a, i32 %b) {
//   entry:
//     %result = add i32 %a, %b
//     ret i32 %result
//   }
//
static void test_emit_simple_add() {
    const std::string src =
        "define i32 @simple_add(i32 %a, i32 %b) {\n"
        "entry:\n"
        "  %result = add i32 %a, %b\n"
        "  ret i32 %result\n"
        "}\n";

    parser::IRParser p;
    auto mod = p.parse_string(src);
    CHECK(mod != nullptr, "simple_add parses");
    CHECK(mod->function_count() >= 1, "module has at least one function");

    auto fn = mod->function("simple_add");
    CHECK(fn != nullptr, "simple_add function found");

    PTXEmitter emitter(GpuArch::SM_80);
    std::string ptx = emitter.emit(*fn);

    CHECK(!ptx.empty(), "emitter produced non-empty PTX");
    CHECK(ptx.find(".version") != std::string::npos,
          "PTX contains .version directive");
    CHECK(ptx.find(".target sm_80") != std::string::npos,
          "PTX contains .target sm_80");
    CHECK(ptx.find(".address_size 64") != std::string::npos,
          "PTX contains .address_size 64");
    CHECK(ptx.find(".visible .entry simple_add") != std::string::npos,
          "PTX contains kernel signature");
    CHECK(ptx.find("add.u32") != std::string::npos,
          "PTX contains add.u32 instruction");
    CHECK(ptx.find("ret;") != std::string::npos,
          "PTX contains ret instruction");
    CHECK(ptx.find("ld.param.u32") != std::string::npos,
          "PTX loads parameters into registers");
    CHECK(emitter.num_registers() > 0,
          "emitter allocated at least one register");
}

// ── Test 2: PTXEmitter honours set_arch() ──────────────────────────────────

static void test_emit_arch_targeting() {
    const std::string src =
        "define i32 @f(i32 %a) {\n"
        "entry:\n"
        "  %r = add i32 %a, %a\n"
        "  ret i32 %r\n"
        "}\n";
    parser::IRParser p;
    auto mod = p.parse_string(src);
    auto fn = mod->function("f");

    for (GpuArch arch : {GpuArch::SM_70, GpuArch::SM_75, GpuArch::SM_80,
                          GpuArch::SM_86, GpuArch::SM_89, GpuArch::SM_90}) {
        PTXEmitter emitter(arch);
        std::string ptx = emitter.emit(*fn);
        std::string arch_name = gpu_arch_name(arch);
        CHECK(ptx.find(".target " + arch_name) != std::string::npos,
              "PTX for " + arch_name + " contains correct .target");
    }
}

// ── Test 3: schedule_instructions preserves GEP→Load dependency ────────────
//
//   define i32 @test(i32* %ptr, i32 %idx) {
//   entry:
//     %gep = getelementptr i32, i32* %ptr, i32 %idx
//     %unused = add i32 %idx, 1
//     %val = load i32, i32* %gep
//     ret i32 %val
//   }
//
// The buggy implementation moved all Loads to the front, producing:
//     %val = load i32, i32* %gep    ← %gep not yet defined!
//     %gep = getelementptr ...
//     %unused = add i32 %idx, 1
//     ret i32 %val
//
// After the fix, %gep must precede %val in the result.
//
static void test_schedule_preserves_gep_load_dep() {
    Module mod("test_mod");
    TypeContext& ctx = mod.type_context();

    auto i32_ty = ctx.int32();
    auto ptr_i32 = ctx.pointer_to(i32_ty);
    auto fn_type = std::make_shared<FunctionType>(
        i32_ty, std::vector<std::shared_ptr<Type>>{ptr_i32, i32_ty});

    auto& fn = mod.add_function("test_gep_load", fn_type);
    fn.add_argument(ptr_i32, "ptr");
    fn.add_argument(i32_ty, "idx");
    fn.set_attribute("kernel", "1");
    auto& entry = fn.add_block("entry");

    // Build operands: plain Value placeholders for arguments.
    auto ptr_arg = std::make_shared<Value>(ptr_i32, "ptr");
    auto idx_arg = std::make_shared<Value>(i32_ty, "idx");

    // GEP: %gep = getelementptr i32, i32* %ptr, i32 %idx
    auto gep = inst::make_gep(ptr_i32, ptr_arg, {idx_arg}, "gep");
    // %unused = add i32 %idx, 1
    auto one = ConstantInt::get(ctx, 1, 32);
    auto unused = inst::make_add(idx_arg, one, "unused");
    // %val = load i32, i32* %gep
    auto load = inst::make_load(i32_ty, gep, "val");
    // ret i32 %val
    auto ret = inst::make_ret(load);

    entry.add_instruction(gep);
    entry.add_instruction(unused);
    entry.add_instruction(load);
    entry.add_instruction(ret);

    // Snapshot original IR (for the no-mutation check below)
    auto orig_gep_pos = entry.instructions()[0];
    auto orig_load_pos = entry.instructions()[2];

    pattern::ArchDescriptor arch;
    arch.is_gpu = true;
    arch.compute_capability = 80;
    arch.warp_size = 32;
    PTXOptimizer opt(arch);
    auto result = opt.optimise(fn);

    CHECK(result != nullptr, "optimise returned a result");

    // Find the result's entry block.
    auto result_entry = result->block("entry");
    CHECK(result_entry != nullptr, "result has an entry block");

    // Find positions of gep and load in the result.
    size_t gep_pos = SIZE_MAX, load_pos = SIZE_MAX;
    const auto& instrs = result_entry->instructions();
    for (size_t i = 0; i < instrs.size(); ++i) {
        if (instrs[i]->opcode() == Opcode::GetElementPtr &&
            instrs[i]->name() == "gep") {
            gep_pos = i;
        }
        if (instrs[i]->opcode() == Opcode::Load &&
            instrs[i]->name() == "val") {
            load_pos = i;
        }
    }
    CHECK(gep_pos != SIZE_MAX, "result contains the gep instruction");
    CHECK(load_pos != SIZE_MAX, "result contains the load instruction");
    CHECK(gep_pos < load_pos,
          "gep precedes load in the scheduled result (SSA preserved)");

    // the input function must not have been mutated.
    // The original instructions in the input should not carry the
    // "coalescing_hint" or "spill_candidate" metadata that the optimiser
    // adds to its result.
    bool input_mutated = false;
    for (auto& bb : fn.blocks()) {
        for (auto& inst : bb->instructions()) {
            if (inst->metadata().count("coalescing_hint") ||
                inst->metadata().count("spill_candidate")) {
                input_mutated = true;
                break;
            }
        }
        if (input_mutated) break;
    }
    CHECK(!input_mutated, "optimise() did not mutate the input function");

    (void)orig_gep_pos;
    (void)orig_load_pos;
}

// ── Test 4: PTXEmitter covers more opcodes ─────────────────────────────────
//
// Build a small kernel with sub, mul, shl, and, icmp + cond branch.
// Verify PTX contains the expected mnemonics.
//
static void test_emit_more_opcodes() {
    Module mod("m");
    TypeContext& ctx = mod.type_context();
    auto i32_ty = ctx.int32();

    auto fn_type = std::make_shared<FunctionType>(
        i32_ty, std::vector<std::shared_ptr<Type>>{i32_ty, i32_ty});
    auto& fn = mod.add_function("k", fn_type);
    fn.add_argument(i32_ty, "a");
    fn.add_argument(i32_ty, "b");
    auto& entry = fn.add_block("entry");
    auto& then_bb = fn.add_block("then");
    auto& else_bb = fn.add_block("else");

    auto a = std::make_shared<Value>(i32_ty, "a");
    auto b = std::make_shared<Value>(i32_ty, "b");
    auto one = ConstantInt::get(ctx, 1, 32);

    // Build instructions directly: inst:: only has add/sub/mul/icmp/allocation
    // helpers, so for shl/and we construct Instruction objects manually.
    auto sub = inst::make_sub(a, b, "s");
    auto mul = inst::make_mul(sub, b, "m");

    auto shl = std::make_shared<Instruction>(Opcode::Shl, i32_ty, "sh");
    shl->add_operand(mul);
    shl->add_operand(one);

    auto and_ = std::make_shared<Instruction>(Opcode::And, i32_ty, "an");
    and_->add_operand(shl);
    and_->add_operand(one);

    auto cmp = inst::make_icmp(CmpPredicate::EQ, and_, one, "c");
    auto br = inst::make_br(cmp, "then", "else");

    entry.add_instruction(sub);
    entry.add_instruction(mul);
    entry.add_instruction(shl);
    entry.add_instruction(and_);
    entry.add_instruction(cmp);
    entry.add_instruction(br);

    // then: ret 1
    auto& then_block = then_bb;
    then_block.add_instruction(inst::make_ret(one));
    // else: ret 0
    auto zero = ConstantInt::get(ctx, 0, 32);
    else_bb.add_instruction(inst::make_ret(zero));

    PTXEmitter emitter(GpuArch::SM_80);
    std::string ptx = emitter.emit(fn);

    CHECK(ptx.find("sub.u32") != std::string::npos, "sub.u32 emitted");
    CHECK(ptx.find("mul.lo.u32") != std::string::npos, "mul.lo.u32 emitted");
    CHECK(ptx.find("shl.b32") != std::string::npos, "shl.b32 emitted");
    CHECK(ptx.find("and.b32") != std::string::npos, "and.b32 emitted");
    CHECK(ptx.find("setp.eq.u32") != std::string::npos,
          "setp.eq.u32 emitted for icmp eq");
    CHECK(ptx.find("bra $L_then") != std::string::npos,
          "conditional bra to true label emitted");
    CHECK(ptx.find("bra $L_else") != std::string::npos,
          "fallthrough bra to false label emitted");
}

// ── Test 5: PTXOptimizer populates ptx_text + attributes ───────────────────

static void test_optimizer_emits_ptx() {
    Module mod("m");
    TypeContext& ctx = mod.type_context();
    auto i32_ty = ctx.int32();
    auto fn_type = std::make_shared<FunctionType>(
        i32_ty, std::vector<std::shared_ptr<Type>>{i32_ty, i32_ty});
    auto& fn = mod.add_function("k", fn_type);
    fn.add_argument(i32_ty, "a");
    fn.add_argument(i32_ty, "b");
    fn.set_attribute("kernel", "1");
    auto& entry = fn.add_block("entry");
    auto a = std::make_shared<Value>(i32_ty, "a");
    auto b = std::make_shared<Value>(i32_ty, "b");
    auto add = inst::make_add(a, b, "sum");
    entry.add_instruction(add);
    entry.add_instruction(inst::make_ret(add));

    pattern::ArchDescriptor arch;
    arch.is_gpu = true;
    arch.compute_capability = 80;
    arch.warp_size = 32;
    PTXOptimizer opt(arch);
    auto result = opt.optimise(fn);

    CHECK(result != nullptr, "optimise returned a result");
    CHECK(!opt.ptx_text().empty(), "ptx_text() is non-empty after optimise");
    CHECK(result->attributes().count("gpu.ptx_text_present"),
          "result carries gpu.ptx_text_present attribute");
    CHECK(result->attributes().count("gpu.arch"),
          "result carries gpu.arch attribute");
    CHECK(result->attributes().count("gpu.register_pressure"),
          "result carries gpu.register_pressure attribute");
}

// ── Test 6: CUDADriver graceful fallback ────────────────────────────────────
//
// In the CPU-only CI environment, libcuda is not installed. The driver
// must report has_gpu()==false without crashing.
//
static void test_cuda_driver_no_gpu() {
    CUDADriver& drv = CUDADriver::instance();
    bool has = drv.has_gpu();
    CHECK(has == false,
          "CUDADriver::has_gpu() returns false in CPU-only environment");

    // compile_ptx should return an invalid handle (no crash).
    ModuleHandle h = drv.compile_ptx("invalid ptx");
    CHECK(!h.valid(), "compile_ptx returns invalid handle when no GPU");

    // launch_kernel should return false.
    bool launched = drv.launch_kernel(h, "k", 1,1,1, 1,1,1, 0, {});
    CHECK(!launched, "launch_kernel returns false when no GPU");

    // synchronize should return false.
    bool synced = drv.synchronize();
    CHECK(!synced, "synchronize returns false when no GPU");

    // last_error should be non-empty (diagnostic message set).
    CHECK(!drv.last_error().empty(),
          "last_error is populated when has_gpu() is false");
}

// ── Main ────────────────────────────────────────────────────────────────────

int main() {
    std::cout << "=== Clunk GPU Emitter Tests ===" << std::endl;

    std::cout << "  emit simple_add..." << std::endl;
    test_emit_simple_add();

    std::cout << "  arch targeting..." << std::endl;
    test_emit_arch_targeting();

    std::cout << "  schedule preserves GEP->Load..." << std::endl;
    test_schedule_preserves_gep_load_dep();

    std::cout << "  emit more opcodes..." << std::endl;
    test_emit_more_opcodes();

    std::cout << "  optimizer emits PTX..." << std::endl;
    test_optimizer_emits_ptx();

    std::cout << "  CUDA driver no-GPU..." << std::endl;
    test_cuda_driver_no_gpu();

    std::cout << "\n=== Results: " << g_pass << " passed, "
              << g_fail << " failed ===" << std::endl;
    return g_fail > 0 ? 1 : 0;
}
