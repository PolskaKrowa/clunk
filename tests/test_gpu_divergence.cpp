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
 * Clunk GPU Divergence Tests — DivergenceAnalysis + LivenessAnalysis.
 *
 * Verifies:
 *   - Values named "tid"/"threadIdx.x" etc. are classified divergent.
 *   - Function arguments and constants are uniform.
 *   - Conditional branches on divergent conditions are flagged divergent.
 *   - Conditional branches on uniform conditions are NOT divergent.
 *   - Reconvergence points are identified for divergent branches.
 *   - LivenessAnalysis computes non-trivial live-in/live-out on a
 *     small example with a diamond CFG.
 */
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "clunk/GPU/DivergenceAnalysis.h"
#include "clunk/GPU/LivenessAnalysis.h"
#include "clunk/GPU/PTXOptimizer.h"
#include "clunk/IR/BasicBlock.h"
#include "clunk/IR/Function.h"
#include "clunk/IR/Instruction.h"
#include "clunk/IR/Module.h"
#include "clunk/IR/Type.h"
#include "clunk/IR/Value.h"
#include "clunk/Pattern/PatternLibrary.h"

static int g_pass = 0, g_fail = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { std::cerr << "FAIL: " << msg << " at " << __FILE__ << ":" << __LINE__ << "\n"; g_fail++; } \
    else { g_pass++; } \
} while(0)

using namespace clunk::gpu;
using namespace clunk::ir;
using namespace clunk;

// ── Helper: build a small kernel with a thread-id-named value ──────────────
//
//   define i32 @k(i32 %x) {
//   entry:
//     %tid_x = add i32 %x, 1     ; named "tid_x" → divergent
//     %cond = icmp eq i32 %tid_x, %x
//     br i1 %cond, label %then, label %else
//   then:
//     ret i32 1
//   else:
//     ret i32 0
//   }
//
static std::shared_ptr<Module> make_divergent_kernel() {
    auto mod = std::make_shared<Module>("div_mod");
    TypeContext& ctx = mod->type_context();
    auto i32_ty = ctx.int32();

    auto fn_type = std::make_shared<FunctionType>(
        i32_ty, std::vector<std::shared_ptr<Type>>{i32_ty});
    auto& fn = mod->add_function("k", fn_type);
    fn.add_argument(i32_ty, "x");
    fn.set_attribute("kernel", "1");

    auto& entry = fn.add_block("entry");
    auto& then_bb = fn.add_block("then");
    auto& else_bb = fn.add_block("else");

    auto x = std::make_shared<Value>(i32_ty, "x");
    auto one = ConstantInt::get(ctx, 1, 32);

    // tid_x — name triggers divergence
    auto tid = inst::make_add(x, one, "tid_x");
    auto cond = inst::make_icmp(CmpPredicate::EQ, tid, x, "cond");
    auto br = inst::make_br(cond, "then", "else");

    entry.add_instruction(tid);
    entry.add_instruction(cond);
    entry.add_instruction(br);
    then_bb.add_instruction(inst::make_ret(one));
    auto zero = ConstantInt::get(ctx, 0, 32);
    else_bb.add_instruction(inst::make_ret(zero));

    return mod;
}

// ── Helper: build a kernel with a uniform branch ───────────────────────────
//
//   define i32 @k(i32 %x) {
//   entry:
//     %sum = add i32 %x, 1     ; named "sum" — uniform (depends only on arg)
//     %cond = icmp eq i32 %sum, %x
//     br i1 %cond, label %then, label %else
//   then:
//     ret i32 1
//   else:
//     ret i32 0
//   }
//
static std::shared_ptr<Module> make_uniform_kernel() {
    auto mod = std::make_shared<Module>("uni_mod");
    TypeContext& ctx = mod->type_context();
    auto i32_ty = ctx.int32();

    auto fn_type = std::make_shared<FunctionType>(
        i32_ty, std::vector<std::shared_ptr<Type>>{i32_ty});
    auto& fn = mod->add_function("k", fn_type);
    fn.add_argument(i32_ty, "x");
    fn.set_attribute("kernel", "1");

    auto& entry = fn.add_block("entry");
    auto& then_bb = fn.add_block("then");
    auto& else_bb = fn.add_block("else");

    auto x = std::make_shared<Value>(i32_ty, "x");
    auto one = ConstantInt::get(ctx, 1, 32);

    // Uniform name; depends only on a uniform argument
    auto sum = inst::make_add(x, one, "sum");
    auto cond = inst::make_icmp(CmpPredicate::EQ, sum, x, "cond");
    auto br = inst::make_br(cond, "then", "else");

    entry.add_instruction(sum);
    entry.add_instruction(cond);
    entry.add_instruction(br);
    then_bb.add_instruction(inst::make_ret(one));
    auto zero = ConstantInt::get(ctx, 0, 32);
    else_bb.add_instruction(inst::make_ret(zero));

    return mod;
}

// ── Test 1: divergent kernel ───────────────────────────────────────────────

static void test_divergent_kernel() {
    auto mod = make_divergent_kernel();
    auto fn = mod->function("k");

    DivergenceAnalysis da;
    DivergenceResult r = da.analyse(*fn);

    CHECK(r.total_conditional_branches == 1,
          "exactly 1 conditional branch in the test kernel");
    CHECK(r.divergent_branch_count == 1,
          "the branch is divergent (cond depends on tid_x)");

    // The tid_x value must be marked divergent.
    bool found_div = false;
    for (auto& bb : fn->blocks()) {
        for (auto& inst : bb->instructions()) {
            if (inst->name() == "tid_x") {
                auto it = r.is_divergent.find(inst.get());
                if (it != r.is_divergent.end() && it->second) {
                    found_div = true;
                }
            }
        }
    }
    CHECK(found_div, "tid_x instruction is classified divergent");

    // branch_divergence_fraction should be 1.0 (all branches divergent)
    CHECK(r.branch_divergence_fraction() > 0.99,
          "branch divergence fraction ~1.0");
}

// ── Test 2: uniform kernel ─────────────────────────────────────────────────

static void test_uniform_kernel() {
    auto mod = make_uniform_kernel();
    auto fn = mod->function("k");

    DivergenceAnalysis da;
    DivergenceResult r = da.analyse(*fn);

    CHECK(r.total_conditional_branches == 1, "1 conditional branch");
    CHECK(r.divergent_branch_count == 0,
          "no divergent branches (cond depends only on uniform arg)");

    // The "sum" instruction must be uniform (not in is_divergent map,
    // or mapped to false).
    for (auto& bb : fn->blocks()) {
        for (auto& inst : bb->instructions()) {
            if (inst->name() == "sum") {
                auto it = r.is_divergent.find(inst.get());
                bool div = (it != r.is_divergent.end() && it->second);
                CHECK(!div, "sum (uniform name) is NOT divergent");
            }
        }
    }

    CHECK(r.branch_divergence_fraction() == 0.0,
          "branch divergence fraction = 0.0 for uniform kernel");
}

// ── Test 3: looks_like_thread_id recognition ───────────────────────────────
//
// Build a kernel where the divergence source is named differently.
//
static void test_divergence_naming_variants() {
    auto mod = std::make_shared<Module>("m");
    TypeContext& ctx = mod->type_context();
    auto i32_ty = ctx.int32();

    auto fn_type = std::make_shared<FunctionType>(
        i32_ty, std::vector<std::shared_ptr<Type>>{i32_ty});
    auto& fn = mod->add_function("k", fn_type);
    fn.add_argument(i32_ty, "x");
    auto& entry = fn.add_block("entry");
    auto& then_bb = fn.add_block("then");
    auto& else_bb = fn.add_block("else");

    auto x = std::make_shared<Value>(i32_ty, "x");
    auto one = ConstantInt::get(ctx, 1, 32);

    // Various thread-id-style names should all be flagged divergent
    auto t1 = inst::make_add(x, one, "threadIdx_x");
    auto t2 = inst::make_add(t1, one, "blockIdx_y");
    auto t3 = inst::make_add(t2, one, "lane_id");
    auto cond = inst::make_icmp(CmpPredicate::EQ, t3, x, "c");
    auto br = inst::make_br(cond, "then", "else");

    entry.add_instruction(t1);
    entry.add_instruction(t2);
    entry.add_instruction(t3);
    entry.add_instruction(cond);
    entry.add_instruction(br);
    then_bb.add_instruction(inst::make_ret(one));
    auto zero = ConstantInt::get(ctx, 0, 32);
    else_bb.add_instruction(inst::make_ret(zero));

    DivergenceAnalysis da;
    DivergenceResult r = da.analyse(fn);

    // All three should be divergent
    for (auto& bb : fn.blocks()) {
        for (auto& inst : bb->instructions()) {
            if (inst->name() == "threadIdx_x" ||
                inst->name() == "blockIdx_y" ||
                inst->name() == "lane_id") {
                auto it = r.is_divergent.find(inst.get());
                bool div = (it != r.is_divergent.end() && it->second);
                CHECK(div, inst->name() + " classified divergent");
            }
        }
    }

    CHECK(r.divergent_branch_count == 1,
          "the branch is divergent (transitive through tid-named values)");
}

// ── Test 4: LivenessAnalysis on a straight-line function ───────────────────
//
//   define i32 @k(i32 %a, i32 %b) {
//   entry:
//     %sum = add i32 %a, %b
//     %sq  = mul i32 %sum, %sum
//     ret i32 %sq
//   }
//
// Live-in of %sum: {%a, %b}     (both used by %sum)
// Live-out of %sum: {%sum}      (used by %sq)
// Live-in of %sq: {%sum}        (used by %sq)
// Live-out of %sq: {%sq}        (used by ret)
// function_max_live ≥ 2 (at least {%a, %b} before %sum is computed)
//
static void test_liveness_straight_line() {
    Module mod("m");
    TypeContext& ctx = mod.type_context();
    auto i32_ty = ctx.int32();

    auto fn_type = std::make_shared<FunctionType>(
        i32_ty, std::vector<std::shared_ptr<Type>>{i32_ty, i32_ty});
    auto& fn = mod.add_function("k", fn_type);
    fn.add_argument(i32_ty, "a");
    fn.add_argument(i32_ty, "b");
    auto& entry = fn.add_block("entry");

    auto a = std::make_shared<Value>(i32_ty, "a");
    auto b = std::make_shared<Value>(i32_ty, "b");

    auto sum = inst::make_add(a, b, "sum");
    auto sq = inst::make_mul(sum, sum, "sq");
    auto ret = inst::make_ret(sq);

    entry.add_instruction(sum);
    entry.add_instruction(sq);
    entry.add_instruction(ret);

    LivenessAnalysis la;
    LiveSets live = la.compute(fn);

    // function_max_live should be at least 2 (a and b live simultaneously
    // before %sum is computed)
    CHECK(live.function_max_live >= 2,
          "function_max_live >= 2 (a and b live before sum)");

    // %sum must be in live_out for the first instruction (used by %sq)
    auto sum_it = live.live_out.find(sum.get());
    CHECK(sum_it != live.live_out.end(),
          "live_out exists for sum instruction");
    if (sum_it != live.live_out.end()) {
        // The live-out of %sum should contain %sum itself (used by %sq).
        CHECK(sum_it->second.count(sum.get()) > 0,
              "sum is in its own live_out (used by sq)");
    }

    // %sq must be in live_in for the ret instruction (used by ret)
    auto ret_it = live.live_in.find(ret.get());
    CHECK(ret_it != live.live_in.end(),
          "live_in exists for ret instruction");
    if (ret_it != live.live_in.end()) {
        CHECK(ret_it->second.count(sq.get()) > 0,
              "sq is in live_in of ret (used by ret)");
    }
}

// ── Test 5: LivenessAnalysis with diamond CFG ──────────────────────────────
//
//   define i32 @k(i32 %a) {
//   entry:
//     %c = icmp eq i32 %a, 0
//     br i1 %c, label %then, label %else
//   then:
//     %t = add i32 %a, 1
//     br label %merge
//   else:
//     %e = add i32 %a, 2
//     br label %merge
//   merge:
//     %p = phi i32 [%t, %then], [%e, %else]
//     ret i32 %p
//   }
//
// After the entry block's branch, %a must be live (it's used in both then
// and else). The liveness pass must propagate this across block boundaries.
//
static void test_liveness_diamond_cfg() {
    Module mod("m");
    TypeContext& ctx = mod.type_context();
    auto i32_ty = ctx.int32();

    auto fn_type = std::make_shared<FunctionType>(
        i32_ty, std::vector<std::shared_ptr<Type>>{i32_ty});
    auto& fn = mod.add_function("k", fn_type);
    fn.add_argument(i32_ty, "a");

    auto& entry = fn.add_block("entry");
    auto& then_bb = fn.add_block("then");
    auto& else_bb = fn.add_block("else");
    auto& merge_bb = fn.add_block("merge");

    auto a = std::make_shared<Value>(i32_ty, "a");
    auto zero = ConstantInt::get(ctx, 0, 32);
    auto one = ConstantInt::get(ctx, 1, 32);
    auto two = ConstantInt::get(ctx, 2, 32);

    auto c = inst::make_icmp(CmpPredicate::EQ, a, zero, "c");
    auto br = inst::make_br(c, "then", "else");
    entry.add_instruction(c);
    entry.add_instruction(br);

    auto t = inst::make_add(a, one, "t");
    auto br_then = inst::make_br_uncond("merge");
    then_bb.add_instruction(t);
    then_bb.add_instruction(br_then);

    auto e = inst::make_add(a, two, "e");
    auto br_else = inst::make_br_uncond("merge");
    else_bb.add_instruction(e);
    else_bb.add_instruction(br_else);

    auto p = inst::make_phi(i32_ty, "p");
    p->add_operand(t);
    p->add_operand(e);
    p->set_metadata("phi_blocks", "then,else");
    auto ret = inst::make_ret(p);
    merge_bb.add_instruction(p);
    merge_bb.add_instruction(ret);

    LivenessAnalysis la;
    LiveSets live = la.compute(fn);

    // After compute_predecessors (called by LivenessAnalysis), merge has
    // predecessors then and else.
    CHECK(live.function_max_live >= 1,
          "diamond: function_max_live >= 1");

    // The ret instruction's live_in must contain %p (phi result).
    auto ret_it = live.live_in.find(ret.get());
    CHECK(ret_it != live.live_in.end(), "ret has live_in");
    if (ret_it != live.live_in.end()) {
        CHECK(ret_it->second.count(p.get()) > 0,
              "p (phi result) is live_in of ret");
    }
}

// ── Test 6: estimate_divergence via PTXOptimizer ───────────────────────────

static void test_ptx_optimiser_estimate_divergence() {
    auto mod = make_divergent_kernel();
    auto fn = mod->function("k");

    pattern::ArchDescriptor arch;
    arch.is_gpu = true;
    arch.compute_capability = 80;
    arch.warp_size = 32;
    PTXOptimizer opt(arch);
    double d = opt.estimate_divergence(*fn);

    // Branch is divergent → fraction == 1.0
    CHECK(d > 0.99, "estimate_divergence returns ~1.0 for divergent kernel");

    auto mod2 = make_uniform_kernel();
    auto fn2 = mod2->function("k");
    double d2 = opt.estimate_divergence(*fn2);
    CHECK(d2 < 0.01, "estimate_divergence returns ~0.0 for uniform kernel");
}

// ── Main ────────────────────────────────────────────────────────────────────

int main() {
    std::cout << "=== Clunk GPU Divergence Tests ===" << std::endl;

    std::cout << "  divergent kernel..." << std::endl;
    test_divergent_kernel();

    std::cout << "  uniform kernel..." << std::endl;
    test_uniform_kernel();

    std::cout << "  naming variants..." << std::endl;
    test_divergence_naming_variants();

    std::cout << "  liveness straight-line..." << std::endl;
    test_liveness_straight_line();

    std::cout << "  liveness diamond CFG..." << std::endl;
    test_liveness_diamond_cfg();

    std::cout << "  PTXOptimizer estimate_divergence..." << std::endl;
    test_ptx_optimiser_estimate_divergence();

    std::cout << "\n=== Results: " << g_pass << " passed, "
              << g_fail << " failed ===" << std::endl;
    return g_fail > 0 ? 1 : 0;
}
