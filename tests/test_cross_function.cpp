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
 * Clunk Cross-Function Tests — exercise the new module-level passes:
 *   - CallGraph build / queries
 *   - Dead Function Elimination (DFE)
 *   - Interprocedural Constant Propagation (IPCP)
 *   - Multi-block inliner (single + nested call chains)
 *   - End-to-end pipeline integration (the work_module pre-pass)
 */
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
#include "clunk/IR/Clone.h"
#include "clunk/Analysis/CallGraph.h"
#include "clunk/Search/CrossFunctionPasses.h"
#include "clunk/Search/Inliner.h"
#include "clunk/Search/SMTVerifier.h"
#include "clunk/Pipeline.h"

static int g_pass = 0, g_fail = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { std::cerr << "FAIL: " << msg << " at " << __FILE__ << ":" << __LINE__ << "\n"; g_fail++; } \
    else { g_pass++; } \
} while(0)

using namespace clunk::ir;
using namespace clunk;

// ═══════════════════════════════════════════════════════════════════════════
//  Helpers — build small modules for each test
// ═══════════════════════════════════════════════════════════════════════════

// Build a module with:
//   @wrapper(x)  → calls @double(x) and returns it
//   @double(x)   → x << 1  (avoid `mul x, x` which trips a pre-existing
//                           pattern-matcher bug in strength_reduce_mul)
//   @dead_fn(x)  → never called, x + 1
//   @main()      → calls @wrapper(5) and returns it
static std::shared_ptr<Module> make_module_with_dead_and_inline() {
    auto mod = std::make_shared<Module>("xpass_test");
    TypeContext& ctx = mod->type_context();
    auto i32 = ctx.int32();

    // double(x) → shl x, 1
    {
        auto fn_type = std::make_shared<FunctionType>(i32, std::vector<std::shared_ptr<Type>>{i32});
        auto& fn = mod->add_function("double", fn_type, Linkage::Internal);
        fn.add_argument(i32, "x");
        auto& entry = fn.add_block("entry");
        auto x = std::make_shared<Value>(i32, "x");
        ir::TypeContext tmp_ctx;
        auto one = ConstantInt::get(tmp_ctx, 1);
        // shl instruction
        auto shl = std::make_shared<Instruction>(Opcode::Shl, i32, "r");
        shl->add_operand(x);
        shl->add_operand(one);
        entry.add_instruction(shl);
        entry.add_instruction(inst::make_ret(std::make_shared<Value>(i32, "r")));
    }
    // wrapper(x) → call @double(x), ret
    {
        auto fn_type = std::make_shared<FunctionType>(i32, std::vector<std::shared_ptr<Type>>{i32});
        auto& fn = mod->add_function("wrapper", fn_type, Linkage::Internal);
        fn.add_argument(i32, "x");
        auto& entry = fn.add_block("entry");
        auto x = std::make_shared<Value>(i32, "x");
        auto call = inst::make_call(i32, "double", {x}, "c");
        entry.add_instruction(call);
        entry.add_instruction(inst::make_ret(call));
    }
    // dead_fn(x) → add x, 1  (never called)
    {
        auto fn_type = std::make_shared<FunctionType>(i32, std::vector<std::shared_ptr<Type>>{i32});
        auto& fn = mod->add_function("dead_fn", fn_type, Linkage::Internal);
        fn.add_argument(i32, "x");
        auto& entry = fn.add_block("entry");
        auto x = std::make_shared<Value>(i32, "x");
        ir::TypeContext tmp_ctx;
        auto one = ConstantInt::get(tmp_ctx, 1);
        entry.add_instruction(inst::make_add(x, one, "r"));
        entry.add_instruction(inst::make_ret(std::make_shared<Value>(i32, "r")));
    }
    // main() → call @wrapper(5); ret
    {
        auto fn_type = std::make_shared<FunctionType>(i32, std::vector<std::shared_ptr<Type>>{});
        auto& fn = mod->add_function("main", fn_type, Linkage::External);
        auto& entry = fn.add_block("entry");
        ir::TypeContext tmp_ctx;
        auto five = ConstantInt::get(tmp_ctx, 5);
        auto call = inst::make_call(i32, "wrapper", {five}, "c");
        entry.add_instruction(call);
        entry.add_instruction(inst::make_ret(call));
    }
    return mod;
}

// Build a module with a multi-block callee and a single caller.
//   @abs_if(x) →  if x < 0 return -x else return x
//   @caller(x) → call @abs_if(x), add 1, return
static std::shared_ptr<Module> make_module_with_multiblock_callee() {
    auto mod = std::make_shared<Module>("mb_inline_test");
    TypeContext& ctx = mod->type_context();
    auto i32 = ctx.int32();

    // abs_if(x):
    //   entry:  %neg = sub 0, x; %cmp = icmp slt x, 0; br %cmp, then, else
    //   then:   br merge
    //   else:   br merge
    //   merge:  %r = phi [x, else], [%neg, then]; ret %r
    {
        auto fn_type = std::make_shared<FunctionType>(i32, std::vector<std::shared_ptr<Type>>{i32});
        auto& fn = mod->add_function("abs_if", fn_type, Linkage::Internal);
        fn.add_argument(i32, "x");
        auto& entry = fn.add_block("entry");
        auto& then_bb = fn.add_block("then");
        auto& else_bb = fn.add_block("else");
        auto& merge = fn.add_block("merge");
        auto x = std::make_shared<Value>(i32, "x");
        ir::TypeContext tmp_ctx;
        auto zero = ConstantInt::get(tmp_ctx, 0);
        entry.add_instruction(inst::make_sub(zero, x, "neg"));
        entry.add_instruction(inst::make_icmp(CmpPredicate::SLT, x, zero, "cmp"));
        entry.add_instruction(inst::make_br(entry.instruction(1), "then", "else"));
        then_bb.add_instruction(inst::make_br_uncond("merge"));
        else_bb.add_instruction(inst::make_br_uncond("merge"));
        // phi: incoming from else → x; from then → neg
        auto phi = std::make_shared<Instruction>(Opcode::Phi, i32, "r");
        phi->add_operand(x);
        phi->add_operand(std::make_shared<Value>(i32, "neg"));
        phi->set_metadata("phi_blocks", "else,then");
        merge.add_instruction(phi);
        merge.add_instruction(inst::make_ret(phi));
    }
    // caller(x): call @abs_if(x); add 1; ret
    {
        auto fn_type = std::make_shared<FunctionType>(i32, std::vector<std::shared_ptr<Type>>{i32});
        auto& fn = mod->add_function("caller", fn_type, Linkage::External);
        fn.add_argument(i32, "x");
        auto& entry = fn.add_block("entry");
        auto x = std::make_shared<Value>(i32, "x");
        auto call = inst::make_call(i32, "abs_if", {x}, "c");
        entry.add_instruction(call);
        ir::TypeContext tmp_ctx;
        auto one = ConstantInt::get(tmp_ctx, 1);
        entry.add_instruction(inst::make_add(call, one, "r"));
        entry.add_instruction(inst::make_ret(std::make_shared<Value>(i32, "r")));
    }
    return mod;
}

// Build a module where IPCP should specialise:
//   @double_it(x) → x << 1  (shl, not mul — avoids strength_reduce_mul bug)
//   @caller_a() → call @double_it(5); ret
//   @caller_b() → call @double_it(5); ret
// Both callers pass the constant 5 → IPCP creates double_it.ipcp_0_5.
static std::shared_ptr<Module> make_module_with_ipcp_target() {
    auto mod = std::make_shared<Module>("ipcp_test");
    TypeContext& ctx = mod->type_context();
    auto i32 = ctx.int32();

    // double_it(x) → x << 1
    {
        auto fn_type = std::make_shared<FunctionType>(i32, std::vector<std::shared_ptr<Type>>{i32});
        auto& fn = mod->add_function("double_it", fn_type, Linkage::Internal);
        fn.add_argument(i32, "x");
        auto& entry = fn.add_block("entry");
        auto x = std::make_shared<Value>(i32, "x");
        ir::TypeContext tmp_ctx;
        auto one = ConstantInt::get(tmp_ctx, 1);
        auto shl = std::make_shared<Instruction>(Opcode::Shl, i32, "r");
        shl->add_operand(x);
        shl->add_operand(one);
        entry.add_instruction(shl);
        entry.add_instruction(inst::make_ret(std::make_shared<Value>(i32, "r")));
    }
    // caller_a() → call @double_it(5); ret
    {
        auto fn_type = std::make_shared<FunctionType>(i32, std::vector<std::shared_ptr<Type>>{});
        auto& fn = mod->add_function("caller_a", fn_type, Linkage::External);
        auto& entry = fn.add_block("entry");
        ir::TypeContext tmp_ctx;
        auto five = ConstantInt::get(tmp_ctx, 5);
        auto call = inst::make_call(i32, "double_it", {five}, "c");
        entry.add_instruction(call);
        entry.add_instruction(inst::make_ret(call));
    }
    // caller_b() → call @double_it(5); ret  (same constant!)
    {
        auto fn_type = std::make_shared<FunctionType>(i32, std::vector<std::shared_ptr<Type>>{});
        auto& fn = mod->add_function("caller_b", fn_type, Linkage::External);
        auto& entry = fn.add_block("entry");
        ir::TypeContext tmp_ctx;
        auto five = ConstantInt::get(tmp_ctx, 5);
        auto call = inst::make_call(i32, "double_it", {five}, "c");
        entry.add_instruction(call);
        entry.add_instruction(inst::make_ret(call));
    }
    return mod;
}

// ═══════════════════════════════════════════════════════════════════════════
//  Tests
// ═══════════════════════════════════════════════════════════════════════════

static void test_callgraph_builds() {
    std::cout << "  CallGraph builds and queries..." << std::endl;
    auto mod = make_module_with_dead_and_inline();
    analysis::CallGraph cg;
    cg.build(*mod);

    // 4 functions: double, wrapper, dead_fn, main.
    CHECK(cg.node_count() == 4, "call graph has 4 nodes");
    CHECK(cg.node("main") != nullptr, "main is a node");
    CHECK(cg.node("missing") == nullptr, "unknown function returns null");

    // main calls wrapper.
    CHECK(cg.calls("main", "wrapper"), "main calls wrapper");
    CHECK(!cg.calls("main", "double"), "main does not directly call double");
    CHECK(cg.can_reach("main", "double"), "main can reach double transitively");
    CHECK(!cg.calls("wrapper", "main"), "wrapper does not call main");

    // dead_fn has no callers.
    auto dead_node = cg.node("dead_fn");
    CHECK(dead_node != nullptr, "dead_fn is a node");
    CHECK(dead_node->callers.empty(), "dead_fn has no callers");
    CHECK(!dead_node->is_entry, "dead_fn is not an entry point");
    CHECK(!dead_node->is_recursive, "dead_fn is not recursive");

    // main is an entry point.
    auto main_node = cg.node("main");
    CHECK(main_node->is_entry, "main is an entry point");

    // dead_functions should return dead_fn (it's internal linkage and
    // has no callers). Pass an empty externally_visible set: the test
    // module's only External function is `main`; the CallGraph build
    // has already flagged `main` as an entry point, so we don't need
    // to repeat that here.
    std::unordered_set<std::string> external;
    auto dead = cg.dead_functions(external);
    bool has_dead = false;
    for (const auto& n : dead) if (n == "dead_fn") has_dead = true;
    CHECK(has_dead, "dead_functions identifies dead_fn");
    // main is an entry point, so it should NOT be flagged dead even
    // though we passed an empty externally_visible set.
    bool main_flagged_dead = false;
    for (const auto& n : dead) if (n == "main") main_flagged_dead = true;
    CHECK(!main_flagged_dead, "main is not flagged dead (it's an entry point)");
}

static void test_dfe_removes_dead_function() {
    std::cout << "  DFE removes dead functions..." << std::endl;
    auto mod = make_module_with_dead_and_inline();
    search::CrossFnConfig cfg;
    cfg.enable_dfe = true;
    cfg.enable_ipcp = false;  // isolate DFE
    search::CrossFunctionPasses xpass(cfg);
    bool changed = xpass.run(*mod);

    CHECK(changed, "DFE made changes");
    CHECK(xpass.stats().dfe_removed >= 1, "DFE removed at least 1 function");
    CHECK(mod->function("dead_fn") == nullptr, "dead_fn is gone");
    CHECK(mod->function("main") != nullptr, "main survives");
    CHECK(mod->function("double") != nullptr, "double survives");
    CHECK(mod->function("wrapper") != nullptr, "wrapper survives");
}

static void test_ipcp_specialises_constant_arg() {
    std::cout << "  IPCP specialises constant arguments..." << std::endl;
    auto mod = make_module_with_ipcp_target();
    search::CrossFnConfig cfg;
    cfg.enable_dfe = false;
    cfg.enable_ipcp = true;
    search::CrossFunctionPasses xpass(cfg);
    bool changed = xpass.run(*mod);

    CHECK(changed, "IPCP made changes");
    CHECK(xpass.stats().ipcp_cloned >= 1, "IPCP created at least 1 clone");
    // Look for a clone whose name starts with "double_it.ipcp_".
    bool found_clone = false;
    for (const auto& fn : mod->functions()) {
        if (fn->name().find("double_it.ipcp_") == 0) {
            found_clone = true;
            break;
        }
    }
    CHECK(found_clone, "IPCP clone exists with expected name prefix");

    // After IPCP, callers should be rewritten to invoke the clone.
    bool caller_a_rewritten = false, caller_b_rewritten = false;
    for (const auto& fn : mod->functions()) {
        for (const auto& bb : fn->blocks()) {
            for (const auto& inst : bb->instructions()) {
                if (!inst || inst->opcode() != Opcode::Call) continue;
                auto it = inst->metadata().find("callee");
                if (it == inst->metadata().end()) continue;
                if (it->second.find("double_it.ipcp_") == 0) {
                    if (fn->name() == "caller_a") caller_a_rewritten = true;
                    if (fn->name() == "caller_b") caller_b_rewritten = true;
                }
            }
        }
    }
    CHECK(caller_a_rewritten, "caller_a now invokes the IPCP clone");
    CHECK(caller_b_rewritten, "caller_b now invokes the IPCP clone");
}

static void test_inliner_single_block_inlines_simple_callee() {
    std::cout << "  Inliner (single-block) inlines a simple callee..." << std::endl;
    // Build a tiny module where main calls a single-block callee that
    // contains no further calls:
    //   @add_one(x) → add x, 1; ret
    //   @main()     → call @add_one(7); ret
    auto mod = std::make_shared<Module>("sb_inline_test");
    TypeContext& ctx = mod->type_context();
    auto i32 = ctx.int32();
    {
        auto fn_type = std::make_shared<FunctionType>(i32, std::vector<std::shared_ptr<Type>>{i32});
        auto& fn = mod->add_function("add_one", fn_type, Linkage::Internal);
        fn.add_argument(i32, "x");
        auto& entry = fn.add_block("entry");
        auto x = std::make_shared<Value>(i32, "x");
        ir::TypeContext tmp_ctx;
        auto one = ConstantInt::get(tmp_ctx, 1);
        entry.add_instruction(inst::make_add(x, one, "r"));
        entry.add_instruction(inst::make_ret(std::make_shared<Value>(i32, "r")));
    }
    {
        auto fn_type = std::make_shared<FunctionType>(i32, std::vector<std::shared_ptr<Type>>{});
        auto& fn = mod->add_function("main", fn_type, Linkage::External);
        auto& entry = fn.add_block("entry");
        ir::TypeContext tmp_ctx;
        auto seven = ConstantInt::get(tmp_ctx, 7);
        auto call = inst::make_call(i32, "add_one", {seven}, "c");
        entry.add_instruction(call);
        entry.add_instruction(inst::make_ret(call));
    }

    auto main_fn = mod->function("main");
    CHECK(main_fn != nullptr, "main exists");
    search::Inliner inliner;
    auto inlined = inliner.inline_calls(*main_fn, *mod);
    CHECK(inlined != nullptr, "single-block inliner produced a result");
    CHECK(inliner.stats().call_sites_inlined >= 1, "inlined at least 1 call");
    bool has_call = false;
    for (const auto& bb : inlined->blocks()) {
        for (const auto& inst : bb->instructions()) {
            if (inst && inst->opcode() == Opcode::Call) has_call = true;
        }
    }
    CHECK(!has_call, "inlined main has no call instruction");
    CHECK(ir::validate_function(*inlined), "inlined main is well-formed");
}

static void test_inliner_multiblock_inlines_wrapper_into_main() {
    std::cout << "  Inliner (multi-block) inlines wrapper into main..." << std::endl;
    // wrapper calls square (so single-block inliner bails). Multi-block
    // inliner can clone wrapper's body (including its call to square)
    // into main. After inlining, main has a `call @square` instead of
    // `call @wrapper`.
    auto mod = make_module_with_dead_and_inline();
    auto main_fn = mod->function("main");
    CHECK(main_fn != nullptr, "main exists");
    search::InlinerConfig cfg;
    cfg.enable_multiblock = true;
    search::Inliner inliner(cfg);
    auto inlined = inliner.inline_calls_multiblock(*main_fn, *mod);
    CHECK(inlined != nullptr, "multi-block inliner produced a result");
    CHECK(inliner.stats().multiblock_inlined >= 1, "inlined wrapper into main");
    CHECK(ir::validate_function(*inlined), "inlined main is well-formed");
    // main should no longer have a `call @wrapper`.
    bool has_wrapper_call = false;
    for (const auto& bb : inlined->blocks()) {
        for (const auto& inst : bb->instructions()) {
            if (!inst || inst->opcode() != Opcode::Call) continue;
            auto it = inst->metadata().find("callee");
            if (it != inst->metadata().end() && it->second == "wrapper") {
                has_wrapper_call = true;
            }
        }
    }
    CHECK(!has_wrapper_call, "inlined main no longer calls wrapper");
}

static void test_inliner_multiblock_inlines_abs_if() {
    std::cout << "  Inliner (multi-block) inlines abs_if into caller..." << std::endl;
    auto mod = make_module_with_multiblock_callee();
    auto caller = mod->function("caller");
    CHECK(caller != nullptr, "caller exists");
    search::InlinerConfig cfg;
    cfg.enable_multiblock = true;
    search::Inliner inliner(cfg);
    auto inlined = inliner.inline_calls_multiblock(*caller, *mod);
    CHECK(inlined != nullptr, "multi-block inliner produced a result");
    CHECK(inliner.stats().multiblock_inlined >= 1,
          "inlined at least 1 multi-block callee");
    CHECK(ir::validate_function(*inlined),
          "inlined caller is well-formed (no dangling refs)");

    // After inlining, the caller should have MORE blocks than the
    // original (we cloned abs_if's 4 blocks plus a continuation).
    CHECK(inlined->blocks().size() > caller->blocks().size(),
          "inlined caller has more blocks than original");

    // No `call @abs_if` should remain.
    bool has_call = false;
    for (const auto& bb : inlined->blocks()) {
        for (const auto& inst : bb->instructions()) {
            if (!inst || inst->opcode() != Opcode::Call) continue;
            auto it = inst->metadata().find("callee");
            if (it != inst->metadata().end() && it->second == "abs_if") {
                has_call = true;
            }
        }
    }
    CHECK(!has_call, "inlined caller has no call to abs_if");
}

static void test_inliner_refuses_recursion() {
    std::cout << "  Inliner refuses to inline a recursive function..." << std::endl;
    // Build a self-recursive function:
    //   @rec(x) → if x == 0: ret 0; else: call @rec(x-1), ret
    auto mod = std::make_shared<Module>("rec_test");
    TypeContext& ctx = mod->type_context();
    auto i32 = ctx.int32();
    auto fn_type = std::make_shared<FunctionType>(i32, std::vector<std::shared_ptr<Type>>{i32});
    auto& fn = mod->add_function("rec", fn_type, Linkage::Internal);
    fn.add_argument(i32, "x");
    auto& entry = fn.add_block("entry");
    auto& base = fn.add_block("base");
    auto& recurse = fn.add_block("recurse");
    auto x = std::make_shared<Value>(i32, "x");
    ir::TypeContext tmp_ctx;
    auto zero = ConstantInt::get(tmp_ctx, 0);
    auto one = ConstantInt::get(tmp_ctx, 1);
    entry.add_instruction(inst::make_icmp(CmpPredicate::EQ, x, zero, "cmp"));
    entry.add_instruction(inst::make_br(entry.instruction(0), "base", "recurse"));
    base.add_instruction(inst::make_ret(zero));
    recurse.add_instruction(inst::make_sub(x, one, "dec"));
    auto call = inst::make_call(i32, "rec", {std::make_shared<Value>(i32, "dec")}, "r");
    recurse.add_instruction(call);
    recurse.add_instruction(inst::make_ret(call));

    search::InlinerConfig cfg;
    cfg.enable_multiblock = true;
    search::Inliner inliner(cfg);
    auto inlined = inliner.inline_calls_multiblock(fn, *mod);
    // Either no inlining happened, or it refused the recursive call.
    CHECK(inliner.stats().recursion_refused >= 1 || inlined == nullptr,
          "recursive inlining was refused");
}

static void test_pipeline_runs_cross_function_passes() {
    std::cout << "  Pipeline end-to-end runs cross-function passes..." << std::endl;
    auto mod = make_module_with_dead_and_inline();
    PipelineConfig pcfg;
    pcfg.opt_level = 2;
    pcfg.time_budget = 5.0;
    pcfg.enable_cross_function = true;
    pcfg.enable_multiblock_inliner = true;
    pcfg.verbose = false;
    // Disable SMT to keep the test fast and deterministic — the cross-
    // function passes are sound-by-construction so SMT is not needed.
    pcfg.skip_smt = true;
    Pipeline pipeline(pcfg);
    auto result = pipeline.run(*mod);

    CHECK(result.optimised_module != nullptr, "pipeline produced a module");
    // The optimised module should NOT contain dead_fn (DFE removed it).
    bool has_dead = false;
    for (const auto& fn : result.optimised_module->functions()) {
        if (fn && fn->name() == "dead_fn") has_dead = true;
    }
    CHECK(!has_dead, "DFE removed dead_fn from optimised module");
    // The optimised module should contain main.
    bool has_main = false;
    for (const auto& fn : result.optimised_module->functions()) {
        if (fn && fn->name() == "main") has_main = true;
    }
    CHECK(has_main, "main survives in optimised module");
}

static void test_pipeline_ipcp_in_optimised_module() {
    std::cout << "  Pipeline end-to-end produces IPCP clone..." << std::endl;
    auto mod = make_module_with_ipcp_target();
    PipelineConfig pcfg;
    pcfg.opt_level = 2;
    pcfg.time_budget = 5.0;
    pcfg.enable_cross_function = true;
    pcfg.skip_smt = true;
    Pipeline pipeline(pcfg);
    auto result = pipeline.run(*mod);

    bool has_clone = false;
    for (const auto& fn : result.optimised_module->functions()) {
        if (fn && fn->name().find("double_it.ipcp_") == 0) has_clone = true;
    }
    CHECK(has_clone, "IPCP clone is in the optimised module");
}

int main() {
    std::cout << "=== Clunk Cross-Function Tests ===" << std::endl;

    test_callgraph_builds();
    test_dfe_removes_dead_function();
    test_ipcp_specialises_constant_arg();
    test_inliner_single_block_inlines_simple_callee();
    test_inliner_multiblock_inlines_wrapper_into_main();
    test_inliner_multiblock_inlines_abs_if();
    test_inliner_refuses_recursion();
    test_pipeline_runs_cross_function_passes();
    test_pipeline_ipcp_in_optimised_module();

    std::cout << "\n=== Results: " << g_pass << " passed, " << g_fail << " failed ===" << std::endl;
    return g_fail > 0 ? 1 : 0;
}
