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
 * Clunk Evaluator Tests — test the evaluation engine and heuristic analysis.
 */
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>

#include "clunk/IR/Type.h"
#include "clunk/IR/Value.h"
#include "clunk/IR/Instruction.h"
#include "clunk/IR/BasicBlock.h"
#include "clunk/IR/Function.h"
#include "clunk/IR/Module.h"
#include "clunk/IR/IRBuilder.h"
#include "clunk/Evaluator/EvaluationEngine.h"
#include "clunk/Evaluator/CostModel.h"
#include "clunk/Evaluator/EvaluationCache.h"
#include "clunk/Evaluator/Interpreter.h"
#include "clunk/Parser/IRParser.h"

static int g_pass = 0, g_fail = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { std::cerr << "FAIL: " << msg << " at " << __FILE__ << ":" << __LINE__ << "\n"; g_fail++; } \
    else { g_pass++; } \
} while(0)

using namespace clunk::ir;
using namespace clunk::evaluator;

// ═══════════════════════════════════════════════════════════════════════════
//  Helper: build a simple function for testing
// ═══════════════════════════════════════════════════════════════════════════

static std::shared_ptr<Function> make_simple_add_function(Module& mod) {
    TypeContext& ctx = mod.type_context();
    auto fn_type = std::make_shared<FunctionType>(ctx.int32(), std::vector<std::shared_ptr<Type>>{ctx.int32(), ctx.int32()});
    auto& fn = mod.add_function("add", fn_type);
    fn.add_argument(ctx.int32(), "a");
    fn.add_argument(ctx.int32(), "b");

    auto& entry = fn.add_block("entry");
    auto a_val = ConstantInt::get(ctx, 1, 32);
    auto b_val = ConstantInt::get(ctx, 2, 32);
    entry.add_instruction(inst::make_add(a_val, b_val, "sum"));
    entry.add_instruction(inst::make_ret(entry.instruction(0)));

    return mod.function("add");
}

static std::shared_ptr<Function> make_compute_heavy_function(Module& mod) {
    TypeContext& ctx = mod.type_context();
    auto fn_type = std::make_shared<FunctionType>(ctx.int32(), std::vector<std::shared_ptr<Type>>{ctx.int32()});
    auto& fn = mod.add_function("compute", fn_type);
    fn.add_argument(ctx.int32(), "x");

    auto& entry = fn.add_block("entry");
    IRBuilder builder(ctx);
    builder.set_insert_point(&entry);

    auto x = builder.get_int32(5);
    auto a = builder.create_add(x, x, "a");
    auto b = builder.create_mul(a, x, "b");
    auto c = builder.create_sub(b, x, "c");
    auto d = builder.create_mul(c, c, "d");
    builder.create_ret(d);

    return mod.function("compute");
}

static std::shared_ptr<Function> make_memory_heavy_function(Module& mod) {
    TypeContext& ctx = mod.type_context();
    auto fn_type = std::make_shared<FunctionType>(ctx.int32(), std::vector<std::shared_ptr<Type>>{});
    auto& fn = mod.add_function("memtest", fn_type);

    auto& entry = fn.add_block("entry");
    IRBuilder builder(ctx);
    builder.set_insert_point(&entry);

    auto slot1 = builder.create_alloca(ctx.int32(), "slot1", 4);
    auto slot2 = builder.create_alloca(ctx.int32(), "slot2", 4);
    auto val1 = builder.get_int32(10);
    auto val2 = builder.get_int32(20);
    builder.create_store(val1, slot1, 4);
    builder.create_store(val2, slot2, 4);
    auto ld1 = builder.create_load(slot1, "v1", 4);
    auto ld2 = builder.create_load(slot2, "v2", 4);
    auto sum = builder.create_add(ld1, ld2, "sum");
    builder.create_ret(sum);

    return mod.function("memtest");
}

// ═══════════════════════════════════════════════════════════════════════════
//  Helpers for new tests
// ═══════════════════════════════════════════════════════════════════════════

// Build a function `cheap(i32 %a, i32 %b) { entry: %s = add i32 %a, %b; ret i32 %s }`
// — minimal cost: just one add + one ret.
static std::shared_ptr<Function> make_cheap_add_function(Module& mod) {
    TypeContext& ctx = mod.type_context();
    auto fn_type = std::make_shared<FunctionType>(
        ctx.int32(), std::vector<std::shared_ptr<Type>>{ctx.int32(), ctx.int32()});
    auto& fn = mod.add_function("cheap_add", fn_type);
    fn.add_argument(ctx.int32(), "a");
    fn.add_argument(ctx.int32(), "b");

    auto a_val = std::make_shared<Value>(ctx.int32(), "a");
    auto b_val = std::make_shared<Value>(ctx.int32(), "b");

    auto& entry = fn.add_block("entry");
    entry.add_instruction(inst::make_add(a_val, b_val, "sum"));
    entry.add_instruction(inst::make_ret(entry.instruction(0)));
    return mod.function("cheap_add");
}

// Build a function with several memory accesses — heavier cost than `cheap`.
static std::shared_ptr<Function> make_memory_heavy_v2(Module& mod,
                                                      const std::string& name) {
    TypeContext& ctx = mod.type_context();
    auto fn_type = std::make_shared<FunctionType>(
        ctx.int32(), std::vector<std::shared_ptr<Type>>{});
    auto& fn = mod.add_function(name, fn_type);

    auto& entry = fn.add_block("entry");
    IRBuilder builder(ctx);
    builder.set_insert_point(&entry);

    auto slot1 = builder.create_alloca(ctx.int32(), "slot1", 4);
    auto slot2 = builder.create_alloca(ctx.int32(), "slot2", 4);
    auto slot3 = builder.create_alloca(ctx.int32(), "slot3", 4);
    auto v1 = builder.get_int32(10);
    auto v2 = builder.get_int32(20);
    auto v3 = builder.get_int32(30);
    builder.create_store(v1, slot1, 4);
    builder.create_store(v2, slot2, 4);
    builder.create_store(v3, slot3, 4);
    auto l1 = builder.create_load(slot1, "l1", 4);
    auto l2 = builder.create_load(slot2, "l2", 4);
    auto l3 = builder.create_load(slot3, "l3", 4);
    auto s1 = builder.create_add(l1, l2, "s1");
    auto s2 = builder.create_add(s1, l3, "s2");
    builder.create_ret(s2);
    return mod.function(name);
}

// ═══════════════════════════════════════════════════════════════════════════
//  Sign-convention regression tests 
// ═══════════════════════════════════════════════════════════════════════════

void test_sign_convention_better_candidate() {
    // Original = memory-heavy (expensive). Candidate = cheap add (cheap).
    // ratio = orig_abs / cand_abs. |cand| < |orig| → ratio > 1.0.
    Module mod_orig("sign_orig");
    auto orig = make_memory_heavy_v2(mod_orig, "orig");

    Module mod_cand("sign_cand");
    auto better = make_cheap_add_function(mod_cand);

    EvaluationEngine engine;
    double ratio = engine.score_candidate(*orig, *better);
    CHECK(ratio > 1.0, "better candidate yields ratio > 1.0 ");
    CHECK(std::isfinite(ratio), "ratio is finite");
}

void test_sign_convention_worse_candidate() {
    // Original = cheap add (cheap). Candidate = memory-heavy (expensive).
    // ratio = orig_abs / cand_abs. |cand| > |orig| → ratio < 1.0.
    Module mod_orig("sign_orig2");
    auto orig = make_cheap_add_function(mod_orig);

    Module mod_cand("sign_cand2");
    auto worse = make_memory_heavy_v2(mod_cand, "worse");

    EvaluationEngine engine;
    double ratio = engine.score_candidate(*orig, *worse);
    CHECK(ratio < 1.0, "worse candidate yields ratio < 1.0 ");
    CHECK(std::isfinite(ratio), "ratio is finite");
}

void test_sign_convention_identical_candidate() {
    // Identical functions → ratio ≈ 1.0 (covers the existing
    // test_score_candidate but with explicit math).
    Module mod1("id1");
    Module mod2("id2");
    auto fn1 = make_cheap_add_function(mod1);
    auto fn2 = make_cheap_add_function(mod2);

    EvaluationEngine engine;
    double ratio = engine.score_candidate(*fn1, *fn2);
    CHECK(ratio >= 0.999 && ratio <= 1.001,
          "identical functions yield ratio ~ 1.0");
}

// ═══════════════════════════════════════════════════════════════════════════
//  EvaluationCache tests
// ═══════════════════════════════════════════════════════════════════════════

void test_evaluation_cache_hits_misses() {
    Module mod("cache_test");
    auto fn = make_cheap_add_function(mod);

    EvaluationEngine engine;
    engine.clear_cache();
    auto stats0 = engine.cache_stats();
    CHECK(stats0.hits == 0 && stats0.misses == 0,
          "cache stats start at zero after clear");

    (void)engine.analyse(*fn);                 // miss + insert
    (void)engine.analyse(*fn);                 // hit
    (void)engine.analyse(*fn);                 // hit

    auto stats1 = engine.cache_stats();
    CHECK(stats1.misses >= 1, "first analyse is a miss");
    CHECK(stats1.hits >= 2,    "repeat analyses are hits");
    CHECK(stats1.size >= 1,    "cache has at least one entry");
}

void test_evaluation_cache_clear() {
    Module mod("cache_clear");
    auto fn = make_cheap_add_function(mod);
    EvaluationEngine engine;
    (void)engine.analyse(*fn);
    auto before = engine.cache_stats();
    CHECK(before.size >= 1, "cache populated");
    engine.clear_cache();
    auto after = engine.cache_stats();
    CHECK(after.size == 0, "cache empty after clear");
    // Counters persist (intentionally — they're cumulative observability).
    CHECK(after.hits == before.hits, "hit counter preserved across clear");
}

void test_structural_hash_stability() {
    Module mod1("hash1");
    Module mod2("hash2");
    auto fn1 = make_cheap_add_function(mod1);
    auto fn2 = make_cheap_add_function(mod2);
    uint64_t h1 = structural_hash(*fn1);
    uint64_t h2 = structural_hash(*fn2);
    CHECK(h1 != 0, "hash is non-zero");
    CHECK(h1 == h2, "structurally identical functions hash equally");
}

// ═══════════════════════════════════════════════════════════════════════════
//  CostModel tests
// ═══════════════════════════════════════════════════════════════════════════

void test_cost_model_factory() {
    auto x86  = make_cost_model(Arch::X86_64);
    auto arm  = make_cost_model(Arch::AArch64);
    auto ptx  = make_cost_model(Arch::PTX);
    auto gen  = make_cost_model(Arch::Generic);

    CHECK(x86 && arm && ptx && gen, "factory returns non-null models");
    CHECK(x86->arch() == Arch::X86_64,  "x86 model arch tag");
    CHECK(arm->arch() == Arch::AArch64, "arm model arch tag");
    CHECK(ptx->arch() == Arch::PTX,     "ptx model arch tag");
    CHECK(gen->arch() == Arch::Generic, "generic model arch tag");
}

void test_cost_model_per_arch_differs() {
    auto x86 = make_cost_model(Arch::X86_64);
    auto ptx = make_cost_model(Arch::PTX);

    auto mul_x86 = x86->cost(Opcode::Mul);
    auto mul_ptx = ptx->cost(Opcode::Mul);
    CHECK(mul_x86.latency_cycles != mul_ptx.latency_cycles ||
          mul_x86.uops            != mul_ptx.uops,
          "x86 vs PTX Mul cost differs (per-arch tables are distinct)");

    auto load_x86 = x86->cost(Opcode::Load);
    auto load_ptx = ptx->cost(Opcode::Load);
    CHECK(load_ptx.latency_cycles > load_x86.latency_cycles,
          "PTX load latency > x86 load latency (GPU memory is slower)");
}

void test_cost_model_for_name() {
    auto m1 = make_cost_model_for_name("x86_64");
    auto m2 = make_cost_model_for_name("aarch64");
    auto m3 = make_cost_model_for_name("ptx");
    auto m4 = make_cost_model_for_name("sm_80");
    auto m5 = make_cost_model_for_name("unknown_arch");
    CHECK(m1->arch() == Arch::X86_64,  "x86_64 name → X86_64 model");
    CHECK(m2->arch() == Arch::AArch64, "aarch64 name → AArch64 model");
    CHECK(m3->arch() == Arch::PTX,     "ptx name → PTX model");
    CHECK(m4->arch() == Arch::PTX,     "sm_80 name → PTX model");
    CHECK(m5->arch() == Arch::Generic, "unknown name → Generic fallback");
}

void test_engine_default_cost_model_is_x86() {
    EvaluationEngine engine;
    CHECK(engine.cost_model().arch() == Arch::X86_64,
          "default EvaluationEngine uses X86_64 cost model");
}

void test_engine_set_cost_model() {
    EvaluationEngine engine;
    engine.set_cost_model(make_cost_model(Arch::PTX));
    CHECK(engine.cost_model().arch() == Arch::PTX,
          "set_cost_model installs PTX model");
}

// ═══════════════════════════════════════════════════════════════════════════
//  Critical-path analysis tests
// ═══════════════════════════════════════════════════════════════════════════

void test_critical_path_long_chain_beats_parallel() {
    // Chain:  a = x + x; b = a + a; c = b + b;   (3 dependent adds)
    // Parallel: a = x + x; b = x + x; c = x + x; (3 independent adds)
    // Same total uops, but the chain has a longer critical path.
    Module mod_chain("cp_chain");
    {
        TypeContext& ctx = mod_chain.type_context();
        auto fn_type = std::make_shared<FunctionType>(
            ctx.int32(), std::vector<std::shared_ptr<Type>>{ctx.int32()});
        auto& fn = mod_chain.add_function("chain", fn_type);
        fn.add_argument(ctx.int32(), "x");
        auto xv = std::make_shared<Value>(ctx.int32(), "x");
        auto& entry = fn.add_block("entry");
        IRBuilder builder(ctx);
        builder.set_insert_point(&entry);
        auto a = builder.create_add(xv, xv, "a");
        auto b = builder.create_add(a, a, "b");
        auto c = builder.create_add(b, b, "c");
        builder.create_ret(c);
    }

    Module mod_par("cp_parallel");
    {
        TypeContext& ctx = mod_par.type_context();
        auto fn_type = std::make_shared<FunctionType>(
            ctx.int32(), std::vector<std::shared_ptr<Type>>{ctx.int32()});
        auto& fn = mod_par.add_function("parallel", fn_type);
        fn.add_argument(ctx.int32(), "x");
        auto xv = std::make_shared<Value>(ctx.int32(), "x");
        auto& entry = fn.add_block("entry");
        IRBuilder builder(ctx);
        builder.set_insert_point(&entry);
        auto a = builder.create_add(xv, xv, "a");
        auto b = builder.create_add(xv, xv, "b");
        auto c = builder.create_add(xv, xv, "c");
        builder.create_ret(c);
        (void)a; (void)b; (void)c;
    }

    auto chain_fn = mod_chain.function("chain");
    auto par_fn   = mod_par.function("parallel");
    CHECK(chain_fn && par_fn, "both functions built");

    EvaluationEngine engine;
    auto chain_an = engine.analyse(*chain_fn);
    auto par_an   = engine.analyse(*par_fn);
    CHECK(chain_an.critical_path_cycles > par_an.critical_path_cycles,
          "dependent chain has longer critical path than parallel adds");
}

// ═══════════════════════════════════════════════════════════════════════════
//  Interpreter tests (validation oracle)
// ═══════════════════════════════════════════════════════════════════════════

void test_interpreter_simple_add() {
    // Build add(a, b) = a + b, interpret with {5, 7}, expect 12.
    Module mod("interp_add");
    TypeContext& ctx = mod.type_context();
    auto fn_type = std::make_shared<FunctionType>(
        ctx.int32(), std::vector<std::shared_ptr<Type>>{ctx.int32(), ctx.int32()});
    auto& fn = mod.add_function("add", fn_type);
    fn.add_argument(ctx.int32(), "a");
    fn.add_argument(ctx.int32(), "b");
    auto a_val = std::make_shared<Value>(ctx.int32(), "a");
    auto b_val = std::make_shared<Value>(ctx.int32(), "b");
    auto& entry = fn.add_block("entry");
    entry.add_instruction(inst::make_add(a_val, b_val, "sum"));
    entry.add_instruction(inst::make_ret(entry.instruction(0)));

    auto result = Interpreter::interpret(fn, {5, 7});
    CHECK(result.has_value(), "interpreter returns a value for simple add");
    CHECK(*result == 12, "interpreter computes 5 + 7 = 12");
}

void test_interpreter_memory_ops() {
    // alloca; store 42; load; ret. Expect 42.
    Module mod("interp_mem");
    TypeContext& ctx = mod.type_context();
    auto fn_type = std::make_shared<FunctionType>(
        ctx.int32(), std::vector<std::shared_ptr<Type>>{});
    auto& fn = mod.add_function("mem", fn_type);
    auto& entry = fn.add_block("entry");
    IRBuilder builder(ctx);
    builder.set_insert_point(&entry);
    auto slot = builder.create_alloca(ctx.int32(), "slot", 4);
    auto val  = builder.get_int32(42);
    builder.create_store(val, slot, 4);
    auto ld = builder.create_load(slot, "ld", 4);
    builder.create_ret(ld);

    auto result = Interpreter::interpret(fn, {});
    CHECK(result.has_value(), "interpreter handles alloca/store/load");
    CHECK(*result == 42, "interpreter reads back stored value 42");
}

void test_interpreter_arithmetic_chain() {
    // Compute (a*a + b*b) with a=3, b=4 → 25.
    Module mod("interp_chain");
    TypeContext& ctx = mod.type_context();
    auto fn_type = std::make_shared<FunctionType>(
        ctx.int32(),
        std::vector<std::shared_ptr<Type>>{ctx.int32(), ctx.int32()});
    auto& fn = mod.add_function("chain", fn_type);
    fn.add_argument(ctx.int32(), "a");
    fn.add_argument(ctx.int32(), "b");
    auto a_val = std::make_shared<Value>(ctx.int32(), "a");
    auto b_val = std::make_shared<Value>(ctx.int32(), "b");
    auto& entry = fn.add_block("entry");
    IRBuilder builder(ctx);
    builder.set_insert_point(&entry);
    auto aa = builder.create_mul(a_val, a_val, "aa");
    auto bb = builder.create_mul(b_val, b_val, "bb");
    auto sum = builder.create_add(aa, bb, "sum");
    builder.create_ret(sum);

    auto result = Interpreter::interpret(fn, {3, 4});
    CHECK(result.has_value(), "interpreter returns value for arithmetic chain");
    CHECK(*result == 25, "interpreter computes 3*3 + 4*4 = 25");
}

void test_interpreter_conditional_branch() {
    // if (a >= b) return a; else return b;  (max(a, b))
    Module mod("interp_branch");
    TypeContext& ctx = mod.type_context();
    auto fn_type = std::make_shared<FunctionType>(
        ctx.int32(),
        std::vector<std::shared_ptr<Type>>{ctx.int32(), ctx.int32()});
    auto& fn = mod.add_function("max", fn_type);
    fn.add_argument(ctx.int32(), "a");
    fn.add_argument(ctx.int32(), "b");
    auto a_val = std::make_shared<Value>(ctx.int32(), "a");
    auto b_val = std::make_shared<Value>(ctx.int32(), "b");

    auto& entry = fn.add_block("entry");
    auto& then_bb = fn.add_block("then");
    auto& else_bb = fn.add_block("else");

    IRBuilder builder(ctx);
    builder.set_insert_point(&entry);
    auto cmp = builder.create_icmp(CmpPredicate::SGE, a_val, b_val, "cond");
    builder.create_cond_br(cmp, "then", "else");

    builder.set_insert_point(&then_bb);
    builder.create_ret(a_val);

    builder.set_insert_point(&else_bb);
    builder.create_ret(b_val);

    auto r1 = Interpreter::interpret(fn, {10, 5});
    CHECK(r1.has_value() && *r1 == 10, "max(10, 5) == 10");

    auto r2 = Interpreter::interpret(fn, {3, 8});
    CHECK(r2.has_value() && *r2 == 8, "max(3, 8) == 8");
}

// ═══════════════════════════════════════════════════════════════════════════
//  Existing tests
// ═══════════════════════════════════════════════════════════════════════════

void test_analyse_simple_function() {
    Module mod("eval_test");
    auto fn = make_simple_add_function(mod);
    EvaluationEngine engine;
    auto result = engine.analyse(*fn);

    CHECK(result.function_name == "add", "function_name matches");
    CHECK(result.total_instructions == 2, "simple add has 2 instructions");
    CHECK(result.compute_instructions >= 1, "has >= 1 compute instruction (add)");
}

void test_analyse_compute_heavy() {
    Module mod("eval_compute");
    auto fn = make_compute_heavy_function(mod);
    EvaluationEngine engine;
    auto result = engine.analyse(*fn);

    CHECK(result.function_name == "compute", "function_name is compute");
    CHECK(result.total_instructions >= 5, "compute-heavy has >= 5 instructions");
    CHECK(result.compute_instructions >= 3, "has >= 3 compute instructions");
    CHECK(result.task_weight > 0.0, "task_weight > 0");
}

void test_analyse_memory_heavy() {
    Module mod("eval_mem");
    auto fn = make_memory_heavy_function(mod);
    EvaluationEngine engine;
    auto result = engine.analyse(*fn);

    CHECK(result.function_name == "memtest", "function_name is memtest");
    CHECK(result.memory_instructions >= 4, "has >= 4 memory instructions (2 store + 2 load)");
    CHECK(result.total_instructions >= 6, "has >= 6 total instructions");
}

void test_analyse_module() {
    Module mod("eval_mod");
    make_simple_add_function(mod);
    make_compute_heavy_function(mod);

    EvaluationEngine engine;
    auto results = engine.analyse_module(mod);
    CHECK(results.size() == 2, "analysed 2 functions");
    CHECK(results.count("add") == 1, "has 'add' result");
    CHECK(results.count("compute") == 1, "has 'compute' result");
}

void test_score_candidate() {
    Module mod("eval_score");
    auto fn = make_simple_add_function(mod);

    // Create a candidate (identical for now)
    TypeContext& ctx = mod.type_context();
    auto fn_type = std::make_shared<FunctionType>(ctx.int32(), std::vector<std::shared_ptr<Type>>{ctx.int32(), ctx.int32()});
    Function candidate("add_opt", fn_type);
    candidate.add_argument(ctx.int32(), "a");
    candidate.add_argument(ctx.int32(), "b");
    auto& entry = candidate.add_block("entry");
    auto a_val = ConstantInt::get(ctx, 1, 32);
    auto b_val = ConstantInt::get(ctx, 2, 32);
    entry.add_instruction(inst::make_add(a_val, b_val, "sum"));
    entry.add_instruction(inst::make_ret(entry.instruction(0)));

    EvaluationEngine engine;
    double ratio = engine.score_candidate(*fn, candidate);
    // Identical functions should give ratio close to 1.0
    CHECK(ratio > 0.0, "score_candidate returns positive ratio");
    // For identical functions, improvement ratio should be ~1.0
    CHECK(ratio >= 0.9 && ratio <= 1.1, "identical functions have ratio ~1.0");
}

void test_memory_penalty() {
    HeuristicWeights weights;
    EvaluationEngine engine(weights);

    CHECK(engine.memory_penalty(MemoryLayer::Register) == weights.register_penalty, "register penalty");
    CHECK(engine.memory_penalty(MemoryLayer::L1) == weights.l1_penalty, "L1 penalty");
    CHECK(engine.memory_penalty(MemoryLayer::L2) == weights.l2_penalty, "L2 penalty");
    CHECK(engine.memory_penalty(MemoryLayer::L3) == weights.l3_penalty, "L3 penalty");
    CHECK(engine.memory_penalty(MemoryLayer::DRAM) == weights.dram_penalty, "DRAM penalty");
    CHECK(engine.memory_penalty(MemoryLayer::NVMe) == weights.nvme_penalty, "NVMe penalty");
    CHECK(engine.memory_penalty(MemoryLayer::Network) == weights.network_penalty, "Network penalty");

    // Progressive penalties: slower layers should have more negative penalties
    CHECK(engine.memory_penalty(MemoryLayer::L1) > engine.memory_penalty(MemoryLayer::L2),
          "L1 penalty less severe than L2");
    CHECK(engine.memory_penalty(MemoryLayer::L2) > engine.memory_penalty(MemoryLayer::DRAM),
          "L2 penalty less severe than DRAM");
    CHECK(engine.memory_penalty(MemoryLayer::DRAM) > engine.memory_penalty(MemoryLayer::Network),
          "DRAM penalty less severe than Network");
}

void test_core_placement() {
    Module mod("eval_core");
    auto fn = make_compute_heavy_function(mod);

    EvaluationEngine engine;
    auto result = engine.analyse(*fn);

    // CorePreference should be one of the valid enum values
    CHECK(result.core_preference == FunctionAnalysis::CorePreference::P_Core ||
          result.core_preference == FunctionAnalysis::CorePreference::E_Core ||
          result.core_preference == FunctionAnalysis::CorePreference::Any,
          "core_preference is valid");
}

void test_task_weight_estimation() {
    Module mod("eval_weight");

    // Simple function should have lower task weight
    auto simple = make_simple_add_function(mod);
    auto compute = make_compute_heavy_function(mod);

    EvaluationEngine engine;
    auto simple_result = engine.analyse(*simple);
    auto compute_result = engine.analyse(*compute);

    CHECK(simple_result.task_weight >= 0.0, "simple task_weight >= 0");
    CHECK(compute_result.task_weight >= 0.0, "compute task_weight >= 0");
    CHECK(compute_result.task_weight >= simple_result.task_weight,
          "compute-heavy function has >= task weight than simple");
}

void test_custom_weights() {
    HeuristicWeights custom;
    custom.dram_penalty = -100.0;
    custom.div_latency_penalty = -50.0;
    custom.register_reuse_bonus = 10.0;

    EvaluationEngine engine(custom);
    CHECK(engine.weights().dram_penalty == -100.0, "custom DRAM penalty");
    CHECK(engine.weights().div_latency_penalty == -50.0, "custom div penalty");
    CHECK(engine.weights().register_reuse_bonus == 10.0, "custom register reuse bonus");
}

void test_memory_layer_name() {
    CHECK(!memory_layer_name(MemoryLayer::Register).empty(), "Register has name");
    CHECK(!memory_layer_name(MemoryLayer::DRAM).empty(), "DRAM has name");
    CHECK(!memory_layer_name(MemoryLayer::GPU_Global).empty(), "GPU_Global has name");
    CHECK(!memory_layer_name(MemoryLayer::Network).empty(), "Network has name");
}

void test_block_scores() {
    Module mod("eval_blocks");
    TypeContext& ctx = mod.type_context();

    // Create function with multiple blocks
    auto fn_type = std::make_shared<FunctionType>(ctx.int32(), std::vector<std::shared_ptr<Type>>{ctx.int32()});
    auto& fn = mod.add_function("branching", fn_type);
    fn.add_argument(ctx.int32(), "x");

    auto& entry = fn.add_block("entry");
    auto& then_bb = fn.add_block("then");
    auto& else_bb = fn.add_block("else");
    auto& merge = fn.add_block("merge");

    IRBuilder builder(ctx);
    builder.set_insert_point(&entry);
    auto zero = builder.get_int32(0);
    auto xv = builder.get_int32(5);
    auto cmp = builder.create_icmp(CmpPredicate::SGE, xv, zero, "cond");
    builder.create_cond_br(cmp, "then", "else");

    builder.set_insert_point(&then_bb);
    builder.create_br("merge");

    builder.set_insert_point(&else_bb);
    builder.create_br("merge");

    builder.set_insert_point(&merge);
    builder.create_ret(zero);

    EvaluationEngine engine;
    auto result = engine.analyse(fn);

    CHECK(result.block_scores.size() >= 1, "has block scores");
    for (auto& [name, score] : result.block_scores) {
        CHECK(!name.empty(), "block score name is non-empty");
    }
}

// ═══════════════════════════════════════════════════════════════════════════

// ── LLVM-optimal-code knowledge: constant materialisation + free bitcast ────

static std::shared_ptr<Function> eval_parse_fn(const std::string& ir,
                                               const std::string& name) {
    clunk::parser::IRParser p;
    auto mod = p.parse_string(ir);
    return mod ? mod->function(name) : nullptr;
}

// A "large" immediate (one that doesn't fit a signed 32-bit field) costs more
// than a small one: real codegen must materialise it with an extra mov /
// constant-pool load. Same shape, only the constant magnitude differs, so
// `xor x, HUGE` must score WORSE (lower — higher is better) than `xor x, 255`.
void test_large_constant_costs_more() {
    auto small = eval_parse_fn(R"(
define i64 @xs(i64 %x) {
entry:
  %r = xor i64 %x, 255
  ret i64 %r
}
)", "xs");
    auto large = eval_parse_fn(R"(
define i64 @xl(i64 %x) {
entry:
  %r = xor i64 %x, 78187493520
  ret i64 %r
}
)", "xl");
    CHECK(small && large, "parsed small/large-constant functions");
    if (!small || !large) return;

    EvaluationEngine engine;
    double s_small = engine.analyse(*small).score;
    double s_large = engine.analyse(*large).score;
    CHECK(s_small > s_large,
          "large immediate scores worse than a small one (materialisation cost)");
}

// A bitcast is free: adding a `bitcast` in front of a use must not change the
// score, so the search can't chase a phantom bitcast cost.
void test_bitcast_is_free() {
    auto nobc = eval_parse_fn(R"(
define i32 @nobc(i32 %x) {
entry:
  %r = add i32 %x, %x
  ret i32 %r
}
)", "nobc");
    auto withbc = eval_parse_fn(R"(
define i32 @withbc(i32 %x) {
entry:
  %c = bitcast i32 %x to i32
  %r = add i32 %c, %c
  ret i32 %r
}
)", "withbc");
    CHECK(nobc && withbc, "parsed bitcast functions");
    if (!nobc || !withbc) return;

    EvaluationEngine engine;
    double s_base = engine.analyse(*nobc).score;
    double s_bc   = engine.analyse(*withbc).score;
    CHECK(std::abs(s_base - s_bc) < 1e-9,
          "an extra bitcast adds no cost (bitcast is free)");
}

int main() {
    std::cout << "=== Clunk Evaluator Tests ===" << std::endl;

    std::cout << "  Simple function analysis..." << std::endl;
    test_analyse_simple_function();

    std::cout << "  Compute-heavy function..." << std::endl;
    test_analyse_compute_heavy();

    std::cout << "  Memory-heavy function..." << std::endl;
    test_analyse_memory_heavy();

    std::cout << "  Module analysis..." << std::endl;
    test_analyse_module();

    std::cout << "  Score candidate..." << std::endl;
    test_score_candidate();

    std::cout << "  Memory penalty..." << std::endl;
    test_memory_penalty();

    std::cout << "  Core placement..." << std::endl;
    test_core_placement();

    std::cout << "  Task weight estimation..." << std::endl;
    test_task_weight_estimation();

    std::cout << "  Custom weights..." << std::endl;
    test_custom_weights();

    std::cout << "  Memory layer name..." << std::endl;
    test_memory_layer_name();

    std::cout << "  Block scores..." << std::endl;
    test_block_scores();

    std::cout << "  Sign convention (better candidate)..." << std::endl;
    test_sign_convention_better_candidate();

    std::cout << "  Sign convention (worse candidate)..." << std::endl;
    test_sign_convention_worse_candidate();

    std::cout << "  Sign convention (identical candidate)..." << std::endl;
    test_sign_convention_identical_candidate();

    std::cout << "  EvaluationCache hits/misses..." << std::endl;
    test_evaluation_cache_hits_misses();

    std::cout << "  EvaluationCache clear..." << std::endl;
    test_evaluation_cache_clear();

    std::cout << "  Structural hash stability..." << std::endl;
    test_structural_hash_stability();

    std::cout << "  CostModel factory..." << std::endl;
    test_cost_model_factory();

    std::cout << "  CostModel per-arch differs..." << std::endl;
    test_cost_model_per_arch_differs();

    std::cout << "  CostModel name factory..." << std::endl;
    test_cost_model_for_name();

    std::cout << "  Engine default cost model is x86_64..." << std::endl;
    test_engine_default_cost_model_is_x86();

    std::cout << "  Engine set_cost_model..." << std::endl;
    test_engine_set_cost_model();

    std::cout << "  Critical path (chain > parallel)..." << std::endl;
    test_critical_path_long_chain_beats_parallel();

    std::cout << "  Interpreter simple_add..." << std::endl;
    test_interpreter_simple_add();

    std::cout << "  Interpreter memory ops..." << std::endl;
    test_interpreter_memory_ops();

    std::cout << "  Interpreter arithmetic chain..." << std::endl;
    test_interpreter_arithmetic_chain();

    std::cout << "  Interpreter conditional branch..." << std::endl;
    test_interpreter_conditional_branch();

    std::cout << "  LLVM-knowledge: large constant costs more..." << std::endl;
    test_large_constant_costs_more();

    std::cout << "  LLVM-knowledge: bitcast is free..." << std::endl;
    test_bitcast_is_free();

    std::cout << "\n=== Results: " << g_pass << " passed, " << g_fail << " failed ===" << std::endl;
    return g_fail > 0 ? 1 : 0;
}
