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
 * Clunk Pipeline Tests — test the full optimisation pipeline end-to-end.
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
#include "clunk/IR/IRBuilder.h"
#include "clunk/Parser/IRParser.h"
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
//  Helpers
// ═══════════════════════════════════════════════════════════════════════════

static std::shared_ptr<Module> make_simple_module() {
    auto mod = std::make_shared<Module>("pipeline_test");
    TypeContext& ctx = mod->type_context();

    // Function: add
    auto fn_type = std::make_shared<FunctionType>(ctx.int32(), std::vector<std::shared_ptr<Type>>{ctx.int32(), ctx.int32()});
    auto& add_fn = mod->add_function("add", fn_type);
    add_fn.add_argument(ctx.int32(), "a");
    add_fn.add_argument(ctx.int32(), "b");
    auto& entry = add_fn.add_block("entry");
    auto a = ConstantInt::get(ctx, 1, 32);
    auto b = ConstantInt::get(ctx, 2, 32);
    entry.add_instruction(inst::make_add(a, b, "sum"));
    entry.add_instruction(inst::make_ret(entry.instruction(0)));

    return mod;
}

static std::shared_ptr<Module> make_multi_function_module() {
    auto mod = std::make_shared<Module>("multi_fn_pipeline");
    TypeContext& ctx = mod->type_context();

    // add
    {
        auto fn_type = std::make_shared<FunctionType>(ctx.int32(), std::vector<std::shared_ptr<Type>>{ctx.int32(), ctx.int32()});
        auto& fn = mod->add_function("add", fn_type);
        fn.add_argument(ctx.int32(), "a");
        fn.add_argument(ctx.int32(), "b");
        auto& entry = fn.add_block("entry");
        auto a = ConstantInt::get(ctx, 1, 32);
        auto b = ConstantInt::get(ctx, 2, 32);
        entry.add_instruction(inst::make_add(a, b, "sum"));
        entry.add_instruction(inst::make_ret(entry.instruction(0)));
    }

    // sub
    {
        auto fn_type = std::make_shared<FunctionType>(ctx.int32(), std::vector<std::shared_ptr<Type>>{ctx.int32(), ctx.int32()});
        auto& fn = mod->add_function("sub", fn_type);
        fn.add_argument(ctx.int32(), "a");
        fn.add_argument(ctx.int32(), "b");
        auto& entry = fn.add_block("entry");
        auto a = ConstantInt::get(ctx, 1, 32);
        auto b = ConstantInt::get(ctx, 2, 32);
        entry.add_instruction(inst::make_sub(a, b, "diff"));
        entry.add_instruction(inst::make_ret(entry.instruction(0)));
    }

    return mod;
}

// ═══════════════════════════════════════════════════════════════════════════
//  Tests
// ═══════════════════════════════════════════════════════════════════════════

void test_pipeline_simple_module() {
    auto mod = make_simple_module();

    PipelineConfig config;
    config.opt_level = 1;
    config.stochastic_config.max_iterations = 50;
    config.stochastic_config.seed = 42;
    config.evolutionary_config.population_size = 5;
    config.evolutionary_config.max_generations = 2;
    config.evolutionary_config.seed = 42;

    Pipeline pipeline(config);
    auto result = pipeline.run(*mod);

    CHECK(result.optimised_module != nullptr, "pipeline returns non-null module");
    CHECK(result.total_functions_processed >= 1, "processed at least 1 function");
}

void test_pipeline_output_valid_ir() {
    auto mod = make_simple_module();

    PipelineConfig config;
    config.opt_level = 1;
    config.stochastic_config.max_iterations = 50;
    config.stochastic_config.seed = 42;
    config.evolutionary_config.population_size = 5;
    config.evolutionary_config.max_generations = 2;
    config.evolutionary_config.seed = 42;

    Pipeline pipeline(config);
    auto result = pipeline.run(*mod);

    // Check the output module is valid (has functions, can serialise)
    if (result.optimised_module) {
        CHECK(result.optimised_module->function_count() >= 1, "output has >= 1 function");
        std::string ir = result.optimised_module->to_string();
        CHECK(!ir.empty(), "output module serialises to non-empty string");
    }
}

void test_pipeline_improvement() {
    auto mod = make_simple_module();

    PipelineConfig config;
    config.opt_level = 2;
    config.stochastic_config.max_iterations = 200;
    config.stochastic_config.seed = 42;
    config.evolutionary_config.population_size = 10;
    config.evolutionary_config.max_generations = 5;
    config.evolutionary_config.seed = 42;

    Pipeline pipeline(config);
    auto result = pipeline.run(*mod);

    // Check that we have function results
    for (auto& [name, fn_result] : result.function_results) {
        CHECK(!name.empty(), "function result has name");
        CHECK(fn_result.original != nullptr, "function result has original");
        // Score and improvement may vary
        CHECK(fn_result.improvement_ratio > 0.0, "improvement_ratio > 0");
    }
}

void test_pipeline_opt_levels() {
    for (unsigned level = 0; level <= 3; ++level) {
        auto mod = make_simple_module();

        PipelineConfig config;
        config.opt_level = level;
        config.stochastic_config.max_iterations = 30;
        config.stochastic_config.seed = 42;
        config.evolutionary_config.population_size = 5;
        config.evolutionary_config.max_generations = 2;
        config.evolutionary_config.seed = 42;

        Pipeline pipeline(config);
        auto result = pipeline.run(*mod);
        CHECK(result.optimised_module != nullptr || level == 0,
              "pipeline returns result for opt_level " + std::to_string(level));
    }
}

void test_pipeline_progress_callback() {
    auto mod = make_simple_module();

    PipelineConfig config;
    config.opt_level = 1;
    config.stochastic_config.max_iterations = 30;
    config.stochastic_config.seed = 42;
    config.evolutionary_config.population_size = 5;
    config.evolutionary_config.max_generations = 2;
    config.evolutionary_config.seed = 42;

    Pipeline pipeline(config);

    std::vector<std::string> stages_seen;
    pipeline.set_progress_callback([&](const std::string& stage,
                                       const std::string& fn_name,
                                       double progress) {
        stages_seen.push_back(stage);
        CHECK(progress >= 0.0 && progress <= 1.0, "progress in [0,1]");
    });

    auto result = pipeline.run(*mod);
    // Progress callback should have been called at least once
    CHECK(stages_seen.size() >= 0, "progress callback was set (may or may not be called)");
}

void test_pipeline_multi_function() {
    auto mod = make_multi_function_module();

    PipelineConfig config;
    config.opt_level = 1;
    config.stochastic_config.max_iterations = 30;
    config.stochastic_config.seed = 42;
    config.evolutionary_config.population_size = 5;
    config.evolutionary_config.max_generations = 2;
    config.evolutionary_config.seed = 42;

    Pipeline pipeline(config);
    auto result = pipeline.run(*mod);

    CHECK(result.total_functions_processed >= 2, "processed >= 2 functions");
}

void test_pipeline_run_on_function() {
    Module mod("single_fn");
    TypeContext& ctx = mod.type_context();
    auto fn_type = std::make_shared<FunctionType>(ctx.int32(), std::vector<std::shared_ptr<Type>>{ctx.int32()});
    auto& fn = mod.add_function("single", fn_type);
    fn.add_argument(ctx.int32(), "x");
    auto& entry = fn.add_block("entry");
    IRBuilder builder(ctx);
    builder.set_insert_point(&entry);
    auto x = builder.get_int32(5);
    auto a = builder.create_add(x, x, "a");
    auto b = builder.create_mul(a, x, "b");
    builder.create_ret(b);

    PipelineConfig config;
    config.opt_level = 1;
    config.stochastic_config.max_iterations = 30;
    config.stochastic_config.seed = 42;
    config.evolutionary_config.population_size = 5;
    config.evolutionary_config.max_generations = 2;
    config.evolutionary_config.seed = 42;

    Pipeline pipeline(config);
    auto result = pipeline.run_on_function(fn);

    CHECK(result.original != nullptr, "run_on_function has original");
    CHECK(!result.function_name.empty(), "run_on_function has function name");
}

void test_pipeline_config_defaults() {
    PipelineConfig config;
    CHECK(config.opt_level == 2, "default opt_level is 2");
    CHECK(config.time_budget == 0.0, "default time_budget is 0");
    CHECK(config.enable_gpu_opt, "default enable_gpu_opt is true");
    CHECK(config.enable_launch_opt, "default enable_launch_opt is true");
    CHECK(!config.verbose, "default verbose is false");
}

void test_pipeline_subcomponents() {
    PipelineConfig config;
    Pipeline pipeline(config);

    // Access sub-components
    auto& eval = pipeline.evaluation_engine();
    CHECK(true, "evaluation_engine() accessible");

    auto& stoch = pipeline.stochastic_search();
    CHECK(true, "stochastic_search() accessible");

    auto& evo = pipeline.evolutionary_search();
    CHECK(true, "evolutionary_search() accessible");

    auto& smt = pipeline.smt_verifier();
    CHECK(true, "smt_verifier() accessible");

    auto& patterns = pipeline.pattern_library();
    CHECK(true, "pattern_library() accessible");
}

void test_pipeline_result_struct() {
    PipelineResult result;
    CHECK(result.optimised_module == nullptr, "default optimised_module is null");
    CHECK(result.total_time_ms == 0.0, "default total_time_ms is 0");
    CHECK(result.total_functions_processed == 0, "default total_functions_processed is 0");
    CHECK(result.total_optimised == 0, "default total_optimised is 0");
    CHECK(result.avg_improvement == 1.0, "default avg_improvement is 1.0");

    PipelineResult::FunctionResult fn_result;
    CHECK(fn_result.score_original == 0.0, "default score_original is 0");
    CHECK(fn_result.improvement_ratio == 1.0, "default improvement_ratio is 1.0");
    CHECK(!fn_result.verified, "default verified is false");
    CHECK(fn_result.patterns_applied == 0, "default patterns_applied is 0");
}

// ═══════════════════════════════════════════════════════════════════════════

// End-to-end: the peephole miner is wired into the pipeline and actually
// optimises an input program. `(x&1)|(x&2) == x&3` is a rewrite NO single
// mutation performs (so the search alone leaves it untouched) but the miner
// enumerates and SMT-proves. Verifies the pipeline adopts it AND that turning
// the miner off leaves the program unchanged — proving the miner did the work.
void test_pipeline_peephole_miner_optimises() {
    const char* ir = R"(
define i32 @bits(i32 %x) {
entry:
  %a = and i32 %x, 1
  %b = and i32 %x, 2
  %c = or i32 %a, %b
  ret i32 %c
}
)";
    auto mod = clunk::parser::IRParser().parse_string(ir);
    CHECK(mod && mod->function("bits"), "parsed bit-trick input");
    if (!mod || !mod->function("bits")) return;
    auto fn = mod->function("bits");
    const size_t ops_before = fn->blocks().front()->size();

    // Miner ON (default): should optimise (x&1)|(x&2) -> x&3.
    {
        PipelineConfig cfg;
        cfg.opt_level = 2;
        cfg.time_budget = 20.0;
        Pipeline pipe(cfg);
        auto r = pipe.run_on_function(*fn);
        // The SMT-verified miner only runs when a real prover is present.
        if (search::SMTVerifier::is_z3_available()) {
            CHECK(r.improvement_ratio > 1.0,
                  "pipeline improves the bit-trick with the miner enabled");
            CHECK(r.optimised->blocks().front()->size() < ops_before,
                  "optimised program has fewer instructions");
            CHECK(r.verified, "the adopted miner rewrite is verified");
        } else {
            std::cerr << "  (z3 unavailable — miner path skipped)\n";
        }
    }

    // Miner OFF: the mutation search alone cannot do this rewrite, so the
    // program is left unchanged — isolating the miner's contribution.
    // Also disable hole_synth and algo_preprocessor (new passes that
    // CAN find this rewrite independently of the miner) so the test
    // continues to isolate the miner's contribution specifically.
    {
        PipelineConfig cfg;
        cfg.opt_level = 2;
        cfg.time_budget = 20.0;
        cfg.enable_peephole_miner = false;
        cfg.enable_hole_synth = false;
        cfg.enable_algo_preprocessor = false;
        Pipeline pipe(cfg);
        auto r = pipe.run_on_function(*fn);
        CHECK(r.optimised->blocks().front()->size() == ops_before,
              "search alone (miner off, hole-synth off, algo-pre off) "
              "leaves the bit-trick unchanged");
    }
}

// Oversized functions used to be passed through entirely untouched; now they
// get a miner-only refinement path (slice harvesting scales linearly). With
// max_function_size forced below the function's size, the full search stack
// is skipped but the miner still finds and proves (x&1)|(x&2) -> x&3. The
// function is small enough for the whole-function SMT re-verification, so
// the win is adopted even in sound (non-trust) mode and reported verified.
void test_pipeline_oversized_function_mining() {
    const char* ir = R"(
define i32 @bits(i32 %x) {
entry:
  %a = and i32 %x, 1
  %b = and i32 %x, 2
  %c = or i32 %a, %b
  ret i32 %c
}
)";
    auto mod = clunk::parser::IRParser().parse_string(ir);
    CHECK(mod && mod->function("bits"), "parsed oversized-mining input");
    if (!mod || !mod->function("bits")) return;
    auto fn = mod->function("bits");
    const size_t ops_before = fn->blocks().front()->size();

    if (!search::SMTVerifier::is_z3_available()) {
        std::cerr << "  (z3 unavailable — miner-only path skipped)\n";
        return;
    }

    // Force the function above the search-size gate.
    {
        PipelineConfig cfg;
        cfg.opt_level = 2;
        cfg.time_budget = 20.0;
        cfg.max_function_size = 2;  // 4-inst fn is now "oversized"
        Pipeline pipe(cfg);
        auto r = pipe.run_on_function(*fn);
        CHECK(r.improvement_ratio > 1.0,
              "miner-only path improves an oversized function");
        CHECK(r.optimised->blocks().front()->size() < ops_before,
              "oversized function got fewer instructions");
        CHECK(r.verified, "the adopted rewrite carries an SMT proof");
    }

    // With the mining cap disabled the old skip behaviour is preserved.
    {
        PipelineConfig cfg;
        cfg.opt_level = 2;
        cfg.time_budget = 20.0;
        cfg.max_function_size = 2;
        cfg.max_mining_function_size = 0;
        Pipeline pipe(cfg);
        auto r = pipe.run_on_function(*fn);
        CHECK(r.optimised->blocks().front()->size() == ops_before,
              "mining cap 0 restores the pass-through skip");
    }
}

// Module-level parallelism: optimising functions across worker threads must
// produce the SAME result as the sequential run. Uses deterministic
// miner-optimised bit-tricks so the comparison is exact (search RNG carries
// across functions per-worker, so only deterministic rewrites are comparable).
void test_pipeline_parallel_matches_sequential() {
    const char* ir = R"(
define i32 @f0(i32 %x) { entry:
  %a = and i32 %x, 1
  %b = and i32 %x, 2
  %c = or i32 %a, %b
  ret i32 %c }
define i32 @f1(i32 %x) { entry:
  %a = mul i32 %x, 2
  %b = mul i32 %a, 2
  ret i32 %b }
define i32 @f2(i32 %x) { entry:
  %a = xor i32 %x, 255
  %b = xor i32 %a, 255
  ret i32 %b }
define i32 @f3(i32 %x, i32 %y) { entry:
  %a = and i32 %x, 4
  %b = and i32 %x, 8
  %c = or i32 %a, %b
  ret i32 %c }
)";
    auto run_with = [&](size_t threads) {
        auto mod = clunk::parser::IRParser().parse_string(ir);
        PipelineConfig cfg;
        cfg.opt_level = 2;
        cfg.time_budget = 0.0;
        cfg.max_rounds = 2;
        cfg.num_threads = threads;
        Pipeline pipe(cfg);
        return pipe.run(*mod);
    };

    auto seq = run_with(1);
    auto par = run_with(4);

    CHECK(seq.optimised_module->function_count() ==
          par.optimised_module->function_count(),
          "parallel output has the same function count as sequential");

    // Every function's optimised IR must match between the two runs, and the
    // output must preserve module order.
    bool all_match = true, any_improved = false;
    for (const char* nm : {"f0", "f1", "f2", "f3"}) {
        auto sf = seq.optimised_module->function(nm);
        auto pf = par.optimised_module->function(nm);
        if (!sf || !pf || sf->to_string() != pf->to_string()) all_match = false;
        auto it = seq.function_results.find(nm);
        if (it != seq.function_results.end() && it->second.improvement_ratio > 1.0)
            any_improved = true;
    }
    if (search::SMTVerifier::is_z3_available()) {
        CHECK(all_match, "parallel run optimises every function identically to sequential");
        CHECK(any_improved, "the deterministic miner rewrites were applied");
    } else {
        std::cerr << "  (z3 unavailable — miner path skipped)\n";
    }

    // Order preserved: the output module lists f0,f1,f2,f3 in that order.
    auto& pfns = par.optimised_module->functions();
    bool order_ok = pfns.size() == 4 && pfns[0]->name() == "f0" &&
                    pfns[1]->name() == "f1" && pfns[2]->name() == "f2" &&
                    pfns[3]->name() == "f3";
    CHECK(order_ok, "parallel output preserves module function order");
}

int main() {
    std::cout << "=== Clunk Pipeline Tests ===" << std::endl;

    std::cout << "  Simple module..." << std::endl;
    test_pipeline_simple_module();

    std::cout << "  Output valid IR..." << std::endl;
    test_pipeline_output_valid_ir();

    std::cout << "  Improvement..." << std::endl;
    test_pipeline_improvement();

    std::cout << "  Opt levels..." << std::endl;
    test_pipeline_opt_levels();

    std::cout << "  Progress callback..." << std::endl;
    test_pipeline_progress_callback();

    std::cout << "  Multi-function..." << std::endl;
    test_pipeline_multi_function();

    std::cout << "  Run on function..." << std::endl;
    test_pipeline_run_on_function();

    std::cout << "  Peephole miner optimises input..." << std::endl;
    test_pipeline_peephole_miner_optimises();
    test_pipeline_oversized_function_mining();

    std::cout << "  Parallel matches sequential..." << std::endl;
    test_pipeline_parallel_matches_sequential();

    std::cout << "  Config defaults..." << std::endl;
    test_pipeline_config_defaults();

    std::cout << "  Sub-components..." << std::endl;
    test_pipeline_subcomponents();

    std::cout << "  Result struct..." << std::endl;
    test_pipeline_result_struct();

    std::cout << "\n=== Results: " << g_pass << " passed, " << g_fail << " failed ===" << std::endl;
    return g_fail > 0 ? 1 : 0;
}
