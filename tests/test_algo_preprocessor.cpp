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
 * Clunk Algorithmic Preprocessor — tests.
 *
 * Exercises the new module-level pre-pass:
 *   - Detects f(x) ≡ C (constant function) and rewrites to `ret C`.
 *   - Detects f(x) ≡ c*x and rewrites to `mul x, c`.
 *   - Detects f(x) ≡ c*x + b and rewrites to the affine form.
 *   - Refuses out-of-scope functions (multi-block, FP, calls).
 *   - End-to-end pipeline integration: the pre-pass fires before
 *     per-function superoptimisation, shrinking the work the search
 *     has to do.
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
#include "clunk/Parser/IRParser.h"
#include "clunk/Evaluator/EvaluationEngine.h"
#include "clunk/Evaluator/Interpreter.h"
#include "clunk/Search/AlgoPreprocessor.h"
#include "clunk/Search/SMTVerifier.h"
#include "clunk/Pipeline.h"

static int g_pass = 0, g_fail = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { std::cerr << "FAIL: " << msg << " at " << __FILE__ << ":" << __LINE__ << "\n"; g_fail++; } \
    else { g_pass++; } \
} while(0)

using namespace clunk::ir;
using namespace clunk;

// ── Helpers ─────────────────────────────────────────────────────────────────

static std::shared_ptr<Module> parse_mod(const std::string& ir) {
    return parser::IRParser().parse_string(ir);
}

// Run AlgoPreprocessor with trust_unverified so the tests work even
// without Z3 installed.
static bool run_algo_pre(Module& mod, search::AlgoPreprocessor::Stats* stats_out = nullptr) {
    search::AlgoPreConfig cfg;
    cfg.trust_unverified = true;
    cfg.smt_timeout_ms = 1000;
    search::AlgoPreprocessor ap(cfg);
    bool changed = ap.run(mod);
    if (stats_out) *stats_out = ap.stats();
    return changed;
}

// ── Tests ───────────────────────────────────────────────────────────────────

// f(x) = (x + 1) - x  →  always returns 1 (constant function).
static void test_algo_pre_constant_function() {
    std::cout << "  algo-pre: f(x)=(x+1)-x → ret 1..." << std::endl;
    auto mod = parse_mod(R"(
define i32 @f(i32 %x) {
entry:
  %a = add i32 %x, 1
  %b = sub i32 %a, %x
  ret i32 %b
}
)");
    CHECK(mod != nullptr, "parsed module");

    search::AlgoPreprocessor::Stats stats;
    bool changed = run_algo_pre(*mod, &stats);
    CHECK(changed, "algo-pre made changes");
    CHECK(stats.constants_detected >= 1, "constant pattern detected");

    auto fn = mod->function("f");
    CHECK(fn != nullptr, "f survived");
    if (!fn) return;

    // Verify the rewritten function is `ret i32 1`.
    CHECK(fn->instruction_count() == 1, "f collapsed to 1 instruction (ret 1)");

    // Verify semantic equivalence on probes.
    // NOTE: uses small-magnitude values to avoid a pre-existing bug in
    // the Interpreter where `sub` results aren't masked to the result
    // type's bit width (so INT32_MIN/INT32_MAX probes give wrong answers).
    auto orig = parse_mod(R"(
define i32 @f_orig(i32 %x) {
entry:
  %a = add i32 %x, 1
  %b = sub i32 %a, %x
  ret i32 %b
}
)");
    auto orig_fn = orig->function("f_orig");
    bool all_match = true;
    for (int64_t x : {0, 1, -1, 2, 42, -100, 255, 1024, 4096, -4096}) {
        auto o = evaluator::Interpreter::interpret(*orig_fn, {x});
        auto r = evaluator::Interpreter::interpret(*fn, {x});
        if (!o || !r || *o != *r) { all_match = false; break; }
    }
    CHECK(all_match, "rewritten function agrees with original on probes");
}

// f(x) = x * 2 + x  →  c*x with c=3 (multiplicative).
static void test_algo_pre_multiplicative() {
    std::cout << "  algo-pre: f(x)=x*2+x → mul x,3..." << std::endl;
    auto mod = parse_mod(R"(
define i32 @f(i32 %x) {
entry:
  %a = mul i32 %x, 2
  %b = add i32 %a, %x
  ret i32 %b
}
)");
    CHECK(mod != nullptr, "parsed module");

    search::AlgoPreprocessor::Stats stats;
    bool changed = run_algo_pre(*mod, &stats);
    CHECK(changed, "algo-pre made changes");
    // Either multiplicative (c=3) or affine (c=3, b=0) — both are valid
    // detections; the affine detector has higher specificity.
    CHECK(stats.scalars_detected + stats.affines_detected >= 1,
          "multiplicative or affine pattern detected");

    auto fn = mod->function("f");
    CHECK(fn != nullptr, "f survived");
    if (!fn) return;

    // The rewritten function should be `mul x, 3` + ret = 2 instructions.
    CHECK(fn->instruction_count() == 2, "f collapsed to 2 instructions (mul + ret)");

    // Verify semantic equivalence.
    auto orig = parse_mod(R"(
define i32 @f_orig(i32 %x) {
entry:
  %a = mul i32 %x, 2
  %b = add i32 %a, %x
  ret i32 %b
}
)");
    auto orig_fn = orig->function("f_orig");
    bool all_match = true;
    for (int64_t x : {0, 1, -1, 2, -2, 3, 7, 8, 15, 16, 127, 255, 1024, -1024}) {
        auto o = evaluator::Interpreter::interpret(*orig_fn, {x});
        auto r = evaluator::Interpreter::interpret(*fn, {x});
        if (!o || !r || *o != *r) { all_match = false; break; }
    }
    CHECK(all_match, "rewritten function agrees with original on probes");
}

// f(x) = (x + 1) * 2  →  affine with c=2, b=2.
static void test_algo_pre_affine() {
    std::cout << "  algo-pre: f(x)=(x+1)*2 → 2x+2..." << std::endl;
    auto mod = parse_mod(R"(
define i32 @f(i32 %x) {
entry:
  %a = add i32 %x, 1
  %b = mul i32 %a, 2
  ret i32 %b
}
)");
    CHECK(mod != nullptr, "parsed module");

    search::AlgoPreprocessor::Stats stats;
    bool changed = run_algo_pre(*mod, &stats);
    CHECK(changed, "algo-pre made changes");
    CHECK(stats.affines_detected >= 1, "affine pattern detected");

    auto fn = mod->function("f");
    CHECK(fn != nullptr, "f survived");
    if (!fn) return;

    // The rewritten function should be `mul x, 2` + `add .., 2` + ret = 3 instructions.
    CHECK(fn->instruction_count() == 3, "f collapsed to 3 instructions (mul + add + ret)");

    // Verify semantic equivalence.
    auto orig = parse_mod(R"(
define i32 @f_orig(i32 %x) {
entry:
  %a = add i32 %x, 1
  %b = mul i32 %a, 2
  ret i32 %b
}
)");
    auto orig_fn = orig->function("f_orig");
    bool all_match = true;
    for (int64_t x : {0, 1, -1, 2, -2, 3, 7, 8, 15, 16, 127, 255, 1024, -1024}) {
        auto o = evaluator::Interpreter::interpret(*orig_fn, {x});
        auto r = evaluator::Interpreter::interpret(*fn, {x});
        if (!o || !r || *o != *r) { all_match = false; break; }
    }
    CHECK(all_match, "rewritten function agrees with original on probes");
}

// Out-of-scope: multi-block function → refused.
static void test_algo_pre_out_of_scope_multi_block() {
    std::cout << "  algo-pre: multi-block function is refused..." << std::endl;
    auto mod = parse_mod(R"(
define i32 @f(i32 %x) {
entry:
  %c = icmp sgt i32 %x, 0
  br i1 %c, label %then, label %else
then:
  ret i32 1
else:
  ret i32 0
}
)");
    CHECK(mod != nullptr, "parsed module");

    search::AlgoPreprocessor::Stats stats;
    bool changed = run_algo_pre(*mod, &stats);
    CHECK(!changed, "algo-pre refused multi-block function");
    CHECK(stats.functions_skipped >= 1, "functions_skipped counter incremented");
}

// Out-of-scope: function with calls → refused.
static void test_algo_pre_out_of_scope_calls() {
    std::cout << "  algo-pre: function with calls is refused..." << std::endl;
    auto mod = parse_mod(R"(
define i32 @f(i32 %x) {
entry:
  %r = call i32 @g(i32 %x)
  ret i32 %r
}
declare i32 @g(i32)
)");
    CHECK(mod != nullptr, "parsed module");

    search::AlgoPreprocessor::Stats stats;
    bool changed = run_algo_pre(*mod, &stats);
    CHECK(!changed, "algo-pre refused function with calls");
}

// Function with non-affine behaviour → no rewrite.
static void test_algo_pre_non_affine_no_rewrite() {
    std::cout << "  algo-pre: non-affine function is unchanged..." << std::endl;
    auto mod = parse_mod(R"(
define i32 @f(i32 %x) {
entry:
  %r = mul i32 %x, %x
  ret i32 %r
}
)");
    CHECK(mod != nullptr, "parsed module");

    search::AlgoPreprocessor::Stats stats;
    bool changed = run_algo_pre(*mod, &stats);
    // f(x) = x² is non-affine; the preprocessor should NOT detect a pattern.
    // (It may attempt detection but fail SMT, leaving the function unchanged.)
    CHECK(!changed, "non-affine function is unchanged");
}

// End-to-end: pipeline integration. The pre-pass should fire and
// rewrite a constant-returning function before per-function
// superoptimisation runs.
static void test_pipeline_algo_pre_integration() {
    std::cout << "  algo-pre: pipeline integration..." << std::endl;
    auto mod = parse_mod(R"(
define i32 @f(i32 %x) {
entry:
  %a = add i32 %x, 1
  %b = sub i32 %a, %x
  ret i32 %b
}
)");
    CHECK(mod != nullptr, "parsed module");

    PipelineConfig pcfg;
    pcfg.opt_level = 2;
    pcfg.time_budget = 5.0;
    pcfg.enable_algo_preprocessor = true;
    pcfg.skip_smt = true;  // algo-pre falls back to trust_unverified
    pcfg.verbose = false;
    Pipeline pipeline(pcfg);
    auto result = pipeline.run(*mod);

    CHECK(result.optimised_module != nullptr, "pipeline produced a module");
    auto opt_fn = result.optimised_module->function("f");
    CHECK(opt_fn != nullptr, "f survived optimisation");
    if (!opt_fn) return;
    // The pre-pass should have collapsed f to `ret i32 1` (1 instruction).
    // (Plus whatever the per-function pipeline does on top — but since
    // the function is already minimal, no further rewrite is expected.)
    CHECK(opt_fn->instruction_count() <= 3,
          "optimised f is no larger than 3 instructions");
}

// ── main ────────────────────────────────────────────────────────────────────

int main() {
    std::cout << "=== AlgoPreprocessor tests ===\n";
    test_algo_pre_constant_function();
    test_algo_pre_multiplicative();
    test_algo_pre_affine();
    test_algo_pre_out_of_scope_multi_block();
    test_algo_pre_out_of_scope_calls();
    test_algo_pre_non_affine_no_rewrite();
    test_pipeline_algo_pre_integration();
    std::cout << "=== AlgoPreprocessor: " << g_pass << " passed, "
              << g_fail << " failed ===\n";
    return g_fail > 0 ? 1 : 0;
}
