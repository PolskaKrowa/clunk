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
 * Clunk LoopOpt Tests — natural-loop analysis, LICM, constant-trip
 * unrolling, and their pipeline integration.
 */
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "clunk/IR/Function.h"
#include "clunk/IR/LoopAnalysis.h"
#include "clunk/IR/Module.h"
#include "clunk/Parser/IRParser.h"
#include "clunk/Evaluator/Interpreter.h"
#include "clunk/Search/LoopOpt.h"
#include "clunk/Search/SMTVerifier.h"
#include "clunk/Pipeline.h"

static int g_pass = 0, g_fail = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { std::cerr << "FAIL: " << msg << " at " << __FILE__ << ":" << __LINE__ << "\n"; g_fail++; } \
    else { g_pass++; } \
} while(0)

using namespace clunk;
using namespace clunk::ir;
using namespace clunk::search;

// sum 0..n-1 — a loop whose trip count depends on an argument (must NOT
// unroll), with a loop-invariant multiply (MUST hoist).
static const char* kSumLoop = R"(
define i32 @sum_licm(i32 %n, i32 %a, i32 %b) {
entry:
  br label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ %inext, %loop ]
  %acc = phi i32 [ 0, %entry ], [ %accnext, %loop ]
  %inv = mul i32 %a, %b
  %term = add i32 %inv, %i
  %accnext = add i32 %acc, %term
  %inext = add i32 %i, 1
  %c = icmp slt i32 %inext, %n
  br i1 %c, label %loop, label %exit
exit:
  ret i32 %accnext
}
)";

// Constant trip count 4, argument-dependent body — must fully unroll.
static const char* kAcc4 = R"(
define i32 @acc4(i32 %x) {
entry:
  br label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ %inext, %loop ]
  %acc = phi i32 [ 0, %entry ], [ %accnext, %loop ]
  %accnext = add i32 %acc, %x
  %inext = add i32 %i, 1
  %c = icmp slt i32 %inext, 4
  br i1 %c, label %loop, label %exit
exit:
  ret i32 %accnext
}
)";

static std::shared_ptr<Module> parse(const char* src) {
    parser::IRParser p;
    return p.parse_string(src);
}

static void test_loop_analysis() {
    auto mod = parse(kSumLoop);
    auto fn = mod->function("sum_licm");
    CHECK(fn != nullptr, "sum_licm parses");
    if (!fn) return;

    CHECK(has_back_edge(*fn), "sum_licm has a back-edge");
    auto loops = find_natural_loops(*fn);
    CHECK(loops.size() == 1, "one natural loop found");
    if (loops.size() != 1) return;
    CHECK(loops[0].header == "loop", "header identified");
    CHECK(loops[0].is_single_block(), "single-block loop");
    CHECK(loops[0].preheader == "entry", "preheader identified");
    CHECK(loops[0].latches.size() == 1 && loops[0].latches[0] == "loop",
          "latch identified");

    auto mod2 = parse("define i32 @s(i32 %x) {\nentry:\n  %r = add i32 %x, 1\n  ret i32 %r\n}\n");
    CHECK(!has_back_edge(*mod2->function("s")), "straight-line fn has no back-edge");
    CHECK(find_natural_loops(*mod2->function("s")).empty(), "no loops in straight-line fn");
}

static void test_licm() {
    auto mod = parse(kSumLoop);
    auto fn = mod->function("sum_licm");
    LoopOptimizer lopt;
    auto hoisted = lopt.hoist_invariants(*fn);
    CHECK(hoisted != nullptr, "LICM fires on sum_licm");
    if (!hoisted) return;
    CHECK(lopt.stats().instructions_hoisted == 1, "exactly the mul hoists");

    // The mul must now live in the entry block, before its terminator.
    auto entry = hoisted->block("entry");
    bool mul_in_entry = false;
    for (auto& inst : entry->instructions()) {
        if (inst->opcode() == Opcode::Mul) mul_in_entry = true;
    }
    CHECK(mul_in_entry, "invariant mul lives in the preheader");
    auto loop_bb = hoisted->block("loop");
    for (auto& inst : loop_bb->instructions()) {
        CHECK(inst->opcode() != Opcode::Mul, "no mul left in the loop body");
    }

    // Semantics: agree with the original on concrete inputs (the
    // interpreter runs loops).
    for (int64_t n : {0, 1, 3, 7}) {
        auto r0 = evaluator::Interpreter::interpret(*fn, {n, 5, 3});
        auto r1 = evaluator::Interpreter::interpret(*hoisted, {n, 5, 3});
        CHECK(r0.has_value() == r1.has_value() && (!r0 || *r0 == *r1),
              "LICM preserves semantics for n=" + std::to_string(n));
    }
}

static void test_unroll() {
    auto mod = parse(kAcc4);
    auto fn = mod->function("acc4");
    LoopOptimizer lopt;
    auto unrolled = lopt.unroll_constant_loops(*fn);
    CHECK(unrolled != nullptr, "constant-trip loop unrolls");
    if (!unrolled) return;
    CHECK(lopt.stats().loops_unrolled == 1, "one loop unrolled");
    CHECK(lopt.stats().iterations_expanded == 4, "four iterations expanded");
    CHECK(!has_back_edge(*unrolled), "unrolled function is straight-line");

    // acc4(x) == 4*x — check the expansion agrees with the original.
    for (int64_t x : {0, 1, -3, 100}) {
        auto r0 = evaluator::Interpreter::interpret(*fn, {x});
        auto r1 = evaluator::Interpreter::interpret(*unrolled, {x});
        CHECK(r0 && r1 && *r0 == *r1 && *r1 == 4 * x,
              "unroll preserves semantics for x=" + std::to_string(x));
    }

    // The compound payoff: the unrolled function is now SMT-verifiable
    // (the original, with its back-edge, is not).
    if (SMTVerifier::is_z3_available()) {
        auto ref = parse(R"(
define i32 @acc4(i32 %x) {
entry:
  %r = shl i32 %x, 2
  ret i32 %r
}
)");
        SMTVerifier verifier;
        auto res = verifier.verify(*unrolled, *ref->function("acc4"));
        CHECK(res.status == VerificationResult::Equivalent,
              "unrolled acc4 provably equals shl x, 2 (got: " + res.message + ")");
    }
}

static void test_unroll_refusals() {
    // Argument-dependent trip count: must NOT unroll.
    auto mod = parse(kSumLoop);
    LoopOptimizer lopt;
    CHECK(lopt.unroll_constant_loops(*mod->function("sum_licm")) == nullptr,
          "argument-dependent trip count refused");

    // Trip count above the cap: must NOT unroll.
    auto big = parse(R"(
define i32 @big(i32 %x) {
entry:
  br label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ %inext, %loop ]
  %inext = add i32 %i, 1
  %c = icmp slt i32 %inext, 1000
  br i1 %c, label %loop, label %exit
exit:
  ret i32 %inext
}
)");
    CHECK(lopt.unroll_constant_loops(*big->function("big")) == nullptr,
          "trip count above max_trip refused");
}

static void test_pipeline_loop() {
    auto mod = parse(kAcc4);

    PipelineConfig cfg;
    cfg.opt_level = 2;
    cfg.time_budget = 8.0;
    cfg.max_time_per_function = 6.0;
    cfg.max_rounds = 2;
    cfg.num_threads = 1;

    Pipeline pipeline(cfg);
    auto result = pipeline.run(*mod);
    auto it = result.function_results.find("acc4");
    CHECK(it != result.function_results.end(), "pipeline processed acc4");
    if (it == result.function_results.end()) return;
    CHECK(it->second.improvement_ratio > 1.0,
          "pipeline improved acc4 (loop unrolled)");
    CHECK(!has_back_edge(*it->second.optimised),
          "adopted result is straight-line");
}

int main() {
    std::cerr << "test_loop_opt: LICM + constant-trip unrolling\n";
    test_loop_analysis();
    test_licm();
    test_unroll();
    test_unroll_refusals();
    test_pipeline_loop();
    std::cerr << "passed " << g_pass << ", failed " << g_fail << "\n";
    return g_fail == 0 ? 0 : 1;
}
