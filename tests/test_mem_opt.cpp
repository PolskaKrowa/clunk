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
 * Clunk MemOpt + Inliner Tests — alias oracle, store-to-load forwarding,
 * redundant-load elimination, dead-store elimination, and single-block
 * call-site inlining.
 */
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "clunk/IR/Function.h"
#include "clunk/IR/Module.h"
#include "clunk/Parser/IRParser.h"
#include "clunk/Evaluator/Interpreter.h"
#include "clunk/Search/Inliner.h"
#include "clunk/Search/MemOpt.h"
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

static std::shared_ptr<Module> parse(const char* src) {
    parser::IRParser p;
    return p.parse_string(src);
}

static size_t count_opcode(const Function& fn, Opcode op) {
    size_t n = 0;
    for (auto& bb : fn.blocks()) {
        for (auto& inst : bb->instructions()) {
            if (inst && inst->opcode() == op) ++n;
        }
    }
    return n;
}

// ═══════════════════════════════════════════════════════════════════════════
//  Alias oracle
// ═══════════════════════════════════════════════════════════════════════════

static void test_alias_oracle() {
    auto mod = parse(R"(
define i32 @f(ptr %p, ptr noalias %q, ptr noalias %r) {
entry:
  %a = alloca i32
  %b = alloca i32
  %g0 = getelementptr i32, ptr %q, i64 0
  %g1 = getelementptr i32, ptr %q, i64 1
  %g1b = getelementptr i32, ptr %q, i64 1
  ret i32 0
}
)");
    auto fn = mod->function("f");
    CHECK(fn != nullptr, "alias fixture parses");
    if (!fn) return;
    AliasOracle oracle(*fn);

    // Look up values by walking the block.
    auto val = [&](const std::string& name) -> std::shared_ptr<Value> {
        for (auto& inst : fn->entry_block()->instructions()) {
            if (inst->has_name() && inst->name() == name) return inst;
        }
        return std::make_shared<Value>(
            std::make_shared<PointerType>(std::make_shared<IntegerType>(32)),
            name);
    };

    CHECK(oracle.alias(val("a"), val("b")) == AliasResult::NoAlias,
          "distinct allocas: NoAlias");
    CHECK(oracle.alias(val("a"), val("a")) == AliasResult::MustAlias,
          "same alloca: MustAlias");
    CHECK(oracle.alias(val("a"), val("p")) == AliasResult::NoAlias,
          "alloca vs argument: NoAlias");
    CHECK(oracle.alias(val("q"), val("r")) == AliasResult::NoAlias,
          "two noalias args: NoAlias");
    CHECK(oracle.alias(val("p"), val("q")) == AliasResult::MayAlias,
          "plain arg vs noalias arg: MayAlias (only PAIRS of noalias are disjoint)");
    CHECK(oracle.alias(val("g0"), val("g1")) == AliasResult::NoAlias,
          "distinct constant GEP elements: NoAlias");
    CHECK(oracle.alias(val("g1"), val("g1b")) == AliasResult::MustAlias,
          "identical constant GEPs: MustAlias");
}

// ═══════════════════════════════════════════════════════════════════════════
//  Memory optimisations
// ═══════════════════════════════════════════════════════════════════════════

static void test_store_to_load_forwarding() {
    auto mod = parse(R"(
define i32 @fwd(i32 %v) {
entry:
  %a = alloca i32
  store i32 %v, ptr %a
  %x = load i32, ptr %a
  %r = add i32 %x, 1
  ret i32 %r
}
)");
    auto fn = mod->function("fwd");
    MemOptimizer mopt;
    auto opt = mopt.optimize(*fn);
    CHECK(opt != nullptr, "forwarding fires");
    if (!opt) return;
    CHECK(mopt.stats().loads_forwarded == 1, "one load forwarded");
    CHECK(count_opcode(*opt, Opcode::Load) == 0, "load is gone");
    for (int64_t v : {0, 42, -7}) {
        auto r0 = evaluator::Interpreter::interpret(*fn, {v});
        auto r1 = evaluator::Interpreter::interpret(*opt, {v});
        CHECK(r0 && r1 && *r0 == *r1, "forwarding preserves semantics");
    }
}

static void test_redundant_load_elimination() {
    auto mod = parse(R"(
define i32 @rle(ptr %p) {
entry:
  %x = load i32, ptr %p
  %y = load i32, ptr %p
  %r = add i32 %x, %y
  ret i32 %r
}
)");
    MemOptimizer mopt;
    auto opt = mopt.optimize(*mod->function("rle"));
    CHECK(opt != nullptr, "RLE fires");
    if (opt) {
        CHECK(mopt.stats().loads_eliminated == 1, "one redundant load eliminated");
        CHECK(count_opcode(*opt, Opcode::Load) == 1, "one load remains");
    }
}

static void test_dead_store_elimination() {
    auto mod = parse(R"(
define i32 @dse(i32 %v, i32 %w) {
entry:
  %a = alloca i32
  store i32 %v, ptr %a
  store i32 %w, ptr %a
  %x = load i32, ptr %a
  ret i32 %x
}
)");
    auto fn = mod->function("dse");
    MemOptimizer mopt;
    auto opt = mopt.optimize(*fn);
    CHECK(opt != nullptr, "DSE fires");
    if (!opt) return;
    CHECK(mopt.stats().stores_eliminated == 1, "first store eliminated");
    auto r = evaluator::Interpreter::interpret(*opt, {1, 2});
    CHECK(r && *r == 2, "DSE keeps the LAST stored value");
}

static void test_clobber_blocks_forwarding() {
    // A call between store and load clobbers everything.
    auto mod = parse(R"(
define i32 @clob(i32 %v) {
entry:
  %a = alloca i32
  store i32 %v, ptr %a
  %z = call i32 @opaque(i32 %v)
  %x = load i32, ptr %a
  %r = add i32 %x, %z
  ret i32 %r
}
)");
    MemOptimizer mopt;
    auto opt = mopt.optimize(*mod->function("clob"));
    CHECK(opt == nullptr, "call clobbers: nothing forwarded");

    // A may-aliasing store between store p / load p blocks forwarding.
    auto mod2 = parse(R"(
define i32 @mayal(ptr %p, ptr %q, i32 %v, i32 %w) {
entry:
  store i32 %v, ptr %p
  store i32 %w, ptr %q
  %x = load i32, ptr %p
  ret i32 %x
}
)");
    MemOptimizer mopt2;
    auto opt2 = mopt2.optimize(*mod2->function("mayal"));
    CHECK(opt2 == nullptr, "may-alias store blocks forwarding");
}

// ═══════════════════════════════════════════════════════════════════════════
//  Inliner
// ═══════════════════════════════════════════════════════════════════════════

static const char* kCallerModule = R"(
define i32 @square(i32 %x) {
entry:
  %r = mul i32 %x, %x
  ret i32 %r
}

define i32 @sum_sq(i32 %a, i32 %b) {
entry:
  %sa = call i32 @square(i32 %a)
  %sb = call i32 @square(i32 %b)
  %r = add i32 %sa, %sb
  ret i32 %r
}

define i32 @looper(i32 %n) {
entry:
  br label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ %inext, %loop ]
  %inext = add i32 %i, 1
  %c = icmp slt i32 %inext, %n
  br i1 %c, label %loop, label %exit
exit:
  ret i32 %inext
}

define i32 @calls_looper(i32 %n) {
entry:
  %r = call i32 @looper(i32 %n)
  ret i32 %r
}
)";

static void test_inliner() {
    auto mod = parse(kCallerModule);
    auto caller = mod->function("sum_sq");
    CHECK(caller != nullptr, "caller parses");
    if (!caller) return;

    Inliner inliner;
    auto inlined = inliner.inline_calls(*caller, *mod);
    CHECK(inlined != nullptr, "inliner fires on sum_sq");
    if (!inlined) return;
    CHECK(inliner.stats().call_sites_inlined == 2, "both call sites inlined");
    CHECK(count_opcode(*inlined, Opcode::Call) == 0, "no calls remain");
    CHECK(count_opcode(*inlined, Opcode::Mul) == 2, "both squares present");

    for (int64_t a : {0, 3, -4}) {
        auto r = evaluator::Interpreter::interpret(*inlined, {a, 5});
        CHECK(r && *r == a * a + 25, "inlined body computes a*a + b*b");
    }

    // The compound payoff: the inlined caller is SMT-verifiable; the
    // original (with opaque calls) is not.
    if (SMTVerifier::is_z3_available()) {
        SMTVerifier verifier;
        auto ref = parse(R"(
define i32 @sum_sq(i32 %a, i32 %b) {
entry:
  %sa = mul i32 %a, %a
  %sb = mul i32 %b, %b
  %r = add i32 %sa, %sb
  ret i32 %r
}
)");
        auto res = verifier.verify(*inlined, *ref->function("sum_sq"));
        CHECK(res.status == VerificationResult::Equivalent,
              "inlined caller is provably a*a + b*b");
        auto res2 = verifier.verify(*caller, *ref->function("sum_sq"));
        CHECK(res2.status == VerificationResult::Unknown,
              "pre-inlining caller (opaque calls) stays Unknown");
    }

    // Multi-block callee: refused.
    Inliner inliner2;
    CHECK(inliner2.inline_calls(*mod->function("calls_looper"), *mod) == nullptr,
          "multi-block callee not inlined");
}

static void test_pipeline_mem_and_inline() {
    auto mod = parse(kCallerModule);

    PipelineConfig cfg;
    cfg.opt_level = 2;
    cfg.time_budget = 12.0;
    cfg.max_time_per_function = 5.0;
    cfg.max_rounds = 2;
    cfg.num_threads = 1;

    Pipeline pipeline(cfg);
    auto result = pipeline.run(*mod);
    auto it = result.function_results.find("sum_sq");
    CHECK(it != result.function_results.end(), "pipeline processed sum_sq");
    if (it == result.function_results.end()) return;
    CHECK(it->second.improvement_ratio > 1.0, "inlining improved sum_sq");
    CHECK(count_opcode(*it->second.optimised, Opcode::Call) == 0,
          "optimised sum_sq has no calls");
}

int main() {
    std::cerr << "test_mem_opt: alias oracle + memory opts + inliner\n";
    test_alias_oracle();
    test_store_to_load_forwarding();
    test_redundant_load_elimination();
    test_dead_store_elimination();
    test_clobber_blocks_forwarding();
    test_inliner();
    test_pipeline_mem_and_inline();
    std::cerr << "passed " << g_pass << ", failed " << g_fail << "\n";
    return g_fail == 0 ? 0 : 1;
}
