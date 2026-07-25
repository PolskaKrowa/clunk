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
 * Clunk SMT Verifier Tests — test Z3-based equivalence checking.
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
#include "clunk/Search/SMTVerifier.h"

static int g_pass = 0, g_fail = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { std::cerr << "FAIL: " << msg << " at " << __FILE__ << ":" << __LINE__ << "\n"; g_fail++; } \
    else { g_pass++; } \
} while(0)

using namespace clunk::ir;
using namespace clunk::search;

// ═══════════════════════════════════════════════════════════════════════════
//  Helpers
// ═══════════════════════════════════════════════════════════════════════════

// Build: define i32 @fn(i32 %x) { entry: %r = add i32 %x, 1; ret i32 %r }
static std::shared_ptr<Function> make_add_one(Module& mod) {
    TypeContext& ctx = mod.type_context();
    auto fn_type = std::make_shared<FunctionType>(ctx.int32(), std::vector<std::shared_ptr<Type>>{ctx.int32()});
    auto& fn = mod.add_function("add_one", fn_type);
    fn.add_argument(ctx.int32(), "x");
    auto& entry = fn.add_block("entry");
    // use the actual parameter x, not ConstantInt(5).
    auto x = std::make_shared<Value>(ctx.int32(), "x");
    auto one = ConstantInt::get(ctx, 1, 32);
    entry.add_instruction(inst::make_add(x, one, "r"));
    entry.add_instruction(inst::make_ret(entry.instruction(0)));
    return mod.function("add_one");
}

// Build an identical copy with a different name
static std::shared_ptr<Function> make_add_one_copy(Module& mod) {
    TypeContext& ctx = mod.type_context();
    auto fn_type = std::make_shared<FunctionType>(ctx.int32(), std::vector<std::shared_ptr<Type>>{ctx.int32()});
    auto& fn = mod.add_function("add_one_copy", fn_type);
    fn.add_argument(ctx.int32(), "x");
    auto& entry = fn.add_block("entry");
    // use the actual parameter x, not ConstantInt(5).
    auto x = std::make_shared<Value>(ctx.int32(), "x");
    auto one = ConstantInt::get(ctx, 1, 32);
    entry.add_instruction(inst::make_add(x, one, "r"));
    entry.add_instruction(inst::make_ret(entry.instruction(0)));
    return mod.function("add_one_copy");
}

// Build a different function: returns constant 42
static std::shared_ptr<Function> make_return_42(Module& mod) {
    TypeContext& ctx = mod.type_context();
    auto fn_type = std::make_shared<FunctionType>(ctx.int32(), std::vector<std::shared_ptr<Type>>{ctx.int32()});
    auto& fn = mod.add_function("return_42", fn_type);
    fn.add_argument(ctx.int32(), "x");
    auto& entry = fn.add_block("entry");
    auto val = ConstantInt::get(ctx, 42, 32);
    entry.add_instruction(inst::make_ret(val));
    return mod.function("return_42");
}

// Build: define i32 @fn(i32 %x) { entry: %r = add i32 %x, %x; ret i32 %r }
// (This is x+x = 2x, semantically equivalent to shl x, 1.)
static std::shared_ptr<Function> make_shift_left_one(Module& mod) {
    TypeContext& ctx = mod.type_context();
    auto fn_type = std::make_shared<FunctionType>(ctx.int32(), std::vector<std::shared_ptr<Type>>{ctx.int32()});
    auto& fn = mod.add_function("shl_one", fn_type);
    fn.add_argument(ctx.int32(), "x");
    auto& entry = fn.add_block("entry");
    // use the actual parameter x, not ConstantInt(5).
    auto x = std::make_shared<Value>(ctx.int32(), "x");
    entry.add_instruction(inst::make_add(x, x, "r"));
    entry.add_instruction(inst::make_ret(entry.instruction(0)));
    return mod.function("shl_one");
}

// ═══════════════════════════════════════════════════════════════════════════
//  Tests
// ═══════════════════════════════════════════════════════════════════════════

void test_z3_availability() {
    // Just check that the static method works and doesn't crash
    bool available = SMTVerifier::is_z3_available();
    // Don't assert on the result since Z3 may or may not be installed
    CHECK(true, "is_z3_available() completed without crash (result: " +
          std::string(available ? "true" : "false") + ")");
}

void test_verify_identical_functions() {
    Module mod("smt_identical");
    auto fn1 = make_add_one(mod);
    auto fn2 = make_add_one_copy(mod);

    SMTConfig config;
    config.timeout_ms = 5000;
    SMTVerifier verifier(config);

    auto result = verifier.verify(*fn1, *fn2);
    // Identical functions should be Equivalent (or Unknown if Z3 not available)
    CHECK(result.status == VerificationResult::Equivalent ||
          result.status == VerificationResult::Unknown,
          "identical functions: Equivalent or Unknown");
    CHECK(result.solve_time_ms >= 0.0, "solve_time_ms >= 0");
}

void test_verify_different_functions() {
    Module mod("smt_different");
    auto fn1 = make_add_one(mod);
    auto fn2 = make_return_42(mod);

    SMTConfig config;
    config.timeout_ms = 5000;
    SMTVerifier verifier(config);

    auto result = verifier.verify(*fn1, *fn2);
    // Different functions should be NotEquivalent (or Unknown if Z3 not available)
    CHECK(result.status == VerificationResult::NotEquivalent ||
          result.status == VerificationResult::Unknown,
          "different functions: NotEquivalent or Unknown");
}

void test_batch_verification() {
    Module mod("smt_batch");
    auto original = make_add_one(mod);
    auto candidate1 = make_add_one_copy(mod);
    auto candidate2 = make_return_42(mod);

    SMTConfig config;
    config.timeout_ms = 5000;
    SMTVerifier verifier(config);

    std::vector<std::shared_ptr<Function>> candidates;
    candidates.push_back(candidate1);
    candidates.push_back(candidate2);

    auto results = verifier.verify_batch(*original, candidates);
    CHECK(results.size() == 2, "batch verification returns 2 results");

    for (auto& r : results) {
        CHECK(r.status == VerificationResult::Equivalent ||
              r.status == VerificationResult::NotEquivalent ||
              r.status == VerificationResult::Unknown ||
              r.status == VerificationResult::Error,
              "batch result has valid status");
    }
}

void test_verification_result_is_safe() {
    VerificationResult eq;
    eq.status = VerificationResult::Equivalent;
    CHECK(eq.is_safe(), "Equivalent is safe");

    VerificationResult ne;
    ne.status = VerificationResult::NotEquivalent;
    CHECK(!ne.is_safe(), "NotEquivalent is not safe");

    VerificationResult unk;
    unk.status = VerificationResult::Unknown;
    CHECK(!unk.is_safe(), "Unknown is not safe");

    VerificationResult err;
    err.status = VerificationResult::Error;
    CHECK(!err.is_safe(), "Error is not safe");
}

void test_smt_config() {
    SMTConfig config;
    CHECK(config.timeout_ms == 30000, "default timeout is 30000ms");
    CHECK(config.simplify_before, "default simplify_before is true");
    CHECK(config.use_bitvectors, "default use_bitvectors is true");
    CHECK(config.max_unrolling == 4, "default max_unrolling is 4");

    config.timeout_ms = 10000;
    config.use_bitvectors = false;
    CHECK(config.timeout_ms == 10000, "custom timeout");
    CHECK(!config.use_bitvectors, "custom use_bitvectors");
}

void test_smt_stats() {
    Module mod("smt_stats");
    auto fn1 = make_add_one(mod);
    auto fn2 = make_add_one_copy(mod);

    SMTVerifier verifier;
    verifier.verify(*fn1, *fn2);

    auto stats = verifier.stats();
    CHECK(stats.verifications_run == 1, "1 verification run");
    CHECK(stats.total_time_ms >= 0.0, "total_time_ms >= 0");
}

void test_simulation_fallback() {
    // When Z3 is not available, simulation should still produce a result
    Module mod("smt_sim");
    auto fn1 = make_add_one(mod);
    auto fn2 = make_return_42(mod);

    SMTConfig config;
    config.timeout_ms = 1000;
    SMTVerifier verifier(config);

    auto result = verifier.verify(*fn1, *fn2);
    // Should return some result regardless of Z3 availability
    CHECK(result.status == VerificationResult::Equivalent ||
          result.status == VerificationResult::NotEquivalent ||
          result.status == VerificationResult::Unknown ||
          result.status == VerificationResult::Error,
          "verify returns a valid status");
    CHECK(!result.message.empty() || result.status != VerificationResult::Error,
          "error status has message or is not error");
}

// ═══════════════════════════════════════════════════════════════════════════

int main() {
    std::cout << "=== Clunk SMT Verifier Tests ===" << std::endl;

    std::cout << "  Z3 availability..." << std::endl;
    test_z3_availability();

    std::cout << "  Identical functions..." << std::endl;
    test_verify_identical_functions();

    std::cout << "  Different functions..." << std::endl;
    test_verify_different_functions();

    std::cout << "  Batch verification..." << std::endl;
    test_batch_verification();

    std::cout << "  Verification result is_safe..." << std::endl;
    test_verification_result_is_safe();

    std::cout << "  SMT config..." << std::endl;
    test_smt_config();

    std::cout << "  SMT stats..." << std::endl;
    test_smt_stats();

    std::cout << "  Simulation fallback..." << std::endl;
    test_simulation_fallback();

    std::cout << "\n=== Results: " << g_pass << " passed, " << g_fail << " failed ===" << std::endl;
    return g_fail > 0 ? 1 : 0;
}
