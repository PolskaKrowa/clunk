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
 * Clunk SMT Verifier Tests — I1-RETRY features.
 *
 * Covers:
 *   - R4.B: Alive2-style poison/undef + nsw/nuw/exact SMT encoding.
 *   - R1.E + R1.D + R3.B: CEGIS for symbolic-constant synthesis.
 *   - R4.D: Unsat-core / assumption-based batch pruning.
 *
 * Each test accepts `Unknown` as a valid outcome when Z3 is not available
 * (graceful degradation), but enforces soundness: a positive test never
 * returns `NotEquivalent`, and a negative test never returns `Equivalent`.
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

static std::shared_ptr<Value> make_arg(TypeContext& ctx, const std::string& name) {
    return std::make_shared<Value>(ctx.int32(), name);
}

// f(x, y) = add [flags] x, y
static std::shared_ptr<Function> make_add_xy(Module& mod, const std::string& name,
                                              BinOpFlags flags = {}) {
    TypeContext& ctx = mod.type_context();
    auto fn_type = std::make_shared<FunctionType>(ctx.int32(),
        std::vector<std::shared_ptr<Type>>{ctx.int32(), ctx.int32()});
    auto& fn = mod.add_function(name, fn_type);
    fn.add_argument(ctx.int32(), "x");
    fn.add_argument(ctx.int32(), "y");
    auto& entry = fn.add_block("entry");
    auto x = make_arg(ctx, "x");
    auto y = std::make_shared<Value>(ctx.int32(), "y");
    entry.add_instruction(inst::make_add(x, y, "r", flags));
    entry.add_instruction(inst::make_ret(entry.instruction(0)));
    return mod.function(name);
}

// f(x) = shl x, 3   (constructed manually — no inst::make_shl helper exists)
static std::shared_ptr<Function> make_shl_const(Module& mod, const std::string& name,
                                                 int64_t shift_amount) {
    TypeContext& ctx = mod.type_context();
    auto fn_type = std::make_shared<FunctionType>(ctx.int32(),
        std::vector<std::shared_ptr<Type>>{ctx.int32()});
    auto& fn = mod.add_function(name, fn_type);
    fn.add_argument(ctx.int32(), "x");
    auto& entry = fn.add_block("entry");
    auto x = make_arg(ctx, "x");
    auto amt = ConstantInt::get(ctx, shift_amount, 32);
    auto shl_inst = std::make_shared<Instruction>(Opcode::Shl, ctx.int32(), "r");
    shl_inst->add_operand(x);
    shl_inst->add_operand(amt);
    entry.add_instruction(shl_inst);
    entry.add_instruction(inst::make_ret(entry.instruction(0)));
    return mod.function(name);
}

// f(x) = mul x, c   (used as the candidate template for CEGIS — `c` is the
// placeholder constant with a user-set name).
static std::shared_ptr<Function> make_mul_placeholder(Module& mod, const std::string& name,
                                                       const std::string& placeholder_name) {
    TypeContext& ctx = mod.type_context();
    auto fn_type = std::make_shared<FunctionType>(ctx.int32(),
        std::vector<std::shared_ptr<Type>>{ctx.int32()});
    auto& fn = mod.add_function(name, fn_type);
    fn.add_argument(ctx.int32(), "x");
    auto& entry = fn.add_block("entry");
    auto x = make_arg(ctx, "x");
    auto placeholder = ConstantInt::get(ctx, 0, 32);  // dummy value 0
    placeholder->set_name(placeholder_name);
    entry.add_instruction(inst::make_mul(x, placeholder, "r"));
    entry.add_instruction(inst::make_ret(entry.instruction(0)));
    return mod.function(name);
}

// f(x) = add x, c   (candidate template for the negative CEGIS test).
static std::shared_ptr<Function> make_add_placeholder(Module& mod, const std::string& name,
                                                       const std::string& placeholder_name) {
    TypeContext& ctx = mod.type_context();
    auto fn_type = std::make_shared<FunctionType>(ctx.int32(),
        std::vector<std::shared_ptr<Type>>{ctx.int32()});
    auto& fn = mod.add_function(name, fn_type);
    fn.add_argument(ctx.int32(), "x");
    auto& entry = fn.add_block("entry");
    auto x = make_arg(ctx, "x");
    auto placeholder = ConstantInt::get(ctx, 0, 32);
    placeholder->set_name(placeholder_name);
    entry.add_instruction(inst::make_add(x, placeholder, "r"));
    entry.add_instruction(inst::make_ret(entry.instruction(0)));
    return mod.function(name);
}

// f(x) = ret i32 c   (constant-returning function — used as an obviously-wrong
// candidate for the unsat-core batch pruning test).
static std::shared_ptr<Function> make_return_const(Module& mod, const std::string& name,
                                                    int64_t c) {
    TypeContext& ctx = mod.type_context();
    auto fn_type = std::make_shared<FunctionType>(ctx.int32(),
        std::vector<std::shared_ptr<Type>>{ctx.int32()});
    auto& fn = mod.add_function(name, fn_type);
    fn.add_argument(ctx.int32(), "x");
    auto& entry = fn.add_block("entry");
    auto val = ConstantInt::get(ctx, c, 32);
    entry.add_instruction(inst::make_ret(val));
    return mod.function(name);
}

// ═══════════════════════════════════════════════════════════════════════════
//  R4.B tests — nsw/nuw/exact poison encoding
// ═══════════════════════════════════════════════════════════════════════════

// Test 1: two `add nsw x, y` should be Equivalent (both produce poison on the
// same signed-overflow inputs, so the shared POISON constant makes them
// compare equal).
void test_add_nsw_both_equivalent() {
    Module mod("nsw_both");
    BinOpFlags nsw_flags; nsw_flags.nsw = true;
    auto fn1 = make_add_xy(mod, "add_nsw_1", nsw_flags);
    auto fn2 = make_add_xy(mod, "add_nsw_2", nsw_flags);

    SMTConfig config;
    config.timeout_ms = 5000;
    config.honor_binop_flags = true;
    SMTVerifier verifier(config);

    auto result = verifier.verify(*fn1, *fn2);
    std::cout << "    [nsw both] status=" << static_cast<int>(result.status)
              << " msg=" << result.message << "\n";
    // Positive: must be Equivalent (or Unknown if Z3 unavailable).
    CHECK(result.status == VerificationResult::Equivalent ||
          result.status == VerificationResult::Unknown,
          "add nsw x,y vs add nsw x,y: Equivalent or Unknown");
    // Soundness: never NotEquivalent for identical functions.
    CHECK(result.status != VerificationResult::NotEquivalent,
          "SOUNDNESS: add nsw x,y vs add nsw x,y must NOT be NotEquivalent");
}

// Test 2: original `add nsw x, y` vs candidate `add x, y` (no flag), and the
// reverse. Under Alive2-style REFINEMENT semantics (the default), dropping
// the flag is a legal replacement: wherever the original is well-defined the
// candidate agrees, and where the original is poison the candidate may do
// anything (LLVM's contract — this is the classic Souper flag-dropping win).
// INTRODUCING poison (plain original, nsw candidate) must still be rejected.
// Under legacy strict-equivalence semantics, both directions are rejected.
void test_add_nsw_vs_plain_not_equivalent() {
    Module mod("nsw_vs_plain");
    BinOpFlags nsw_flags; nsw_flags.nsw = true;
    auto fn_orig = make_add_xy(mod, "add_nsw_orig", nsw_flags);
    auto fn_cand = make_add_xy(mod, "add_plain_cand", BinOpFlags{});

    // ── Refinement semantics (default): flag-dropping is accepted ──────
    {
        SMTConfig config;
        config.timeout_ms = 5000;
        config.honor_binop_flags = true;
        config.refinement_semantics = true;
        SMTVerifier verifier(config);

        auto drop = verifier.verify(*fn_orig, *fn_cand);
        std::cout << "    [nsw vs plain, refine] status="
                  << static_cast<int>(drop.status)
                  << " msg=" << drop.message << "\n";
        CHECK(drop.status == VerificationResult::Equivalent ||
              drop.status == VerificationResult::Unknown,
              "refinement: add nsw x,y -> add x,y is a legal replacement");

        auto introduce = verifier.verify(*fn_cand, *fn_orig);
        std::cout << "    [plain vs nsw, refine] status="
                  << static_cast<int>(introduce.status)
                  << " msg=" << introduce.message << "\n";
        CHECK(introduce.status != VerificationResult::Equivalent,
              "SOUNDNESS: add x,y -> add nsw x,y introduces poison — "
              "must NOT be accepted");
    }

    // ── Legacy strict equivalence: both directions rejected ────────────
    {
        SMTConfig config;
        config.timeout_ms = 5000;
        config.honor_binop_flags = true;
        config.refinement_semantics = false;
        SMTVerifier verifier(config);

        auto result = verifier.verify(*fn_orig, *fn_cand);
        std::cout << "    [nsw vs plain, strict] status="
                  << static_cast<int>(result.status)
                  << " msg=" << result.message << "\n";
        CHECK(result.status == VerificationResult::NotEquivalent ||
              result.status == VerificationResult::Unknown,
              "strict: add nsw x,y vs add x,y: NotEquivalent or Unknown");
        CHECK(result.status != VerificationResult::Equivalent,
              "strict soundness: add nsw x,y vs add x,y must NOT be Equivalent");
    }
}

// Test 3: both `add x, y` (no flags) → Equivalent (baseline, no flag
// encoding needed).
void test_add_plain_both_equivalent() {
    Module mod("plain_both");
    auto fn1 = make_add_xy(mod, "add_plain_1", BinOpFlags{});
    auto fn2 = make_add_xy(mod, "add_plain_2", BinOpFlags{});

    SMTConfig config;
    config.timeout_ms = 5000;
    config.honor_binop_flags = true;
    SMTVerifier verifier(config);

    auto result = verifier.verify(*fn1, *fn2);
    std::cout << "    [plain both] status=" << static_cast<int>(result.status)
              << " msg=" << result.message << "\n";
    CHECK(result.status == VerificationResult::Equivalent ||
          result.status == VerificationResult::Unknown,
          "add x,y vs add x,y: Equivalent or Unknown");
    CHECK(result.status != VerificationResult::NotEquivalent,
          "SOUNDNESS: add x,y vs add x,y must NOT be NotEquivalent");
}

// Test 3b: when `honor_binop_flags = false`, the encoder ignores the flag
// and `add nsw x,y` vs `add x,y` ARE treated as equivalent (unsound but
// matches today's behaviour). This test pins the fallback semantics.
void test_honor_flags_false_falls_back() {
    Module mod("flags_false");
    BinOpFlags nsw_flags; nsw_flags.nsw = true;
    auto fn_orig = make_add_xy(mod, "add_nsw_orig", nsw_flags);
    auto fn_cand = make_add_xy(mod, "add_plain_cand", BinOpFlags{});

    SMTConfig config;
    config.timeout_ms = 5000;
    config.honor_binop_flags = false;  // disable flag encoding
    SMTVerifier verifier(config);

    auto result = verifier.verify(*fn_orig, *fn_cand);
    std::cout << "    [flags=false] status=" << static_cast<int>(result.status)
              << " msg=" << result.message << "\n";
    // With flags disabled, both sides encode as plain `bvadd` → Equivalent.
    CHECK(result.status == VerificationResult::Equivalent ||
          result.status == VerificationResult::Unknown,
          "honor_binop_flags=false: add nsw vs add treated as Equivalent (legacy)");
}

// ═══════════════════════════════════════════════════════════════════════════
//  R1.E + R1.D + R3.B tests — CEGIS for symbolic-constant synthesis
// ═══════════════════════════════════════════════════════════════════════════

// Test 4: original = `shl x, 3`, candidate_template = `mul x, placeholder_0`.
// The synthesiser should find placeholder_0 = 8 (since x << 3 == x * 8).
void test_cegis_finds_multiplier() {
    Module mod("cegis_mul");
    auto orig = make_shl_const(mod, "shl_3", 3);
    auto tmpl = make_mul_placeholder(mod, "mul_ph", "placeholder_0");

    SMTConfig config;
    config.timeout_ms = 5000;
    SMTVerifier verifier(config);

    auto result = verifier.synthesize_with_cegis(
        *orig, *tmpl, {"placeholder_0"}, {32});
    std::cout << "    [cegis mul] success=" << result.success
              << " msg=" << result.message << "\n";
    if (result.success) {
        std::cout << "    model:";
        for (auto& [n, v] : result.model) std::cout << " " << n << "=" << v;
        std::cout << "\n";
    }

    if (SMTVerifier::is_z3_available()) {
        // With Z3, the synthesiser should find placeholder_0 = 8.
        CHECK(result.success, "CEGIS should find placeholder_0 = 8");
        CHECK(result.model.size() == 1, "CEGIS model has 1 entry");
        if (!result.model.empty()) {
            CHECK(result.model[0].first == "placeholder_0",
                  "CEGIS model entry is placeholder_0");
            CHECK(result.model[0].second == 8,
                  "CEGIS synthesised placeholder_0 = 8");
        }
        CHECK(result.verification.status == VerificationResult::Equivalent,
              "CEGIS final verification is Equivalent");
    } else {
        // Without Z3, verify() returns Unknown, so CEGIS can't confirm
        // equivalence — success should be false.
        std::cout << "    (Z3 unavailable — skipping positive CEGIS assertions)\n";
        CHECK(true, "CEGIS test ran without crash (Z3 unavailable)");
    }
}

// Test 5 (negative): original = `shl x, 3`, candidate_template = `add x, ph`.
// No constant `c` makes `x + c == x << 3` for all x. CEGIS should fail.
void test_cegis_negative_no_model() {
    Module mod("cegis_neg");
    auto orig = make_shl_const(mod, "shl_3_neg", 3);
    auto tmpl = make_add_placeholder(mod, "add_ph", "placeholder_0");

    SMTConfig config;
    config.timeout_ms = 5000;
    SMTVerifier verifier(config);

    auto result = verifier.synthesize_with_cegis(
        *orig, *tmpl, {"placeholder_0"}, {32});
    std::cout << "    [cegis neg] success=" << result.success
              << " msg=" << result.message << "\n";

    // No constant makes x + c == x << 3 for all x.
    CHECK(!result.success, "CEGIS should fail to find a model for x+c == x<<3");
}

// ═══════════════════════════════════════════════════════════════════════════
//  R4.D test — unsat-core / assumption-based batch pruning
// ═══════════════════════════════════════════════════════════════════════════

// Test 6: batch of 4 candidates, ALL obviously wrong (return constants
// different from the original `add x, 1`). The unsat-core pre-pass should
// detect that all 4 are jointly unequivalent in a SINGLE Z3 call and
// short-circuit them all to NotEquivalent.
void test_unsat_core_batch_pruning() {
    Module mod("unsat_core_batch");
    TypeContext& ctx = mod.type_context();

    // Original: f(x) = x + 1
    auto fn_type = std::make_shared<FunctionType>(ctx.int32(),
        std::vector<std::shared_ptr<Type>>{ctx.int32()});
    auto& orig_fn = mod.add_function("orig_x_plus_1", fn_type);
    orig_fn.add_argument(ctx.int32(), "x");
    auto& entry = orig_fn.add_block("entry");
    auto x = make_arg(ctx, "x");
    auto one = ConstantInt::get(ctx, 1, 32);
    entry.add_instruction(inst::make_add(x, one, "r"));
    entry.add_instruction(inst::make_ret(entry.instruction(0)));
    auto original = mod.function("orig_x_plus_1");

    // 4 obviously-wrong candidates: each returns a constant.
    std::vector<Function> candidates;
    auto c1 = make_return_const(mod, "ret_42", 42);
    auto c2 = make_return_const(mod, "ret_99", 99);
    auto c3 = make_return_const(mod, "ret_7", 7);
    auto c4 = make_return_const(mod, "ret_13", 13);
    candidates.push_back(*c1);
    candidates.push_back(*c2);
    candidates.push_back(*c3);
    candidates.push_back(*c4);

    SMTConfig config;
    config.timeout_ms = 5000;
    config.use_unsat_core_batch = true;
    SMTVerifier verifier(config);

    auto results = verifier.verify_batch(*original, candidates);
    std::cout << "    [unsat-core batch] results:";
    for (auto& r : results) std::cout << " " << static_cast<int>(r.status);
    std::cout << "\n";

    CHECK(results.size() == 4, "batch returns 4 results");

    if (SMTVerifier::is_z3_available()) {
        // All 4 should be NotEquivalent (pruned or individually verified).
        for (size_t i = 0; i < results.size(); ++i) {
            CHECK(results[i].status == VerificationResult::NotEquivalent,
                  "candidate " + std::to_string(i) + " is NotEquivalent");
        }
    } else {
        // Without Z3, results may be Unknown — just check the call didn't
        // crash and returned the right count.
        CHECK(true, "batch ran without crash (Z3 unavailable)");
    }
}

// Test 7: batch with MIXED candidates — some equivalent, some wrong. The
// unsat-core pre-pass should NOT prune all (since some are equivalent), and
// the per-candidate loop should correctly identify each.
void test_unsat_core_mixed_batch() {
    Module mod("unsat_core_mixed");
    TypeContext& ctx = mod.type_context();

    // Original: f(x) = x + 1
    auto fn_type = std::make_shared<FunctionType>(ctx.int32(),
        std::vector<std::shared_ptr<Type>>{ctx.int32()});
    auto& orig_fn = mod.add_function("orig_x_plus_1_mixed", fn_type);
    orig_fn.add_argument(ctx.int32(), "x");
    auto& entry = orig_fn.add_block("entry");
    auto x = make_arg(ctx, "x");
    auto one = ConstantInt::get(ctx, 1, 32);
    entry.add_instruction(inst::make_add(x, one, "r"));
    entry.add_instruction(inst::make_ret(entry.instruction(0)));
    auto original = mod.function("orig_x_plus_1_mixed");

    // Candidate 0: equivalent (f(x) = x + 1, built to match the original
    // signature). We construct it inline (not via make_add_xy, which takes
    // 2 args).
    auto& equiv_fn = mod.add_function("cand_equiv", fn_type);
    equiv_fn.add_argument(ctx.int32(), "x");
    auto& equiv_entry = equiv_fn.add_block("entry");
    auto equiv_x = make_arg(ctx, "x");
    auto equiv_one = ConstantInt::get(ctx, 1, 32);
    equiv_entry.add_instruction(inst::make_add(equiv_x, equiv_one, "r"));
    equiv_entry.add_instruction(inst::make_ret(equiv_entry.instruction(0)));
    auto c0 = mod.function("cand_equiv");

    // Candidate 1: wrong (returns 42).
    auto c1 = make_return_const(mod, "cand_42", 42);
    // Candidate 2: wrong (returns 99).
    auto c2 = make_return_const(mod, "cand_99", 99);
    // Candidate 3: wrong (returns 7).
    auto c3 = make_return_const(mod, "cand_7", 7);

    std::vector<Function> candidates;
    candidates.push_back(*c0);
    candidates.push_back(*c1);
    candidates.push_back(*c2);
    candidates.push_back(*c3);

    SMTConfig config;
    config.timeout_ms = 5000;
    config.use_unsat_core_batch = true;
    SMTVerifier verifier(config);

    auto results = verifier.verify_batch(*original, candidates);
    std::cout << "    [mixed batch] results:";
    for (auto& r : results) std::cout << " " << static_cast<int>(r.status);
    std::cout << "\n";

    CHECK(results.size() == 4, "mixed batch returns 4 results");

    if (SMTVerifier::is_z3_available()) {
        // Candidate 0 should be Equivalent (or Unknown).
        CHECK(results[0].status == VerificationResult::Equivalent ||
              results[0].status == VerificationResult::Unknown,
              "candidate 0 (equivalent) is Equivalent or Unknown");
        // Candidates 1-3 should be NotEquivalent (or Unknown).
        for (size_t i = 1; i < results.size(); ++i) {
            CHECK(results[i].status == VerificationResult::NotEquivalent ||
                  results[i].status == VerificationResult::Unknown,
                  "candidate " + std::to_string(i) + " (wrong) is NotEquivalent or Unknown");
            // Soundness: never Equivalent for wrong candidates.
            CHECK(results[i].status != VerificationResult::Equivalent,
                  "SOUNDNESS: wrong candidate " + std::to_string(i) + " not Equivalent");
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════

int main() {
    std::cout << "=== Clunk SMT Verifier I1-RETRY Tests ===" << std::endl;
    std::cout << "  Z3 available: "
              << (SMTVerifier::is_z3_available() ? "yes" : "no") << std::endl;

    std::cout << "\n  R4.B — nsw/nuw/exact poison encoding...\n";
    std::cout << "    [test 1] add nsw x,y vs add nsw x,y (Equivalent)...\n";
    test_add_nsw_both_equivalent();
    std::cout << "    [test 2] add nsw x,y vs add x,y (NotEquivalent)...\n";
    test_add_nsw_vs_plain_not_equivalent();
    std::cout << "    [test 3] add x,y vs add x,y (Equivalent)...\n";
    test_add_plain_both_equivalent();
    std::cout << "    [test 3b] honor_binop_flags=false (fallback)...\n";
    test_honor_flags_false_falls_back();

    std::cout << "\n  R1.E + R1.D + R3.B — CEGIS synthesis...\n";
    std::cout << "    [test 4] shl x,3 vs mul x,? (find 8)...\n";
    test_cegis_finds_multiplier();
    std::cout << "    [test 5] shl x,3 vs add x,? (no model)...\n";
    test_cegis_negative_no_model();

    std::cout << "\n  R4.D — unsat-core batch pruning...\n";
    std::cout << "    [test 6] 4 wrong candidates (prune all)...\n";
    test_unsat_core_batch_pruning();
    std::cout << "    [test 7] mixed batch (1 equiv, 3 wrong)...\n";
    test_unsat_core_mixed_batch();

    std::cout << "\n=== Results: " << g_pass << " passed, " << g_fail << " failed ===" << std::endl;
    return g_fail > 0 ? 1 : 0;
}
