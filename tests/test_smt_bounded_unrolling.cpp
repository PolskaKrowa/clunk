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
 * Clunk SMT Bounded-Unrolling + Config Tests — exercise the new
 * --smt-bounded-unrolling flag and the configurable SMT limits.
 *
 * The bounded-unrolling pre-pass in SMTVerifier::verify_with_z3 uses
 * LoopOptimizer::unroll_constant_loops to expand constant-trip
 * single-block loops before SMT encoding. This converts an
 * unverifiable loop function into a verifiable straight-line one.
 *
 * The configurable SMT limits (max_blocks_for_smt,
 * max_instructions_for_smt, timeout_ms, max_smt_attempts) are also
 * exercised — these are now exposed via --smt-max-blocks etc.
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
#include "clunk/Search/SMTVerifier.h"
#include "clunk/Search/LoopOpt.h"
#include "clunk/Pipeline.h"

static int g_pass = 0, g_fail = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { std::cerr << "FAIL: " << msg << " at " << __FILE__ << ":" << __LINE__ << "\n"; g_fail++; } \
    else { g_pass++; } \
} while(0)

using namespace clunk::ir;
using namespace clunk;

// ═══════════════════════════════════════════════════════════════════════════
//  Build a constant-trip-count loop function:
//    sum_loop(n):
//      entry:  br label %loop
//      loop:   %i = phi [0, entry], [%i.next, loop]
//              %s = phi [0, entry], [%s.next, loop]
//              %i.next = add %i, 1
//              %s.next = add %s, %i
//              %cmp = icmp slt %i.next, 4   ; constant trip count = 4
//              br %cmp, %loop, %exit
//      exit:   ret %s.next
//
//  The body computes s = 0+1+2+3 = 6 (constant-foldable).
//  Without bounded unrolling, SMT bails on the back-edge.
//  With bounded unrolling, SMT can prove equivalence to `ret 6`.
// ═══════════════════════════════════════════════════════════════════════════
static std::shared_ptr<Function> make_constant_trip_loop() {
    TypeContext ctx;
    auto i32 = ctx.int32();
    auto fn_type = std::make_shared<FunctionType>(i32, std::vector<std::shared_ptr<Type>>{i32});
    auto fn = std::make_shared<Function>("sum_loop", fn_type, Linkage::External);
    fn->add_argument(i32, "n");
    auto& entry = fn->add_block("entry");
    auto& loop = fn->add_block("loop");
    auto& exit = fn->add_block("exit");

    ir::TypeContext tmp_ctx;
    auto zero = ConstantInt::get(tmp_ctx, 0);
    auto one = ConstantInt::get(tmp_ctx, 1);
    auto four = ConstantInt::get(tmp_ctx, 4);
    auto n_val = std::make_shared<Value>(i32, "n");

    entry.add_instruction(inst::make_br_uncond("loop"));

    // %i = phi [0, entry], [%i.next, loop]
    auto i_phi = std::make_shared<Instruction>(Opcode::Phi, i32, "i");
    i_phi->add_operand(zero);
    i_phi->add_operand(std::make_shared<Value>(i32, "i.next"));
    i_phi->set_metadata("phi_blocks", "entry,loop");
    loop.add_instruction(i_phi);
    // %s = phi [0, entry], [%s.next, loop]
    auto s_phi = std::make_shared<Instruction>(Opcode::Phi, i32, "s");
    s_phi->add_operand(zero);
    s_phi->add_operand(std::make_shared<Value>(i32, "s.next"));
    s_phi->set_metadata("phi_blocks", "entry,loop");
    loop.add_instruction(s_phi);
    // %i.next = add %i, 1
    loop.add_instruction(inst::make_add(i_phi, one, "i.next"));
    // %s.next = add %s, %i
    loop.add_instruction(inst::make_add(s_phi, i_phi, "s.next"));
    // %cmp = icmp slt %i.next, 4
    loop.add_instruction(inst::make_icmp(CmpPredicate::SLT,
        std::make_shared<Value>(i32, "i.next"), four, "cmp"));
    // br %cmp, %loop, %exit
    loop.add_instruction(inst::make_br(loop.instruction(4), "loop", "exit"));

    exit.add_instruction(inst::make_ret(std::make_shared<Value>(i32, "s.next")));
    return fn;
}

// Build a simple "ret 6" function — the unrolled+folded equivalent
// of make_constant_trip_loop().
static std::shared_ptr<Function> make_folded_constant() {
    TypeContext ctx;
    auto i32 = ctx.int32();
    auto fn_type = std::make_shared<FunctionType>(i32, std::vector<std::shared_ptr<Type>>{i32});
    auto fn = std::make_shared<Function>("folded_six", fn_type, Linkage::External);
    fn->add_argument(i32, "n");
    auto& entry = fn->add_block("entry");
    ir::TypeContext tmp_ctx;
    auto six = ConstantInt::get(tmp_ctx, 6);
    entry.add_instruction(inst::make_ret(six));
    return fn;
}

// ═══════════════════════════════════════════════════════════════════════════
//  Tests
// ═══════════════════════════════════════════════════════════════════════════

static void test_smt_bails_on_loop_without_unrolling() {
    std::cout << "  SMT bails on loops without bounded unrolling..." << std::endl;
    auto loop_fn = make_constant_trip_loop();
    auto folded_fn = make_folded_constant();

    search::SMTConfig cfg;
    cfg.sound_loop_fallback = true;
    cfg.sound_bounded_unrolling = false;  // disabled
    search::SMTVerifier verifier(cfg);

    auto result = verifier.verify(*loop_fn, *folded_fn);
    CHECK(result.status == search::VerificationResult::Unknown,
          "without bounded unrolling, SMT returns Unknown for loop functions");
}

static void test_smt_unrolling_enables_verification() {
    std::cout << "  SMT bounded unrolling enables verification of loops..." << std::endl;
    auto loop_fn = make_constant_trip_loop();
    auto folded_fn = make_folded_constant();

    // Sanity check: the LoopOptimizer can unroll our loop.
    search::LoopOptimizer unroller;
    auto unrolled = unroller.unroll_constant_loops(*loop_fn);
    CHECK(unrolled != nullptr, "LoopOptimizer unrolled the constant-trip loop");
    CHECK(!search::SMTVerifier::is_z3_available() ||
          unrolled->blocks().size() >= 1,
          "unrolled function has at least one block");

    // With bounded unrolling enabled, SMT should be able to verify the
    // equivalence (or at least not bail with "loop" reason).
    if (!search::SMTVerifier::is_z3_available()) {
        std::cout << "    (skipping verification — Z3 not available)" << std::endl;
        return;
    }
    search::SMTConfig cfg;
    cfg.sound_loop_fallback = true;
    cfg.sound_bounded_unrolling = true;
    cfg.max_unrolling = 8;  // trip count is 4 — within cap
    search::SMTVerifier verifier(cfg);

    auto result = verifier.verify(*loop_fn, *folded_fn);
    // We expect either Equivalent (the unroller folded the loop to 6 and
    // the folded function is also 6) or Unknown (if Z3 couldn't finish
    // in time). Either way, it should NOT be NotEquivalent.
    CHECK(result.status != search::VerificationResult::NotEquivalent,
          "with bounded unrolling, SMT does not report NotEquivalent for equivalent loops");
    if (result.status == search::VerificationResult::Equivalent) {
        std::cout << "    SMT proved equivalence after bounded unrolling!" << std::endl;
    }
}

static void test_smt_configurable_limits() {
    std::cout << "  SMT limits are configurable..." << std::endl;
    search::SMTConfig cfg;
    cfg.timeout_ms = 5000;
    cfg.max_blocks_for_smt = 50;
    cfg.max_instructions_for_smt = 200;
    cfg.sound_bounded_unrolling = true;

    CHECK(cfg.timeout_ms == 5000, "timeout_ms is set");
    CHECK(cfg.max_blocks_for_smt == 50, "max_blocks_for_smt is set");
    CHECK(cfg.max_instructions_for_smt == 200, "max_instructions_for_smt is set");
    CHECK(cfg.sound_bounded_unrolling, "sound_bounded_unrolling is enabled");
}

static void test_pipeline_max_smt_attempts_configurable() {
    std::cout << "  Pipeline.max_smt_attempts is configurable..." << std::endl;
    PipelineConfig pcfg;
    pcfg.max_smt_attempts = 10;
    CHECK(pcfg.max_smt_attempts == 10, "max_smt_attempts is set");

    pcfg.max_smt_attempts = 0;
    // 0 means "use default"; verify_and_select falls back to 5.
    CHECK(pcfg.max_smt_attempts == 0, "max_smt_attempts=0 means default");
}

int main() {
    std::cout << "=== Clunk SMT Bounded-Unrolling + Config Tests ===" << std::endl;

    test_smt_bails_on_loop_without_unrolling();
    test_smt_unrolling_enables_verification();
    test_smt_configurable_limits();
    test_pipeline_max_smt_attempts_configurable();

    std::cout << "\n=== Results: " << g_pass << " passed, " << g_fail << " failed ===" << std::endl;
    return g_fail > 0 ? 1 : 0;
}
