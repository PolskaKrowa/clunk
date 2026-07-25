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
 * Clunk Peephole Miner harvesting tests — Souper/Minotaur-style slice
 * extraction (R1.A + R3.A) and path-condition computation (R1.B).
 *
 * Verifies that the miner:
 *   - harvests a simple integer slice into a mini-function with the same
 *     opcodes (R1.A),
 *   - replaces a memory operand with an opaque var instead of bailing (R1.A),
 *   - respects the depth bound, producing a mini-function with at most
 *     `max_depth` instructions (R3.A),
 *   - computes the correct path conditions for an if-then-else CFG (R1.B),
 *   - computes the correct path conditions for a loop header (R1.B),
 *   - harvests and rewrites a foldable slice end-to-end (R1.A + R3.A).
 */
#include <cstdio>
#include <iostream>
#include <memory>
#include <string>

#include "clunk/Parser/IRParser.h"
#include "clunk/IR/Clone.h"
#include "clunk/IR/Instruction.h"
#include "clunk/Evaluator/EvaluationEngine.h"
#include "clunk/Search/PeepholeMiner.h"
#include "clunk/Search/SMTVerifier.h"

static int g_pass = 0, g_fail = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { std::cerr << "FAIL: " << msg << " at " << __FILE__ << ":" << __LINE__ << "\n"; g_fail++; } \
    else { g_pass++; } \
} while(0)

using namespace clunk;

static std::shared_ptr<ir::Function> parse_fn(const std::string& ir, const std::string& name) {
    parser::IRParser p;
    auto mod = p.parse_string(ir);
    if (!mod) return nullptr;
    return mod->function(name);
}

// ── Test 1: harvest a simple integer slice ─────────────────────────────────
// Source: %a = add %x, %x; %b = mul %a, %a; ret %b
// Harvest %b (index 1) with max_depth=4. The slice should include both %a
// and %b (%a is single-use, sliceable, same block). %x (a function arg)
// becomes an opaque var. The mini-function should have add + mul + ret.
void test_harvest_simple_slice() {
    const char* ir = R"(
define i32 @f(i32 %x) {
entry:
  %a = add i32 %x, %x
  %b = mul i32 %a, %a
  ret i32 %b
}
)";
    auto fn = parse_fn(ir, "f");
    CHECK(fn != nullptr, "parsed simple-slice source");
    if (!fn) return;

    evaluator::EvaluationEngine engine;
    search::PeepholeMiner miner(&engine, {});

    auto h = miner.harvest_slice(*fn, "entry", /*inst_index=*/1, /*max_depth=*/4);
    CHECK(h.has_value(), "harvest_slice returned a slice for %b");
    if (!h) return;

    CHECK(h->mini_function != nullptr, "harvested mini-function is non-null");
    CHECK(h->source_block == "entry", "source_block is 'entry'");
    CHECK(h->source_inst_index == 1, "source_inst_index is 1");
    CHECK(ir::validate_function(*h->mini_function), "mini-function is well-formed SSA");

    // The mini-function should have add + mul + ret = 3 instructions.
    auto& bb = h->mini_function->blocks().front();
    CHECK(bb->size() == 3, "mini-function has 3 instructions (add, mul, ret)");
    CHECK(bb->instruction(0)->opcode() == ir::Opcode::Add,
          "first instruction is add");
    CHECK(bb->instruction(1)->opcode() == ir::Opcode::Mul,
          "second instruction is mul");
    CHECK(bb->instruction(2)->opcode() == ir::Opcode::Ret,
          "terminator is ret");

    // %x became an opaque var.
    CHECK(h->opaque_var_names.size() == 1, "one opaque var (for %x)");
    CHECK(h->var_origins.size() == 1, "one var-origin pair");
    if (!h->var_origins.empty()) {
        CHECK(h->var_origins[0].first == "arg0", "opaque var named arg0");
        CHECK(h->var_origins[0].second == "x", "arg0 originated from %x");
    }

    // The mini-function's argument is the opaque var.
    CHECK(h->mini_function->argument_count() == 1, "mini-function has 1 argument");
    if (h->mini_function->argument_count() == 1) {
        CHECK(h->mini_function->arguments()[0].name == "arg0",
              "mini-function argument named arg0");
    }
}

// ── Test 2: harvest with a memory operand ──────────────────────────────────
// Source: %v = load i32, i32* %p; %r = add i32 %v, 1; ret %r
// Harvest %r (index 1) with max_depth=4. The Load is not a sliceable op, so
// %v becomes an opaque var. The slice is just {%r} (add). The constant 1
// stays inline. The mini-function should be add + ret.
void test_harvest_with_memory_operand() {
    const char* ir = R"(
define i32 @f(i32* %p) {
entry:
  %v = load i32, i32* %p
  %r = add i32 %v, 1
  ret i32 %r
}
)";
    auto fn = parse_fn(ir, "f");
    CHECK(fn != nullptr, "parsed memory-operand source");
    if (!fn) return;

    evaluator::EvaluationEngine engine;
    search::PeepholeMiner miner(&engine, {});

    auto h = miner.harvest_slice(*fn, "entry", /*inst_index=*/1, /*max_depth=*/4);
    CHECK(h.has_value(), "harvest_slice returned a slice for %r despite the Load");
    if (!h) return;

    CHECK(h->mini_function != nullptr, "harvested mini-function is non-null");
    CHECK(ir::validate_function(*h->mini_function), "mini-function is well-formed SSA");

    // The mini-function should have add + ret = 2 instructions (the Load was
    // NOT pulled into the slice — it became an opaque var).
    auto& bb = h->mini_function->blocks().front();
    CHECK(bb->size() == 2, "mini-function has 2 instructions (add, ret)");
    CHECK(bb->instruction(0)->opcode() == ir::Opcode::Add,
          "first instruction is add");
    CHECK(bb->instruction(1)->opcode() == ir::Opcode::Ret,
          "terminator is ret");

    // %v (the Load result) became an opaque var.
    CHECK(h->opaque_var_names.size() == 1, "one opaque var (for %v)");
    if (!h->var_origins.empty()) {
        CHECK(h->var_origins[0].first == "arg0", "opaque var named arg0");
        CHECK(h->var_origins[0].second == "v", "arg0 originated from %v (the load)");
    }

    // No Load/Store in the mini-function — memory was abstracted away.
    bool has_memory = false;
    for (auto& in : bb->instructions())
        if (in && in->is_memory_op()) has_memory = true;
    CHECK(!has_memory, "mini-function contains no memory ops (Load became arg0)");
}

// ── Test 3: depth-bounded harvest ──────────────────────────────────────────
// Source: a 10-deep linear chain of adds, each using the previous result
// once and %y once. Harvest %a9 (index 9) with max_depth=4. The slice should
// contain at most 4 instructions (%a6..%a9); %a5 (depth-exceeded) and %y
// (multi-use across all 10 adds) become opaque vars.
void test_harvest_depth_bound() {
    std::string ir = "define i32 @f(i32 %x, i32 %y) {\nentry:\n";
    ir += "  %a0 = add i32 %x, %y\n";
    for (int i = 1; i < 10; ++i)
        ir += "  %a" + std::to_string(i) + " = add i32 %a" +
              std::to_string(i - 1) + ", %y\n";
    ir += "  ret i32 %a9\n}\n";

    auto fn = parse_fn(ir, "f");
    CHECK(fn != nullptr, "parsed 10-deep add chain");
    if (!fn) return;

    evaluator::EvaluationEngine engine;
    search::PeepholeMiner miner(&engine, {});

    auto h = miner.harvest_slice(*fn, "entry", /*inst_index=*/9, /*max_depth=*/4);
    CHECK(h.has_value(), "harvest_slice returned a slice for %a9");
    if (!h) return;

    CHECK(h->mini_function != nullptr, "harvested mini-function is non-null");
    CHECK(ir::validate_function(*h->mini_function), "mini-function is well-formed SSA");

    // The body (excluding ret) must have at most 4 instructions.
    auto& bb = h->mini_function->blocks().front();
    const size_t body_ops = bb->size() > 0 ? bb->size() - 1 : 0;  // excl. ret
    CHECK(body_ops <= 4, "mini-function body has at most 4 instructions (depth bound)");
    CHECK(body_ops >= 1, "mini-function body has at least 1 instruction (the seed)");

    // At least one opaque var: %a5 (depth-exceeded) and/or %y (multi-use).
    CHECK(h->opaque_var_names.size() >= 1,
          "at least one opaque var (depth-exceeded or multi-use operand)");

    // Every body instruction is an add (the chain is all-add).
    bool all_adds = true;
    for (size_t i = 0; i + 1 < bb->size(); ++i)
        if (bb->instruction(i)->opcode() != ir::Opcode::Add) all_adds = false;
    CHECK(all_adds, "all body instructions are adds");
}

// ── Test 4: path conditions for an if-then-else ────────────────────────────
// Source: entry: %c = icmp sgt %x, %y; br i1 %c, label %then, label %else
//         then: ...; br label %merge
//         else: ...; br label %merge
//         merge: %r = phi ...; ret %r
// PC(then) = {sgt(x, y), negated=false}  (took the true branch)
// PC(else) = {sgt(x, y), negated=true}   (took the false branch)
void test_compute_path_conditions_if_then_else() {
    const char* ir = R"(
define i32 @f(i32 %x, i32 %y) {
entry:
  %c = icmp sgt i32 %x, %y
  br i1 %c, label %then, label %else
then:
  %a = add i32 %x, 1
  br label %merge
else:
  %b = sub i32 %x, 1
  br label %merge
merge:
  %r = phi i32 [ %a, %then ], [ %b, %else ]
  ret i32 %r
}
)";
    auto fn = parse_fn(ir, "f");
    CHECK(fn != nullptr, "parsed if-then-else source");
    if (!fn) return;

    evaluator::EvaluationEngine engine;
    search::PeepholeMiner miner(&engine, {});

    // PC for "then" — took the true branch of `br i1 %c, %then, %else`.
    auto pcs_then = miner.compute_path_conditions(*fn, "then");
    CHECK(pcs_then.size() == 1, "one PC for the 'then' block");
    if (!pcs_then.empty()) {
        const auto& pc = pcs_then[0];
        CHECK(pc.block_name == "entry", "PC block_name is 'entry'");
        CHECK(pc.branch_inst_index == 1, "PC branch_inst_index is 1 (the br)");
        CHECK(pc.predicate == ir::CmpPredicate::SGT, "PC predicate is SGT");
        CHECK(pc.lhs_name == "x", "PC lhs_name is 'x'");
        CHECK(pc.rhs_name == "y", "PC rhs_name is 'y'");
        CHECK(pc.negated == false, "PC not negated (true branch to 'then')");
    }

    // PC for "else" — took the false branch.
    auto pcs_else = miner.compute_path_conditions(*fn, "else");
    CHECK(pcs_else.size() == 1, "one PC for the 'else' block");
    if (!pcs_else.empty()) {
        const auto& pc = pcs_else[0];
        CHECK(pc.block_name == "entry", "PC block_name is 'entry'");
        CHECK(pc.branch_inst_index == 1, "PC branch_inst_index is 1 (the br)");
        CHECK(pc.predicate == ir::CmpPredicate::SGT, "PC predicate is SGT");
        CHECK(pc.lhs_name == "x", "PC lhs_name is 'x'");
        CHECK(pc.rhs_name == "y", "PC rhs_name is 'y'");
        CHECK(pc.negated == true, "PC negated (false branch to 'else')");
    }

    // PC for "merge" — both then->merge and else->merge are unconditional,
    // so the only PCs are the ones inherited from then/else (the entry branch).
    // We should get 2 PCs (one per predecessor path), both with SGT predicate
    // — one negated (via else), one not (via then).
    auto pcs_merge = miner.compute_path_conditions(*fn, "merge");
    CHECK(pcs_merge.size() == 2, "two PCs for the 'merge' block (one per pred)");
    if (pcs_merge.size() == 2) {
        bool found_non_negated = false, found_negated = false;
        for (const auto& pc : pcs_merge) {
            if (pc.predicate == ir::CmpPredicate::SGT && !pc.negated) found_non_negated = true;
            if (pc.predicate == ir::CmpPredicate::SGT && pc.negated) found_negated = true;
        }
        CHECK(found_non_negated, "merge PC includes the true-branch (then) condition");
        CHECK(found_negated, "merge PC includes the false-branch (else) condition");
    }
}

// ── Test 5: path conditions for a loop header ──────────────────────────────
// Source: entry: br label %loop
//         loop: %i = phi [0, %entry], [%i.next, %loop];
//               %cond = icmp slt %i, %n; %i.next = add %i, 1;
//               br i1 %cond, label %loop, label %exit
//         exit: ret %x
// PC(loop) = {slt(i, n), negated=false}  (the back-edge: stay in loop)
// PC(exit) = {slt(i, n), negated=true}   (the exit edge: leave loop)
void test_compute_path_conditions_loop_header() {
    const char* ir = R"(
define i32 @f(i32 %x, i32 %n) {
entry:
  br label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]
  %cond = icmp slt i32 %i, %n
  %i.next = add i32 %i, 1
  br i1 %cond, label %loop, label %exit
exit:
  ret i32 %x
}
)";
    auto fn = parse_fn(ir, "f");
    CHECK(fn != nullptr, "parsed loop source");
    if (!fn) return;

    evaluator::EvaluationEngine engine;
    search::PeepholeMiner miner(&engine, {});

    // PC for "loop" — the back-edge (true branch of the loop-condition br).
    auto pcs_loop = miner.compute_path_conditions(*fn, "loop");
    CHECK(pcs_loop.size() == 1, "one PC for the 'loop' block (the back-edge)");
    if (!pcs_loop.empty()) {
        const auto& pc = pcs_loop[0];
        CHECK(pc.block_name == "loop", "PC block_name is 'loop'");
        CHECK(pc.branch_inst_index == 3, "PC branch_inst_index is 3 (the br in loop)");
        CHECK(pc.predicate == ir::CmpPredicate::SLT, "PC predicate is SLT");
        CHECK(pc.lhs_name == "i", "PC lhs_name is 'i'");
        CHECK(pc.rhs_name == "n", "PC rhs_name is 'n'");
        CHECK(pc.negated == false, "PC not negated (true branch = stay in loop)");
    }

    // PC for "exit" — the false branch (leave loop).
    auto pcs_exit = miner.compute_path_conditions(*fn, "exit");
    CHECK(pcs_exit.size() == 1, "one PC for the 'exit' block (the exit edge)");
    if (!pcs_exit.empty()) {
        const auto& pc = pcs_exit[0];
        CHECK(pc.block_name == "loop", "PC block_name is 'loop'");
        CHECK(pc.branch_inst_index == 3, "PC branch_inst_index is 3 (the br in loop)");
        CHECK(pc.predicate == ir::CmpPredicate::SLT, "PC predicate is SLT");
        CHECK(pc.lhs_name == "i", "PC lhs_name is 'i'");
        CHECK(pc.rhs_name == "n", "PC rhs_name is 'n'");
        CHECK(pc.negated == true, "PC negated (false branch = leave loop)");
    }
}

// ── Test 6: harvest + rewrite a foldable slice ─────────────────────────────
// Source: %a = add %x, 0; %b = mul %a, 1; ret %b   ==   ret %x
// harvest_and_rewrite should find that (x+0)*1 = x, splice the fold back,
// and return a function with strictly fewer instructions.
void test_harvest_and_rewrite_proven_win() {
    const char* ir = R"(
define i32 @f(i32 %x) {
entry:
  %a = add i32 %x, 0
  %b = mul i32 %a, 1
  ret i32 %b
}
)";
    auto fn = parse_fn(ir, "f");
    CHECK(fn != nullptr, "parsed foldable source (x+0)*1");
    if (!fn) return;

    evaluator::EvaluationEngine engine;
    search::MinerConfig cfg;
    cfg.max_length = 2;
    search::PeepholeMiner miner(&engine, cfg);

    auto out = miner.harvest_and_rewrite(*fn);
    CHECK(out.has_value(), "harvest_and_rewrite found and applied a proven win");
    CHECK(*out != nullptr, "rewritten function is non-null");
    if (!out || !*out) return;

    CHECK(ir::validate_function(**out), "rewritten function is well-formed SSA");

    // The rewrite must have strictly fewer instructions than the original.
    auto count_non_terminators = [](const std::shared_ptr<ir::Function>& f) {
        size_t n = 0;
        for (auto& b : f->blocks())
            for (auto& in : b->instructions())
                if (in && !in->is_terminator()) ++n;
        return n;
    };
    const size_t before = count_non_terminators(fn);
    const size_t after  = count_non_terminators(*out);
    CHECK(after < before, "rewrite has fewer non-terminator instructions");
    CHECK(after == 0, "rewrite collapsed (x+0)*1 to just `ret %x` (0 non-term ops)");

    // The miner should report at least one applied splice.
    CHECK(miner.stats().harvest_rewrites_applied >= 1,
          "stats show at least one harvested slice was spliced");

    // Independent re-verification: the whole rewritten function must be
    // SMT-equivalent to the original.
    if (search::SMTVerifier::is_z3_available()) {
        search::SMTVerifier verifier;
        auto res = verifier.verify(*fn, **out);
        CHECK(res.status == search::VerificationResult::Equivalent,
              "rewritten function re-verifies as Equivalent to the original");
    } else {
        std::cerr << "  (z3 unavailable — skipping independent re-verification)\n";
    }
}

// ── Test 7: verify_with_assumptions — precision and soundness ──────────────
// src:  and(x, 3)   cand: add(x, 0) (an identity — i.e. `x`)
// Unconditionally NOT equivalent (x=4: 0 vs 4). Under the assumption
// x <u 4 they ARE equivalent. Under the weaker assumption x <u 8 they are
// again NOT equivalent (x=4..7 disagree) — the assumption must not leak
// strength it doesn't have.
void test_verify_with_assumptions() {
    if (!search::SMTVerifier::is_z3_available()) {
        std::cerr << "  (z3 unavailable — skipping)\n";
        return;
    }
    auto src = parse_fn(
        "define i32 @s(i32 %arg0) {\nentry:\n  %t = and i32 %arg0, 3\n"
        "  ret i32 %t\n}\n", "s");
    auto cand = parse_fn(
        "define i32 @c(i32 %arg0) {\nentry:\n  %t = add i32 %arg0, 0\n"
        "  ret i32 %t\n}\n", "c");
    CHECK(src && cand, "parsed assumption-test functions");
    if (!src || !cand) return;

    search::SMTVerifier verifier;

    auto plain = verifier.verify(*src, *cand);
    CHECK(plain.status == search::VerificationResult::NotEquivalent,
          "and(x,3) vs x: NotEquivalent unconditionally");

    search::ArgAssumption ult4;
    ult4.predicate = ir::CmpPredicate::ULT;
    ult4.lhs_arg = 0;
    ult4.rhs_const = 4;
    auto under = verifier.verify_with_assumptions(*src, *cand, {ult4});
    CHECK(under.status == search::VerificationResult::Equivalent,
          "and(x,3) vs x: Equivalent under x <u 4");

    search::ArgAssumption ult8 = ult4;
    ult8.rhs_const = 8;
    auto weak = verifier.verify_with_assumptions(*src, *cand, {ult8});
    CHECK(weak.status == search::VerificationResult::NotEquivalent,
          "and(x,3) vs x: still NotEquivalent under the weaker x <u 8");

    // negated=true flips the predicate: NOT(x >=u 4) ≡ x <u 4.
    search::ArgAssumption not_uge4;
    not_uge4.predicate = ir::CmpPredicate::UGE;
    not_uge4.lhs_arg = 0;
    not_uge4.rhs_const = 4;
    not_uge4.negated = true;
    auto negated = verifier.verify_with_assumptions(*src, *cand, {not_uge4});
    CHECK(negated.status == search::VerificationResult::Equivalent,
          "and(x,3) vs x: Equivalent under NOT(x >=u 4)");
}

// ── Test 8: PC-aware mining end-to-end ──────────────────────────────────────
// then-block computes (x & 3) + 1 under the dominating condition x <u 4,
// where it simplifies to x + 1. Unconditional equivalence cannot prove
// this, so plain harvest_and_rewrite must find nothing; the PC-aware path
// must find it, splice it, and the WHOLE rewritten function must still
// re-verify Equivalent (the changed block only executes under the branch).
void test_mine_with_path_conditions_end_to_end() {
    if (!search::SMTVerifier::is_z3_available()) {
        std::cerr << "  (z3 unavailable — skipping)\n";
        return;
    }
    const char* ir = R"(
define i32 @f(i32 %x) {
entry:
  %c = icmp ult i32 %x, 4
  br i1 %c, label %then, label %els
then:
  %t = and i32 %x, 3
  %r = add i32 %t, 1
  br label %merge
els:
  br label %merge
merge:
  %p = phi i32 [ %r, %then ], [ 0, %els ]
  ret i32 %p
}
)";
    auto fn = parse_fn(ir, "f");
    CHECK(fn != nullptr, "parsed PC end-to-end function");
    if (!fn) return;

    evaluator::EvaluationEngine engine;

    // Without PCs: the slice (and+add) has no unconditionally-equivalent
    // cheaper form, so plain harvesting proves nothing.
    {
        search::MinerConfig cfg;
        cfg.use_path_conditions = false;
        search::PeepholeMiner miner(&engine, cfg);
        auto out = miner.harvest_and_rewrite(*fn);
        CHECK(!out.has_value(), "PC-unaware harvesting finds no win here");
    }

    // With PCs: `and x,3` folds away under x <u 4.
    search::MinerConfig cfg;
    cfg.use_path_conditions = true;
    search::PeepholeMiner miner(&engine, cfg);
    bool proven = false;
    auto out = miner.mine_with_path_conditions(*fn, &proven);
    CHECK(out.has_value() && *out, "PC-aware mining rewrote the function");
    if (!out || !*out) return;
    CHECK(miner.stats().pc_slices_mined >= 1, "win was proven under a path condition");
    CHECK(proven, "whole rewritten function was SMT re-proven Equivalent");
    CHECK(ir::validate_function(**out), "rewritten function is well-formed SSA");

    // The then-block should have shrunk: and+add+br → add+br.
    auto then_bb = (*out)->block("then");
    CHECK(then_bb != nullptr, "then block still exists");
    if (then_bb) {
        CHECK(then_bb->size() == 2,
              "then block collapsed to one instruction plus the branch");
        bool has_and = false;
        for (auto& in : then_bb->instructions())
            if (in && in->opcode() == ir::Opcode::And) has_and = true;
        CHECK(!has_and, "the and was eliminated under the path condition");
    }

    // Independent re-verification of the whole rewrite.
    search::SMTVerifier verifier;
    auto res = verifier.verify(*fn, **out);
    CHECK(res.status == search::VerificationResult::Equivalent,
          "rewritten function independently re-verifies Equivalent");
}

// ── Test 9: multi-use harvesting keeps shared sub-expressions ──────────────
// %t is used by BOTH the store and the mineable expression. The old
// harvester turned %t into an opaque var (slice = {%u} — too small to
// mine); following the multi-use operand exposes shl+mul = shl, and the
// splice must KEEP %t alive for the store. The store makes the whole
// function unprovable by SMT, so this needs trust_unverified_slices — and
// the miner must honestly report whole_fn_proven == false.
void test_multiuse_harvest_keeps_shared_value() {
    if (!search::SMTVerifier::is_z3_available()) {
        std::cerr << "  (z3 unavailable — skipping)\n";
        return;
    }
    const char* ir = R"(
define i32 @g(i32 %x, i32* %p) {
entry:
  %t = shl i32 %x, 1
  store i32 %t, i32* %p
  %u = mul i32 %t, 2
  ret i32 %u
}
)";
    auto fn = parse_fn(ir, "g");
    CHECK(fn != nullptr, "parsed multi-use function");
    if (!fn) return;

    evaluator::EvaluationEngine engine;
    search::MinerConfig cfg;
    cfg.trust_unverified_slices = true;  // the store defeats whole-fn SMT
    search::PeepholeMiner miner(&engine, cfg);
    bool proven = true;
    auto out = miner.harvest_and_rewrite(*fn, &proven);
    CHECK(out.has_value() && *out, "multi-use slice was mined and spliced");
    if (!out || !*out) return;
    CHECK(!proven, "whole_fn_proven is honestly false (store defeats SMT)");
    CHECK(ir::validate_function(**out), "rewritten function is well-formed SSA");

    auto bb = (*out)->blocks().front();
    bool has_shl_t = false, has_store = false, has_mul = false;
    for (auto& in : bb->instructions()) {
        if (!in) continue;
        if (in->opcode() == ir::Opcode::Shl && in->has_name() && in->name() == "t")
            has_shl_t = true;
        if (in->opcode() == ir::Opcode::Store) has_store = true;
        if (in->opcode() == ir::Opcode::Mul) has_mul = true;
    }
    CHECK(has_shl_t, "shared %t kept alive for its other user");
    CHECK(has_store, "store survived the splice");
    CHECK(!has_mul, "mul was strength-reduced away (shl x,1 * 2 -> shl x,2)");
}

// ── Test 10: select-arm conditional mining ──────────────────────────────────
// Real clang -O3 output shape for `if ((x&1)==0) return x & ~1; return -x;`:
// clang leaves `and i32 %x, -2` in the true arm even though it equals %x
// under the select condition. The select-arm miner must fold it and the
// whole function must re-verify Equivalent (sound mode, no trust needed).
void test_select_arm_conditional_mining() {
    if (!search::SMTVerifier::is_z3_available()) {
        std::cerr << "  (z3 unavailable — skipping)\n";
        return;
    }
    const char* ir = R"(
define i32 @clear_known(i32 %x) {
entry:
  %m = and i32 %x, 1
  %c = icmp eq i32 %m, 0
  %a = and i32 %x, -2
  %n = sub i32 0, %x
  %s = select i1 %c, i32 %a, i32 %n
  ret i32 %s
}
)";
    auto fn = parse_fn(ir, "clear_known");
    CHECK(fn != nullptr, "parsed select-arm function");
    if (!fn) return;

    evaluator::EvaluationEngine engine;
    search::MinerConfig cfg;
    cfg.use_path_conditions = true;
    search::PeepholeMiner miner(&engine, cfg);
    bool proven = false;
    auto out = miner.mine_with_path_conditions(*fn, &proven);
    CHECK(out.has_value() && *out, "select-arm mining rewrote the function");
    if (!out || !*out) return;
    CHECK(proven, "whole rewritten function SMT re-proven (pure, one block)");
    CHECK(ir::validate_function(**out), "rewritten function is well-formed SSA");

    // The true arm's `and x, -2` must be gone: the select now picks %x
    // directly. The condition chain (%m, %c) must survive (the select
    // still needs it).
    auto bb = (*out)->blocks().front();
    size_t n_and = 0;
    bool has_icmp = false, has_select = false;
    for (auto& in : bb->instructions()) {
        if (!in) continue;
        if (in->opcode() == ir::Opcode::And) ++n_and;
        if (in->opcode() == ir::Opcode::ICmp) has_icmp = true;
        if (in->opcode() == ir::Opcode::Select) has_select = true;
    }
    CHECK(n_and == 1, "arm's redundant and eliminated (only the cond's and remains)");
    CHECK(has_icmp && has_select, "condition chain and select survive");

    search::SMTVerifier verifier;
    auto res = verifier.verify(*fn, **out);
    CHECK(res.status == search::VerificationResult::Equivalent,
          "rewritten function independently re-verifies Equivalent");
}

// ── Test 11: cast-aware mining ──────────────────────────────────────────────
// `and (zext i8 %x to i32), 255` is just `zext %x` — requires (a) casts as
// slice members (multi-width analyze), (b) the enumerator's cast seeds, and
// (c) the exact Z3 zero_ext encoding (casts used to be fresh variables, so
// nothing containing a cast could ever be proven).
void test_cast_slice_mining() {
    if (!search::SMTVerifier::is_z3_available()) {
        std::cerr << "  (z3 unavailable — skipping)\n";
        return;
    }
    auto fn = parse_fn(
        "define i32 @zm(i8 %x) {\nentry:\n  %z = zext i8 %x to i32\n"
        "  %m = and i32 %z, 255\n  ret i32 %m\n}\n", "zm");
    CHECK(fn != nullptr, "parsed zext+mask function");
    if (!fn) return;

    evaluator::EvaluationEngine engine;
    search::PeepholeMiner miner(&engine, {});
    auto mp = miner.mine_function(*fn);
    CHECK(mp.has_value(), "mined a win for and(zext x, 255)");
    if (!mp) return;
    auto& bb = mp->replacement->blocks().front();
    bool has_and = false, has_zext = false;
    for (auto& in : bb->instructions()) {
        if (!in) continue;
        if (in->opcode() == ir::Opcode::And) has_and = true;
        if (in->opcode() == ir::Opcode::ZExt) has_zext = true;
    }
    CHECK(has_zext && !has_and, "replacement is the bare zext (mask dropped)");

    search::SMTVerifier verifier;
    auto res = verifier.verify(*fn, *mp->replacement);
    CHECK(res.status == search::VerificationResult::Equivalent,
          "cast replacement independently re-verifies");
}

// ── Test 12: refinement semantics guard poison both ways ───────────────────
// Original `shl %x, 32` (always poison at i32) may be replaced by `ret 0` —
// refining poison to a defined value is LLVM's contract. The REVERSE must be
// rejected: a candidate that turns the defined `xor %x, %x` (= 0) into an
// always-poison shift would miscompile. This also pins down the shift-
// out-of-range poison modelling the legacy encoding lacked entirely.
void test_refinement_shift_oob_poison() {
    if (!search::SMTVerifier::is_z3_available()) {
        std::cerr << "  (z3 unavailable — skipping)\n";
        return;
    }
    auto poison_fn = parse_fn(
        "define i32 @p(i32 %x) {\nentry:\n  %r = shl i32 %x, 32\n"
        "  ret i32 %r\n}\n", "p");
    auto zero_fn = parse_fn(
        "define i32 @z(i32 %x) {\nentry:\n  %r = xor i32 %x, %x\n"
        "  ret i32 %r\n}\n", "z");
    CHECK(poison_fn && zero_fn, "parsed shift-oob pair");
    if (!poison_fn || !zero_fn) return;

    search::SMTVerifier verifier;  // refinement_semantics defaults on
    auto refine_ok = verifier.verify(*poison_fn, *zero_fn);
    CHECK(refine_ok.status == search::VerificationResult::Equivalent,
          "poison source may be refined to a defined value");
    auto introduce = verifier.verify(*zero_fn, *poison_fn);
    CHECK(introduce.status != search::VerificationResult::Equivalent,
          "SOUNDNESS: introducing an always-poison shift is rejected");
}

int main() {
    std::cout << "=== Clunk Peephole Harvest Tests ===\n";
    std::cout << "  harvest a simple integer slice..." << std::endl;
    test_harvest_simple_slice();
    std::cout << "  harvest with a memory operand..." << std::endl;
    test_harvest_with_memory_operand();
    std::cout << "  depth-bounded harvest..." << std::endl;
    test_harvest_depth_bound();
    std::cout << "  path conditions (if-then-else)..." << std::endl;
    test_compute_path_conditions_if_then_else();
    std::cout << "  path conditions (loop header)..." << std::endl;
    test_compute_path_conditions_loop_header();
    std::cout << "  harvest + rewrite a foldable slice..." << std::endl;
    test_harvest_and_rewrite_proven_win();
    std::cout << "  verify_with_assumptions precision + soundness..." << std::endl;
    test_verify_with_assumptions();
    std::cout << "  PC-aware mining end-to-end..." << std::endl;
    test_mine_with_path_conditions_end_to_end();
    std::cout << "  multi-use harvest keeps shared values..." << std::endl;
    test_multiuse_harvest_keeps_shared_value();
    std::cout << "  select-arm conditional mining..." << std::endl;
    test_select_arm_conditional_mining();
    std::cout << "  cast-aware slice mining..." << std::endl;
    test_cast_slice_mining();
    std::cout << "  refinement: shift-oob poison both ways..." << std::endl;
    test_refinement_shift_oob_poison();

    std::cout << "\n=== Results: " << g_pass << " passed, " << g_fail << " failed ===\n";
    return g_fail > 0 ? 1 : 0;
}
