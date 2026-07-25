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
 * Clunk E-graph Tests — egg-style equality-saturation candidate generation.
 *
 * Tests feature R3.D from research/R3_minotaur_egraphs.md:
 *   1. E-graph hash-consing deduplicates identical e-nodes.
 *   2. Congruence closure merges structurally-congruent parents after a merge.
 *   3. IR-to-e-graph lowering + extraction roundtrips a simple function.
 *   4. egraph_rewrite() finds an improvement on (a + 0) * 1 using the
 *      builtin add_zero_elim and mul_one_elim patterns.
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
#include "clunk/IR/Clone.h"  // for ir::validate_function
#include "clunk/Pattern/PatternLibrary.h"
#include "clunk/Evaluator/EvaluationEngine.h"
#include "clunk/Search/EgraphRewriter.h"
#include "clunk/Search/StochasticSearch.h"  // for search::Candidate

static int g_pass = 0, g_fail = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { std::cerr << "FAIL: " << msg << " at " << __FILE__ << ":" << __LINE__ << "\n"; g_fail++; } \
    else { g_pass++; } \
} while(0)

using namespace clunk::ir;
using namespace clunk::pattern;
using namespace clunk::egraph;
using namespace clunk::evaluator;
using namespace clunk::search;

// ═══════════════════════════════════════════════════════════════════════════
//  Helpers
// ═══════════════════════════════════════════════════════════════════════════

static ArchDescriptor make_x86_64_arch() {
    ArchDescriptor arch;
    arch.name = "x86_64";
    arch.vendor = "intel";
    arch.is_gpu = false;
    arch.vector_width = 256;
    arch.has_avx2 = true;
    arch.has_fma = true;
    return arch;
}

// Build `define i32 @f(i32 %a, i32 %b) { entry: %r = add i32 %a, %b; ret i32 %r }`.
static std::shared_ptr<Function> make_add_function() {
    TypeContext ctx;
    auto fn_type = std::make_shared<FunctionType>(
        ctx.int32(), std::vector<std::shared_ptr<Type>>{ctx.int32(), ctx.int32()});
    auto fn = std::make_shared<Function>("add_fn", fn_type);
    fn->add_argument(ctx.int32(), "a");
    fn->add_argument(ctx.int32(), "b");
    auto& entry = fn->add_block("entry");
    auto a_val = std::make_shared<Value>(ctx.int32(), "a");
    auto b_val = std::make_shared<Value>(ctx.int32(), "b");
    auto add_inst = inst::make_add(a_val, b_val, "r");
    entry.add_instruction(add_inst);
    entry.add_instruction(inst::make_ret(add_inst));
    return fn;
}

// Build `define i32 @f(i32 %a) { entry: %r1 = add %a, 0; %r2 = mul %r1, 1; ret %r2 }`.
static std::shared_ptr<Function> make_add_zero_mul_one_function() {
    TypeContext ctx;
    auto fn_type = std::make_shared<FunctionType>(
        ctx.int32(), std::vector<std::shared_ptr<Type>>{ctx.int32()});
    auto fn = std::make_shared<Function>("opt_fn", fn_type);
    fn->add_argument(ctx.int32(), "a");
    auto& entry = fn->add_block("entry");
    auto a_val = std::make_shared<Value>(ctx.int32(), "a");
    auto zero = ConstantInt::get(ctx, 0, 32);
    auto add_inst = inst::make_add(a_val, zero, "r1");
    entry.add_instruction(add_inst);
    auto one = ConstantInt::get(ctx, 1, 32);
    auto mul_inst = inst::make_mul(add_inst, one, "r2");
    entry.add_instruction(mul_inst);
    entry.add_instruction(inst::make_ret(mul_inst));
    return fn;
}

// ═══════════════════════════════════════════════════════════════════════════
//  Test 1: EGraph::add deduplicates identical e-nodes
// ═══════════════════════════════════════════════════════════════════════════

void test_egraph_add_dedupes_identical_nodes() {
    EGraph eg;

    // Two identical argument e-nodes should share an e-class.
    ENode a1;
    a1.is_argument = true;
    a1.argument_name = "x";
    EClassId id1 = eg.add(a1);

    ENode a2;
    a2.is_argument = true;
    a2.argument_name = "x";
    EClassId id2 = eg.add(a2);

    CHECK(id1 == id2, "two identical argument e-nodes share an e-class");
    CHECK(eg.find(id1) == eg.find(id2), "find() agrees after add()");

    // Two identical constant e-nodes should share an e-class.
    ENode c1;
    c1.is_constant = true;
    c1.constant_value = 42;
    EClassId cid1 = eg.add(c1);

    ENode c2;
    c2.is_constant = true;
    c2.constant_value = 42;
    EClassId cid2 = eg.add(c2);

    CHECK(cid1 == cid2, "two identical constant e-nodes share an e-class");

    // Different constants get different e-classes.
    ENode c3;
    c3.is_constant = true;
    c3.constant_value = 7;
    EClassId cid3 = eg.add(c3);
    CHECK(cid1 != cid3, "different constants get different e-classes");

    // Different arguments get different e-classes.
    ENode a3;
    a3.is_argument = true;
    a3.argument_name = "y";
    EClassId id3 = eg.add(a3);
    CHECK(id1 != id3, "different arguments get different e-classes");
}

// ═══════════════════════════════════════════════════════════════════════════
//  Test 2: Congruence closure — merge(a, b) makes (+ a b) ≡ (+ b a)
// ═══════════════════════════════════════════════════════════════════════════

void test_egraph_merge_congruence_closure() {
    EGraph eg;

    // Create argument e-nodes a and b.
    ENode na;
    na.is_argument = true;
    na.argument_name = "a";
    EClassId a_id = eg.add(na);

    ENode nb;
    nb.is_argument = true;
    nb.argument_name = "b";
    EClassId b_id = eg.add(nb);

    // Create (+ a b) and (+ b a).
    ENode plus_ab;
    plus_ab.opcode = Opcode::Add;
    plus_ab.children = {a_id, b_id};
    EClassId plus_ab_id = eg.add(plus_ab);

    ENode plus_ba;
    plus_ba.opcode = Opcode::Add;
    plus_ba.children = {b_id, a_id};
    EClassId plus_ba_id = eg.add(plus_ba);

    CHECK(eg.find(plus_ab_id) != eg.find(plus_ba_id),
          "before merge, (+ a b) and (+ b a) are in different e-classes");

    // Merge a and b. After rebuild, (+ a b) and (+ b a) should be
    // congruent (both have children [c, c] where c = find(a) = find(b))
    // and thus merged.
    eg.merge(a_id, b_id);

    CHECK(eg.find(a_id) == eg.find(b_id),
          "after merge, a and b are in the same e-class");
    CHECK(eg.find(plus_ab_id) == eg.find(plus_ba_id),
          "congruence closure makes (+ a b) and (+ b a) equivalent");
}

// ═══════════════════════════════════════════════════════════════════════════
//  Test 3: Lowering roundtrip — lower add(a, b), extract, verify identical
// ═══════════════════════════════════════════════════════════════════════════

void test_egraph_lowering_roundtrip() {
    auto fn = make_add_function();
    EvaluationEngine eval;

    // Lower to e-graph.
    LoweringResult lower = lower_to_egraph(*fn);
    CHECK(lower.egraph != nullptr, "lowering produces a non-null e-graph");
    CHECK(lower.return_class.has_value(), "lowering records a return e-class");
    CHECK(lower.value_to_class.count("a") > 0, "arg 'a' is in value_to_class");
    CHECK(lower.value_to_class.count("b") > 0, "arg 'b' is in value_to_class");
    CHECK(lower.value_to_class.count("r") > 0, "instruction 'r' is in value_to_class");

    // Construct rewriter, hand over the e-graph, extract.
    EgraphRewriter rw(nullptr, make_x86_64_arch());
    rw.take_egraph(std::move(lower.egraph));
    auto out = rw.extract(lower, eval);

    CHECK(out != nullptr, "extract returns a non-null function");
    if (!out) return;

    CHECK(out->blocks().size() == 1, "extracted function has 1 block");
    if (out->blocks().size() < 1) return;
    auto& entry = out->blocks().front();
    CHECK(entry->size() == 2, "extracted function has 2 instructions (add + ret)");

    if (entry->size() >= 1) {
        auto first = entry->instruction(0);
        CHECK(first->opcode() == Opcode::Add, "first instruction is add");
        CHECK(first->num_operands() == 2, "add has 2 operands");
    }
    if (entry->size() >= 2) {
        auto second = entry->instruction(1);
        CHECK(second->opcode() == Opcode::Ret, "second instruction is ret");
        CHECK(second->num_operands() == 1, "ret has 1 operand");
    }

    // The extracted function should be valid IR.
    CHECK(validate_function(*out), "extracted function is valid IR (no dangling refs)");
}

// ═══════════════════════════════════════════════════════════════════════════
//  Test 4: egraph_rewrite finds improvement on (a + 0) * 1
// ═══════════════════════════════════════════════════════════════════════════

void test_egraph_rewrite_finds_improvement() {
    auto fn = make_add_zero_mul_one_function();
    EvaluationEngine eval;
    PatternLibrary lib;  // auto-seeds builtin patterns (add_zero_elim, mul_one_elim, ...)
    auto arch = make_x86_64_arch();

    // Sanity: confirm the builtin patterns we rely on are present.
    const auto& patterns = lib.patterns();
    CHECK(patterns.count("add_zero_elim") > 0, "add_zero_elim pattern is seeded");
    CHECK(patterns.count("mul_one_elim") > 0, "mul_one_elim pattern is seeded");

    // Original instruction count: add + mul + ret = 3.
    size_t orig_count = 0;
    for (auto& b : fn->blocks()) {
        if (b) orig_count += b->size();
    }
    CHECK(orig_count == 3, "original function has 3 instructions");

    // Run egraph_rewrite.
    auto cand_opt = egraph_rewrite(*fn, lib, arch, eval);
    CHECK(cand_opt.has_value(), "egraph_rewrite returns a candidate (improvement found)");
    if (!cand_opt) return;

    auto& cand = *cand_opt;
    CHECK(cand.function != nullptr, "candidate function is non-null");
    CHECK(!cand.sound, "candidate is sound=false (needs SMT verification)");
    CHECK(cand.description == "egraph_rewrite", "candidate description is 'egraph_rewrite'");

    // The optimised function should have fewer instructions than the
    // original (ideally just `ret %a`).
    size_t cand_count = 0;
    for (auto& b : cand.function->blocks()) {
        if (b) cand_count += b->size();
    }
    CHECK(cand_count < orig_count,
          "candidate has fewer instructions than original");
    CHECK(cand_count == 1,
          "candidate is fully simplified to a single ret instruction");

    // The single instruction should be `ret %a`.
    if (cand_count == 1) {
        auto& entry = cand.function->blocks().front();
        auto inst = entry->instruction(0);
        CHECK(inst->opcode() == Opcode::Ret, "candidate's single instruction is ret");
        CHECK(inst->num_operands() == 1, "ret has 1 operand");
        if (inst->num_operands() == 1) {
            auto op = inst->operand(0);
            CHECK(op != nullptr, "ret operand is non-null");
            CHECK(op->has_name(), "ret operand is a named value");
            CHECK(op->name() == "a", "ret operand is the function arg 'a'");
        }
    }

    // The candidate should be valid IR.
    CHECK(validate_function(*cand.function),
          "candidate function is valid IR (no dangling refs)");
}

// ═══════════════════════════════════════════════════════════════════════════

int main() {
    std::cout << "=== Clunk E-graph Tests ===" << std::endl;

    std::cout << "  EGraph::add dedupes identical nodes..." << std::endl;
    test_egraph_add_dedupes_identical_nodes();

    std::cout << "  EGraph::merge congruence closure..." << std::endl;
    test_egraph_merge_congruence_closure();

    std::cout << "  EGraph lowering roundtrip..." << std::endl;
    test_egraph_lowering_roundtrip();

    std::cout << "  egraph_rewrite finds improvement..." << std::endl;
    test_egraph_rewrite_finds_improvement();

    std::cout << "\n=== Results: " << g_pass << " passed, " << g_fail << " failed ===" << std::endl;
    return g_fail > 0 ? 1 : 0;
}
