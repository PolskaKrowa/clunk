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
 * Clunk DataflowPrune Tests — dead-code elimination, unreachable-block
 * removal, known-bits-driven simplification, and the combined
 * prune_dataflow() driver used by Pipeline::dataflow_prune_phase.
 */
#include <iostream>
#include <memory>
#include <string>

#include "clunk/Evaluator/Interpreter.h"
#include "clunk/IR/DataflowPrune.h"
#include "clunk/IR/Module.h"
#include "clunk/Parser/IRParser.h"

static int g_pass = 0, g_fail = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { std::cerr << "FAIL: " << msg << " at " << __FILE__ << ":" << __LINE__ << "\n"; g_fail++; } \
    else { g_pass++; } \
} while(0)

using namespace clunk;
using namespace clunk::ir;

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

static size_t instruction_count(const Function& fn) {
    size_t n = 0;
    for (auto& bb : fn.blocks()) n += bb->size();
    return n;
}

// ═══════════════════════════════════════════════════════════════════════
//  eliminate_dead_code
// ═══════════════════════════════════════════════════════════════════════

static void test_dce_removes_unused_computation() {
    auto mod = parse(R"(
define i32 @f(i32 %a, i32 %b) {
entry:
  %dead = add i32 %a, %b
  %also_dead = mul i32 %dead, 3
  ret i32 %a
}
)");
    auto fn = mod->function("f");
    CHECK(fn != nullptr, "parses");
    if (!fn) return;
    auto out = eliminate_dead_code(*fn);
    CHECK(out != nullptr, "DCE fires on unused computation");
    if (!out) return;
    CHECK(instruction_count(*out) == 1, "only the ret survives");
    for (int64_t a : {0, 5, -3}) {
        for (int64_t b : {1, -1, 100}) {
            auto r0 = evaluator::Interpreter::interpret(*fn, {a, b});
            auto r1 = evaluator::Interpreter::interpret(*out, {a, b});
            CHECK(r0 && r1 && *r0 == *r1, "DCE preserves semantics");
        }
    }
}

static void test_dce_keeps_side_effects() {
    auto mod = parse(R"(
define i32 @f(i32 %v) {
entry:
  %p = alloca i32
  store i32 %v, ptr %p
  %unused = add i32 %v, 1
  ret i32 %v
}
)");
    auto fn = mod->function("f");
    CHECK(fn != nullptr, "parses");
    if (!fn) return;
    auto out = eliminate_dead_code(*fn);
    CHECK(out != nullptr, "DCE fires (removes %unused)");
    if (!out) return;
    CHECK(count_opcode(*out, Opcode::Store) == 1, "store survives (side effect)");
    CHECK(count_opcode(*out, Opcode::Alloca) == 1, "alloca survives (feeds the store)");
    CHECK(count_opcode(*out, Opcode::Add) == 0, "unused add is removed");
}

static void test_dce_noop_returns_null() {
    auto mod = parse(R"(
define i32 @f(i32 %a) {
entry:
  %r = add i32 %a, 1
  ret i32 %r
}
)");
    auto fn = mod->function("f");
    CHECK(fn != nullptr, "parses");
    if (!fn) return;
    auto out = eliminate_dead_code(*fn);
    CHECK(out == nullptr, "DCE is a no-op on a function with nothing dead");
}

// ═══════════════════════════════════════════════════════════════════════
//  remove_unreachable_blocks
// ═══════════════════════════════════════════════════════════════════════

static void test_removes_unreachable_block() {
    auto mod = parse(R"(
define i32 @f(i32 %a) {
entry:
  br label %live

ghost:
  %g = add i32 %a, 999
  ret i32 %g

live:
  ret i32 %a
}
)");
    auto fn = mod->function("f");
    CHECK(fn != nullptr, "parses");
    CHECK(fn && fn->blocks().size() == 3, "fixture has 3 blocks");
    if (!fn) return;
    auto out = remove_unreachable_blocks(*fn);
    CHECK(out != nullptr, "unreachable block removed");
    if (!out) return;
    CHECK(out->blocks().size() == 2, "ghost block is gone");
    CHECK(out->block("ghost") == nullptr, "ghost specifically is gone");
    CHECK(out->block("live") != nullptr, "live block survives");
    for (int64_t a : {0, 5, -3}) {
        auto r0 = evaluator::Interpreter::interpret(*fn, {a});
        auto r1 = evaluator::Interpreter::interpret(*out, {a});
        CHECK(r0 && r1 && *r0 == *r1, "unreachable-block removal preserves semantics");
    }
}

static void test_phi_fixup_on_removed_predecessor() {
    // `alt` is unreachable (nothing branches to it); the phi in `join`
    // must have its incoming edge from `alt` dropped, not just left
    // dangling / crash on removal.
    auto mod = parse(R"(
define i32 @f(i32 %a) {
entry:
  br label %main

alt:
  br label %join

main:
  br label %join

join:
  %p = phi i32 [%a, %main], [999, %alt]
  ret i32 %p
}
)");
    auto fn = mod->function("f");
    CHECK(fn != nullptr, "parses");
    if (!fn) return;
    auto out = remove_unreachable_blocks(*fn);
    CHECK(out != nullptr, "unreachable block removed");
    if (!out) return;
    CHECK(out->block("alt") == nullptr, "alt is gone");
    auto join = out->block("join");
    CHECK(join != nullptr, "join survives");
    if (!join) return;
    std::shared_ptr<Instruction> phi;
    for (auto& inst : join->instructions()) {
        if (inst->opcode() == Opcode::Phi) phi = inst;
    }
    CHECK(phi != nullptr, "phi survives");
    if (!phi) return;
    CHECK(phi->num_operands() == 1, "phi's dead incoming edge (from alt) was dropped");
    for (int64_t a : {0, 5, -3}) {
        auto r0 = evaluator::Interpreter::interpret(*fn, {a});
        auto r1 = evaluator::Interpreter::interpret(*out, {a});
        CHECK(r0 && r1 && *r0 == *r1, "phi fixup preserves semantics");
    }
}

static void test_all_reachable_returns_null() {
    auto mod = parse(R"(
define i32 @f(i1 %c) {
entry:
  br i1 %c, label %t, label %e
t:
  ret i32 1
e:
  ret i32 0
}
)");
    auto fn = mod->function("f");
    CHECK(fn != nullptr, "parses");
    if (!fn) return;
    auto out = remove_unreachable_blocks(*fn);
    CHECK(out == nullptr, "no-op when every block is reachable");
}

// ═══════════════════════════════════════════════════════════════════════
//  simplify_known_bits
// ═══════════════════════════════════════════════════════════════════════

static void test_simplify_folds_constant_mask() {
    auto mod = parse(R"(
define i32 @f(i32 %a) {
entry:
  %m = and i32 %a, 0
  %r = add i32 %m, 5
  ret i32 %r
}
)");
    auto fn = mod->function("f");
    CHECK(fn != nullptr, "parses");
    if (!fn) return;
    auto out = simplify_known_bits(*fn);
    CHECK(out != nullptr, "known-bits simplification fires");
    if (!out) return;
    CHECK(count_opcode(*out, Opcode::And) == 0, "the and is folded away");
    for (int64_t a : {0, 5, -3}) {
        auto r0 = evaluator::Interpreter::interpret(*fn, {a});
        auto r1 = evaluator::Interpreter::interpret(*out, {a});
        CHECK(r0 && r1 && *r0 == *r1, "known-bits fold preserves semantics");
    }
}

static void test_simplify_folds_branch_condition() {
    auto mod = parse(R"(
define i32 @f(i32 %a) {
entry:
  %c = icmp eq i32 5, 5
  br i1 %c, label %t, label %e
t:
  ret i32 %a
e:
  ret i32 999
}
)");
    auto fn = mod->function("f");
    CHECK(fn != nullptr, "parses");
    if (!fn) return;
    auto out = simplify_known_bits(*fn);
    CHECK(out != nullptr, "branch-condition fold fires");
    if (!out) return;
    auto term = out->entry_block()->terminator();
    CHECK(term && term->opcode() == Opcode::Br && term->num_operands() == 0,
          "conditional branch became unconditional");
    for (int64_t a : {0, 5, -3}) {
        auto r0 = evaluator::Interpreter::interpret(*fn, {a});
        auto r1 = evaluator::Interpreter::interpret(*out, {a});
        CHECK(r0 && r1 && *r0 == *r1, "branch fold preserves semantics");
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  prune_dataflow: the combined chain Pipeline actually calls
// ═══════════════════════════════════════════════════════════════════════

static void test_prune_dataflow_chains_all_three() {
    // A branch condition that's a compile-time constant: the fold should
    // cascade — known-bits turns the br unconditional, that makes `e`
    // unreachable, and `e`'s only-used-there computation goes with it.
    auto mod = parse(R"(
define i32 @f(i32 %a) {
entry:
  %c = icmp eq i32 5, 5
  br i1 %c, label %t, label %e
t:
  ret i32 %a
e:
  %junk = mul i32 %a, 1234
  ret i32 %junk
}
)");
    auto fn = mod->function("f");
    CHECK(fn != nullptr, "parses");
    if (!fn) return;
    PruneStats stats;
    auto out = prune_dataflow(*fn, &stats);
    CHECK(out != nullptr, "prune_dataflow fires");
    if (!out) return;
    CHECK(out->blocks().size() == 2, "unreachable block 'e' is gone (entry + t)");
    CHECK(out->block("e") == nullptr, "'e' specifically is gone");
    CHECK(count_opcode(*out, Opcode::Mul) == 0, "the dead mul inside 'e' is gone too");
    CHECK(stats.comparisons_folded >= 1, "stats record the icmp fold");
    CHECK(stats.branches_simplified >= 1, "stats record the branch simplification");
    CHECK(stats.blocks_removed >= 1, "stats record the block removal");
    for (int64_t a : {0, 5, -3, 42}) {
        auto r0 = evaluator::Interpreter::interpret(*fn, {a});
        auto r1 = evaluator::Interpreter::interpret(*out, {a});
        CHECK(r0 && r1 && *r0 == *r1, "prune_dataflow preserves semantics end to end");
    }
}

static void test_prune_dataflow_noop_returns_null() {
    auto mod = parse(R"(
define i32 @f(i32 %a, i32 %b) {
entry:
  %r = add i32 %a, %b
  ret i32 %r
}
)");
    auto fn = mod->function("f");
    CHECK(fn != nullptr, "parses");
    if (!fn) return;
    auto out = prune_dataflow(*fn);
    CHECK(out == nullptr, "prune_dataflow is a no-op on already-clean code");
}

static void test_prune_dataflow_preserves_loop_semantics() {
    // Sanity check on a function with a genuine back-edge / phi — nothing
    // here should be foldable/prunable, and the pass must not corrupt it.
    auto mod = parse(R"(
define i32 @loop_test(i32 %n) {
entry:
  br label %loop

loop:
  %i = phi i32 [0, %entry], [%next, %loop]
  %next = add i32 %i, 1
  %cond = icmp slt i32 %next, %n
  br i1 %cond, label %loop, label %exit

exit:
  ret i32 %i
}
)");
    auto fn = mod->function("loop_test");
    CHECK(fn != nullptr, "parses");
    if (!fn) return;
    auto out = prune_dataflow(*fn);
    // Whether or not anything fires, the function must still behave
    // identically for a range of inputs.
    for (int64_t n : {0, 1, 5, 10}) {
        auto r0 = evaluator::Interpreter::interpret(*fn, {n});
        auto r1 = out ? evaluator::Interpreter::interpret(*out, {n}) : r0;
        CHECK(r0 && r1 && *r0 == *r1, "loop function semantics preserved (or untouched)");
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  simplify_cse
// ═══════════════════════════════════════════════════════════════════════

static void test_cse_deduplicates_identical_add() {
    auto mod = parse(R"(
define i32 @f(i32 %a, i32 %b) {
entry:
  %x = add i32 %a, %b
  %y = add i32 %a, %b
  %z = mul i32 %x, %y
  ret i32 %z
}
)");
    auto fn = mod->function("f");
    CHECK(fn != nullptr, "parses");
    if (!fn) return;

    PruneStats stats;
    auto out = simplify_cse(*fn, &stats);
    CHECK(out != nullptr, "CSE fires on an exact same-block duplicate");
    if (!out) return;
    CHECK(stats.subexpressions_eliminated == 1, "exactly one duplicate eliminated");
    CHECK(count_opcode(*out, Opcode::Add) == 1, "only one add survives");

    for (int64_t a : {0, 1, -3, 100}) {
        for (int64_t b : {0, -1, 7, 50}) {
            auto r0 = evaluator::Interpreter::interpret(*fn, {a, b});
            auto r1 = evaluator::Interpreter::interpret(*out, {a, b});
            CHECK(r0 && r1 && *r0 == *r1, "CSE preserves semantics");
        }
    }
}

static void test_cse_is_block_local_only() {
    // The exact same computation appears in two DIFFERENT blocks. CSE in
    // this codebase is deliberately block-local (see header comment —
    // no dominator tree available, and same-block scope needs none), so
    // this must NOT be merged across blocks.
    auto mod = parse(R"(
define i32 @f(i32 %a, i32 %b, i1 %c) {
entry:
  br i1 %c, label %t, label %e

t:
  %x = add i32 %a, %b
  ret i32 %x

e:
  %y = add i32 %a, %b
  ret i32 %y
}
)");
    auto fn = mod->function("f");
    CHECK(fn != nullptr, "parses");
    if (!fn) return;
    auto out = simplify_cse(*fn);
    CHECK(out == nullptr, "no cross-block CSE: each add stays in its own block");
}

static void test_cse_leaves_loads_alone() {
    // Loads are excluded from CSE (they need alias analysis — that's
    // MemOpt's job, see clunk/Search/MemOpt.h's redundant-load
    // elimination). Two textually-identical loads must be left as-is by
    // THIS pass even with no intervening store.
    auto mod = parse(R"(
define i32 @f(ptr %p) {
entry:
  %x = load i32, ptr %p
  %y = load i32, ptr %p
  %z = add i32 %x, %y
  ret i32 %z
}
)");
    auto fn = mod->function("f");
    CHECK(fn != nullptr, "parses");
    if (!fn) return;
    auto out = simplify_cse(*fn);
    CHECK(out == nullptr, "simplify_cse never touches loads (out of its scope)");
}

static void test_cse_respects_differing_predicates() {
    // Same opcode/operands but a different comparison predicate must NOT
    // be treated as a duplicate.
    auto mod = parse(R"(
define i32 @f(i32 %a, i32 %b) {
entry:
  %lt = icmp slt i32 %a, %b
  %gt = icmp sgt i32 %a, %b
  %lt2 = zext i1 %lt to i32
  %gt2 = zext i1 %gt to i32
  %r = add i32 %lt2, %gt2
  ret i32 %r
}
)");
    auto fn = mod->function("f");
    CHECK(fn != nullptr, "parses");
    if (!fn) return;
    auto out = simplify_cse(*fn);
    CHECK(out == nullptr, "icmp slt and icmp sgt are not duplicates of each other");
}

static void test_cse_chains_through_dce_in_prune_dataflow() {
    // Once the duplicate is dropped by CSE, it becomes genuinely dead
    // (nothing else in the original references it either) — confirms
    // simplify_cse and eliminate_dead_code compose correctly through the
    // combined prune_dataflow() driver.
    auto mod = parse(R"(
define i32 @f(i32 %a, i32 %b) {
entry:
  %x = add i32 %a, %b
  %y = add i32 %a, %b
  ret i32 %x
}
)");
    auto fn = mod->function("f");
    CHECK(fn != nullptr, "parses");
    if (!fn) return;
    PruneStats stats;
    auto out = prune_dataflow(*fn, &stats);
    CHECK(out != nullptr, "prune_dataflow fires");
    if (!out) return;
    CHECK(instruction_count(*out) == 2, "one add + the ret; the duplicate is gone entirely");
    CHECK(stats.subexpressions_eliminated == 1, "prune_dataflow reports the CSE stat too");
}

int main() {
    test_dce_removes_unused_computation();
    test_dce_keeps_side_effects();
    test_dce_noop_returns_null();

    test_removes_unreachable_block();
    test_phi_fixup_on_removed_predecessor();
    test_all_reachable_returns_null();

    test_simplify_folds_constant_mask();
    test_simplify_folds_branch_condition();

    test_cse_deduplicates_identical_add();
    test_cse_is_block_local_only();
    test_cse_leaves_loads_alone();
    test_cse_respects_differing_predicates();
    test_cse_chains_through_dce_in_prune_dataflow();

    test_prune_dataflow_chains_all_three();
    test_prune_dataflow_noop_returns_null();
    test_prune_dataflow_preserves_loop_semantics();

    std::cout << "\n=== Results: " << g_pass << " passed, " << g_fail << " failed ===\n";
    return g_fail == 0 ? 0 : 1;
}
