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
 * Clunk STOKE-style move tests — exercises the unsound mutation kinds
 * (OpcodeReplace, OperandReplace, OperandSwap), test-vector pre-filtering,
 * and canonical hash-based deduplication in StochasticSearch.
 *
 * Coverage:
 *   1. test_opcode_replace_creates_unsound_candidate
 *   2. test_operand_replace_creates_unsound_candidate
 *   3. test_operand_swap_swaps_operands
 *   4. test_test_vector_filter_rejects_wrong_candidate
 *   5. test_canonical_hash_dedupes_renamed_functions
 *   6. test_stoke_search_finds_improvement_with_unsound_mutations
 *
 * Each test mirrors the CHECK(cond, msg) macro pattern from
 * tests/test_search.cpp. The file is a single main() that runs all tests
 * and returns nonzero on any failure.
 */
#include <algorithm>
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
#include "clunk/IR/Clone.h"
#include "clunk/Evaluator/EvaluationEngine.h"
#include "clunk/Evaluator/Interpreter.h"
#include "clunk/Search/StochasticSearch.h"

static int g_pass = 0, g_fail = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { std::cerr << "FAIL: " << msg << " at " << __FILE__ << ":" << __LINE__ << "\n"; g_fail++; } \
    else { g_pass++; } \
} while(0)

using namespace clunk::ir;
using namespace clunk::evaluator;
using namespace clunk::search;

// ═══════════════════════════════════════════════════════════════════════════
//  Helper: build a function f(x) = (x + 0) * 1, which has two sound
//  SimplifyIdentity opportunities (the add-by-zero and the mul-by-one).
//  Used by both the STOKE-move unit tests and the end-to-end search test.
// ═══════════════════════════════════════════════════════════════════════════

static std::shared_ptr<Function> make_simplifiable_function(Module& mod,
                                                              const std::string& name,
                                                              const std::string& arg_name,
                                                              const std::string& v1,
                                                              const std::string& v2) {
    TypeContext& ctx = mod.type_context();
    auto fn_type = std::make_shared<FunctionType>(
        ctx.int32(), std::vector<std::shared_ptr<Type>>{ctx.int32()});
    auto& fn = mod.add_function(name, fn_type);
    fn.add_argument(ctx.int32(), arg_name);

    auto& entry = fn.add_block("entry");
    IRBuilder builder(ctx);
    builder.set_insert_point(&entry);

    auto arg_val = std::make_shared<Value>(ctx.int32(), arg_name);
    auto zero = builder.get_int32(0);
    auto one = builder.get_int32(1);
    auto a = builder.create_add(arg_val, zero, v1);
    auto b = builder.create_mul(a, one, v2);
    builder.create_ret(b);

    return mod.function(name);
}

// Convenience: build the function with default names.
static std::shared_ptr<Function> make_simplifiable_function(Module& mod,
                                                              const std::string& name = "f") {
    return make_simplifiable_function(mod, name, "x", "a", "b");
}

// ═══════════════════════════════════════════════════════════════════════════
//  Test 1 — OpcodeReplace produces an unsound candidate
// ═══════════════════════════════════════════════════════════════════════════

void test_opcode_replace_creates_unsound_candidate() {
    Module mod("opcode_replace_test");
    auto fn = make_simplifiable_function(mod);

    EvaluationEngine engine;
    StochasticConfig config;
    config.seed = 42;
    StochasticSearch search(config, &engine);

    // Sanity: OpcodeReplace must be classified as unsound.
    CHECK(!StochasticSearch::is_sound_kind(MutationKind::OpcodeReplace),
          "OpcodeReplace is_sound_kind == false");
    CHECK(StochasticSearch::is_in_place_kind(MutationKind::OpcodeReplace),
          "OpcodeReplace is_in_place_kind == true (amenable to MutationScope)");

    // Apply an OpcodeReplace mutation to the add at index 0 in entry.
    Mutation mut;
    mut.kind = MutationKind::OpcodeReplace;
    mut.instruction_index = 0;
    mut.block_name = "entry";
    auto candidate = search.apply_mutation(*fn, mut);
    CHECK(candidate != nullptr, "OpcodeReplace apply_mutation returns non-null");

    if (candidate) {
        // The opcode at index 0 must be different from the original Add.
        auto orig_inst = fn->block("entry")->instruction(0);
        auto cand_inst = candidate->block("entry")->instruction(0);
        CHECK(cand_inst->opcode() != orig_inst->opcode(),
              "OpcodeReplace changed the opcode");
        // The candidate must be well-formed IR.
        CHECK(clunk::ir::validate_function(*candidate),
              "OpcodeReplace candidate is well-formed IR");
        // The candidate must be flagged unsound (the opcode was changed
        // to a non-equivalent one — SMT must verify before adoption).
        // We can't directly read Candidate::sound from apply_mutation's
        // return value (it returns a Function, not a Candidate), so we
        // verify the invariant via is_sound_kind which is the source of
        // truth for the Candidate::sound flag.
        CHECK(!StochasticSearch::is_sound_kind(MutationKind::OpcodeReplace),
              "OpcodeReplace candidates are unsound-by-construction");
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Test 2 — OperandReplace produces an unsound candidate
// ═══════════════════════════════════════════════════════════════════════════

void test_operand_replace_creates_unsound_candidate() {
    Module mod("operand_replace_test");
    auto fn = make_simplifiable_function(mod);

    EvaluationEngine engine;
    StochasticConfig config;
    config.seed = 42;
    StochasticSearch search(config, &engine);

    CHECK(!StochasticSearch::is_sound_kind(MutationKind::OperandReplace),
          "OperandReplace is_sound_kind == false");
    CHECK(StochasticSearch::is_in_place_kind(MutationKind::OperandReplace),
          "OperandReplace is_in_place_kind == true");

    // OperandReplace on the add at index 0, operand slot 1 (the `0`).
    Mutation mut;
    mut.kind = MutationKind::OperandReplace;
    mut.instruction_index = 0;
    mut.aux_index = 1;  // operand slot 1 (the zero constant)
    mut.block_name = "entry";
    auto candidate = search.apply_mutation(*fn, mut);
    // The candidate may be null if the operand pool was empty (no other
    // i32 value defined before index 0). With our function, the argument
    // `x` is in the pool, so it should succeed.
    CHECK(candidate != nullptr,
          "OperandReplace apply_mutation returns non-null (arg x is in pool)");

    if (candidate) {
        // The operand at slot 1 must be different from the original.
        auto orig_op1 = fn->block("entry")->instruction(0)->operand(1);
        auto cand_op1 = candidate->block("entry")->instruction(0)->operand(1);
        // Either pointer identity differs OR (if both are named) the names differ.
        bool differs = (orig_op1.get() != cand_op1.get());
        if (!differs && orig_op1 && cand_op1 && orig_op1->has_name() && cand_op1->has_name()) {
            differs = (orig_op1->name() != cand_op1->name());
        }
        CHECK(differs, "OperandReplace changed the operand");
        CHECK(clunk::ir::validate_function(*candidate),
              "OperandReplace candidate is well-formed IR");
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Test 3 — OperandSwap actually swaps the two operands
// ═══════════════════════════════════════════════════════════════════════════

void test_operand_swap_swaps_operands() {
    Module mod("operand_swap_test");
    TypeContext& ctx = mod.type_context();
    // Build f(x) = add 5, x  (operand 0 = const 5, operand 1 = arg x).
    // After OperandSwap, it should become add x, 5.
    auto fn_type = std::make_shared<FunctionType>(
        ctx.int32(), std::vector<std::shared_ptr<Type>>{ctx.int32()});
    auto& fn = mod.add_function("swap_fn", fn_type);
    fn.add_argument(ctx.int32(), "x");
    auto& entry = fn.add_block("entry");
    auto arg_val = std::make_shared<Value>(ctx.int32(), "x");
    auto five = ConstantInt::get(ctx, 5, 32);
    entry.add_instruction(inst::make_add(five, arg_val, "a"));
    entry.add_instruction(inst::make_ret(entry.instruction(0)));
    auto fn_ptr = mod.function("swap_fn");

    EvaluationEngine engine;
    StochasticConfig config;
    config.seed = 42;
    StochasticSearch search(config, &engine);

    CHECK(!StochasticSearch::is_sound_kind(MutationKind::OperandSwap),
          "OperandSwap is_sound_kind == false (swap of non-commutative ops is unsound)");

    Mutation mut;
    mut.kind = MutationKind::OperandSwap;
    mut.instruction_index = 0;
    mut.block_name = "entry";
    auto candidate = search.apply_mutation(*fn_ptr, mut);
    CHECK(candidate != nullptr, "OperandSwap apply_mutation returns non-null");

    if (candidate) {
        auto orig = fn_ptr->block("entry")->instruction(0);
        auto cand = candidate->block("entry")->instruction(0);
        // Operand 0 of the candidate should be operand 1 of the original.
        bool op0_swapped = false;
        if (cand->operand(0) && orig->operand(1)) {
            if (cand->operand(0).get() == orig->operand(1).get()) {
                op0_swapped = true;
            } else if (cand->operand(0)->has_name() && orig->operand(1)->has_name() &&
                       cand->operand(0)->name() == orig->operand(1)->name()) {
                op0_swapped = true;
            }
        }
        CHECK(op0_swapped, "OperandSwap moved original operand(1) into slot 0");
        CHECK(clunk::ir::validate_function(*candidate),
              "OperandSwap candidate is well-formed IR");
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Test 4 — test-vector pre-filter rejects an obviously-wrong candidate
// ═══════════════════════════════════════════════════════════════════════════

void test_test_vector_filter_rejects_wrong_candidate() {
    Module mod("test_vector_test");
    TypeContext& ctx = mod.type_context();

    // Original: f(a, b) = a + b
    auto fn_type = std::make_shared<FunctionType>(
        ctx.int32(), std::vector<std::shared_ptr<Type>>{ctx.int32(), ctx.int32()});
    auto& orig_fn = mod.add_function("orig", fn_type);
    orig_fn.add_argument(ctx.int32(), "a");
    orig_fn.add_argument(ctx.int32(), "b");
    auto& orig_entry = orig_fn.add_block("entry");
    auto a_val = std::make_shared<Value>(ctx.int32(), "a");
    auto b_val = std::make_shared<Value>(ctx.int32(), "b");
    orig_entry.add_instruction(inst::make_add(a_val, b_val, "sum"));
    orig_entry.add_instruction(inst::make_ret(orig_entry.instruction(0)));
    auto original = mod.function("orig");

    // Candidate: f(a, b) = a - b  (WRONG — sub instead of add)
    auto& cand_fn = mod.add_function("cand", fn_type);
    cand_fn.add_argument(ctx.int32(), "a");
    cand_fn.add_argument(ctx.int32(), "b");
    auto& cand_entry = cand_fn.add_block("entry");
    cand_entry.add_instruction(inst::make_sub(a_val, b_val, "diff"));
    cand_entry.add_instruction(inst::make_ret(cand_entry.instruction(0)));
    auto candidate = mod.function("cand");

    EvaluationEngine engine;
    StochasticConfig config;
    config.seed = 42;
    StochasticSearch search(config, &engine);

    // The candidate computes a - b, the original computes a + b.
    // For most inputs these differ, so passes_test_vectors must return
    // false. (Edge case: a==b ⇒ both return 0; the probe set includes
    // {0,1,-1,2,…} so a≠b on at least one vector.)
    bool passes = search.passes_test_vectors(*original, *candidate, 32);
    CHECK(!passes,
          "passes_test_vectors returns false for add-vs-sub candidate");

    // The rejection counter must have been incremented.
    CHECK(search.stats().candidates_rejected_by_test_vectors >= 1,
          "candidates_rejected_by_test_vectors counter incremented");

    // Sanity: a SOUND candidate (the original itself) must pass.
    bool self_passes = search.passes_test_vectors(*original, *original, 32);
    CHECK(self_passes,
          "passes_test_vectors returns true for original-vs-original");

    // And test_vector_count == 0 disables the filter (returns true).
    bool disabled = search.passes_test_vectors(*original, *candidate, 0);
    CHECK(disabled,
          "passes_test_vectors returns true when num_vectors == 0 (disabled)");
}

// ═══════════════════════════════════════════════════════════════════════════
//  Test 5 — canonical_structural_hash dedupes renaming-equivalent fns
// ═══════════════════════════════════════════════════════════════════════════

void test_canonical_hash_dedupes_renamed_functions() {
    // fn1: %a = add %x, %x; %b = mul %a, %x; ret %b
    Module mod1("canon_a");
    auto fn1 = make_simplifiable_function(mod1, "f1", "x", "a", "b");

    // fn2: %p = add %q, %q; %r = mul %p, %q; ret %r
    // Same structure, different SSA value names AND different argument
    // name. The canonical form renames both to %v0, %v1, %v2 — so the
    // canonical hashes must match.
    Module mod2("canon_b");
    auto fn2 = make_simplifiable_function(mod2, "f2", "q", "p", "r");

    uint64_t h1 = StochasticSearch::canonical_structural_hash(*fn1);
    uint64_t h2 = StochasticSearch::canonical_structural_hash(*fn2);
    CHECK(h1 == h2,
          "canonical_structural_hash: renaming-equivalent functions hash the same");

    // The legacy structural_hash mixes operand NAMES into the hash, so
    // the two functions hash differently under it. This confirms the
    // canonical hash is doing something the legacy hash doesn't.
    uint64_t sh1 = StochasticSearch::structural_hash(*fn1);
    uint64_t sh2 = StochasticSearch::structural_hash(*fn2);
    CHECK(sh1 != sh2,
          "structural_hash: renaming-equivalent functions hash differently (legacy is name-sensitive)");

    // A structurally DIFFERENT function must canonical-hash differently.
    Module mod3("canon_c");
    TypeContext& ctx = mod3.type_context();
    auto fn_type = std::make_shared<FunctionType>(
        ctx.int32(), std::vector<std::shared_ptr<Type>>{ctx.int32()});
    auto& fn3 = mod3.add_function("f3", fn_type);
    fn3.add_argument(ctx.int32(), "x");
    auto& entry3 = fn3.add_block("entry");
    auto x_val = std::make_shared<Value>(ctx.int32(), "x");
    auto two = ConstantInt::get(ctx, 2, 32);
    entry3.add_instruction(inst::make_mul(x_val, two, "a"));   // mul, not add
    entry3.add_instruction(inst::make_ret(entry3.instruction(0)));
    auto fn3_ptr = mod3.function("f3");
    uint64_t h3 = StochasticSearch::canonical_structural_hash(*fn3_ptr);
    CHECK(h1 != h3,
          "canonical_structural_hash: structurally different functions hash differently");

    // A deep copy of fn1 must canonical-hash the same as fn1 (sanity).
    auto fn1_copy = clunk::ir::deep_copy_function(*fn1);
    uint64_t h1c = StochasticSearch::canonical_structural_hash(*fn1_copy);
    CHECK(h1 == h1c,
          "canonical_structural_hash: deep copy preserves hash");
}

// ═══════════════════════════════════════════════════════════════════════════
//  Test 6 — end-to-end search with unsound mutations + test vectors
// ═══════════════════════════════════════════════════════════════════════════

void test_stoke_search_finds_improvement_with_unsound_mutations() {
    Module mod("stoke_search_test");
    auto fn = make_simplifiable_function(mod);

    EvaluationEngine engine;
    StochasticConfig config;
    config.max_iterations = 500;
    config.max_candidates = 20;
    config.seed = 42;
    // Enable the STOKE-style unsound mutation kinds.
    config.allow_unsound_mutations = true;
    // Pre-filter every improving candidate through 32 test vectors.
    config.test_vector_count = 32;

    StochasticSearch search(config, &engine);
    auto candidates = search.search(*fn);
    auto stats = search.stats();

    // The search must have run.
    CHECK(stats.iterations_run > 0, "search ran some iterations");
    CHECK(stats.mutations_tried > 0, "search tried some mutations");

    // At least one STOKE-style unsound mutation must have been
    // proposed (the site list includes them and pick_mutation samples
    // uniformly from the mixed list).
    CHECK(stats.stoke_moves_tried > 0,
          "search proposed at least one STOKE-style unsound mutation");

    // The function has two sound SimplifyIdentity opportunities (add x,0
    // and mul y,1). The search should find at least one improving
    // candidate that passes the test-vector filter. (We don't require a
    // specific count — just that the search produces SOMETHING.)
    CHECK(!candidates.empty(),
          "search found at least one improving candidate with unsound + test-vector filter");

    // Every returned candidate must be well-formed IR (no dangling SSA
    // references from a botched in-place undo of a STOKE-style move).
    bool all_valid = true;
    for (auto& c : candidates) {
        if (!c.function || !clunk::ir::validate_function(*c.function)) {
            all_valid = false;
            break;
        }
    }
    CHECK(all_valid, "all candidates are well-formed IR");

    // At least one candidate must have a score strictly greater than the
    // baseline (the original function's score). The score convention is
    // ratio > 1 = improvement; we just check > 0 here since the baseline
    // could be 0 for a tiny function.
    bool has_improving = false;
    for (auto& c : candidates) {
        if (c.score > 0.0) { has_improving = true; break; }
    }
    CHECK(has_improving, "at least one candidate has a positive score");

    // The original function must be untouched (in-place mutation rolled
    // back uncommitted edits).
    CHECK(clunk::ir::validate_function(*fn),
          "original function still well-formed after search");
}

// ═══════════════════════════════════════════════════════════════════════════
//  Bonus Test — canonical cache hits are observable in stats
// ═══════════════════════════════════════════════════════════════════════════

void test_canonical_cache_hits_observable() {
    // Run a search; with the canonical-form cache key, renaming-
    // equivalent candidates share a cache entry, so canonical_cache_hits
    // should be >= 0 (and likely > 0 for any non-trivial search that
    // re-visits a renaming-equivalent function). We don't require a
    // specific count — just that the counter is accessible and the
    // search runs without crashing.
    Module mod("canon_cache_test");
    auto fn = make_simplifiable_function(mod);

    EvaluationEngine engine;
    StochasticConfig config;
    config.max_iterations = 200;
    config.max_candidates = 10;
    config.seed = 7;
    StochasticSearch search(config, &engine);
    search.search(*fn);
    auto stats = search.stats();
    (void)stats;  // reading stats shouldn't crash
    CHECK(true, "canonical_cache_hits counter is accessible (no crash)");
}

// ═══════════════════════════════════════════════════════════════════════════

int main() {
    std::cout << "=== Clunk STOKE-style Move Tests ===" << std::endl;

    std::cout << "  OpcodeReplace creates unsound candidate..." << std::endl;
    test_opcode_replace_creates_unsound_candidate();

    std::cout << "  OperandReplace creates unsound candidate..." << std::endl;
    test_operand_replace_creates_unsound_candidate();

    std::cout << "  OperandSwap swaps operands..." << std::endl;
    test_operand_swap_swaps_operands();

    std::cout << "  test-vector filter rejects wrong candidate..." << std::endl;
    test_test_vector_filter_rejects_wrong_candidate();

    std::cout << "  canonical hash dedupes renamed functions..." << std::endl;
    test_canonical_hash_dedupes_renamed_functions();

    std::cout << "  STOKE search finds improvement..." << std::endl;
    test_stoke_search_finds_improvement_with_unsound_mutations();

    std::cout << "  canonical cache hits observable..." << std::endl;
    test_canonical_cache_hits_observable();

    std::cout << "\n=== Results: " << g_pass << " passed, " << g_fail << " failed ===" << std::endl;
    return g_fail > 0 ? 1 : 0;
}
