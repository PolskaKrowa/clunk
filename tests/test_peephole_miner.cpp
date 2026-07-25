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
 * Clunk Peephole Miner tests — SMT-verified enumerative superoptimisation.
 *
 * Verifies that the miner:
 *   - finds a strictly-cheaper, SMT-EQUIVALENT replacement for a suboptimal
 *     integer function (x*2*2  ->  x<<2),
 *   - re-verifies as Equivalent (independent SMT check of the emitted win),
 *   - leaves an already-optimal function alone (no false "win"),
 *   - skips ineligible functions (memory/FP/loops) rather than guessing,
 *   - produces a well-formed OptimisationPattern.
 */
#include <cstdio>
#include <iostream>
#include <memory>
#include <string>

#include "clunk/Parser/IRParser.h"
#include "clunk/IR/Clone.h"
#include "clunk/Evaluator/EvaluationEngine.h"
#include "clunk/Search/PeepholeMiner.h"
#include "clunk/Search/SMTVerifier.h"
#include "clunk/Pattern/PatternLibrary.h"

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

// x * 2 * 2  ==  x << 2. Two muls (expensive) reducible to one shift.
static const char* kSubopt = R"(
define i32 @quad(i32 %x) {
entry:
  %a = mul i32 %x, 2
  %b = mul i32 %a, 2
  ret i32 %b
}
)";

// Already tight: x << 3. Nothing cheaper is equivalent.
static const char* kOptimal = R"(
define i32 @sh3(i32 %x) {
entry:
  %r = shl i32 %x, 3
  ret i32 %r
}
)";

// Memory op — must be skipped (SMT can't model it soundly).
static const char* kMemory = R"(
define i32 @ld(i32* %p) {
entry:
  %v = load i32, i32* %p
  ret i32 %v
}
)";

void test_miner_finds_and_proves_win() {
    auto fn = parse_fn(kSubopt, "quad");
    CHECK(fn != nullptr, "parsed suboptimal source");
    if (!fn) return;

    evaluator::EvaluationEngine engine;
    search::MinerConfig cfg;
    cfg.max_length = 2;
    search::PeepholeMiner miner(&engine, cfg);

    auto win = miner.mine_function(*fn);
    CHECK(win.has_value(), "miner found a cheaper equivalent for x*2*2");
    if (!win) return;

    CHECK(win->replacement_score > win->source_score,
          "replacement is strictly cheaper than the source");
    CHECK(win->replacement != nullptr && win->replacement->blocks().size() == 1,
          "replacement is a single-block function");

    // Independent re-verification: the emitted win must PROVE equivalent to
    // the original (not just to whatever the miner encoded).
    search::SMTVerifier verifier;
    if (search::SMTVerifier::is_z3_available()) {
        auto res = verifier.verify(*fn, *win->replacement);
        CHECK(res.status == search::VerificationResult::Equivalent,
              "emitted replacement re-verifies as Equivalent");
    } else {
        std::cerr << "  (z3 unavailable — skipping independent re-verification)\n";
    }

    // The replacement should be shorter than the source (2 ops -> 1).
    size_t repl_ops = win->replacement->blocks().front()->size();  // incl. ret
    CHECK(repl_ops < fn->blocks().front()->size(),
          "replacement has fewer instructions than the source");

    // Pattern conversion is well-formed.
    pattern::ArchDescriptor arch;
    auto pat = search::PeepholeMiner::to_pattern(*win, arch);
    CHECK(!pat.id.empty() && pat.source_function && pat.replacement_function,
          "to_pattern produces a populated OptimisationPattern");
    CHECK(!pat.replacement_ir.empty(), "pattern renders replacement IR text");
}

// (x ^ C) ^ C  ==  x. The true optimum is a bare `ret %x` (length 0), not a
// 1-op stand-in like `add %x, 0`. Exercises length-0 synthesis + the
// representation-independent concrete pre-filter.
void test_miner_reduces_identity_to_passthrough() {
    const char* ir = R"(
define i32 @xid(i32 %x) {
entry:
  %a = xor i32 %x, 255
  %b = xor i32 %a, 255
  ret i32 %b
}
)";
    auto fn = parse_fn(ir, "xid");
    CHECK(fn != nullptr, "parsed (x^C)^C");
    if (!fn) return;

    evaluator::EvaluationEngine engine;
    search::PeepholeMiner miner(&engine, {});
    auto win = miner.mine_function(*fn);
    CHECK(win.has_value(), "mined (x^C)^C -> x");
    if (!win) return;

    // Replacement block is just the terminator: a pure pass-through.
    auto& rb = win->replacement->blocks().front();
    CHECK(rb->size() == 1, "replacement is length-0 (only the ret)");
    auto ret = rb->instruction(0);
    CHECK(ret && ret->opcode() == ir::Opcode::Ret && ret->num_operands() == 1,
          "replacement terminator is a ret");
    if (ret && ret->num_operands() == 1) {
        auto v = ret->operand(0);
        CHECK(v && v->has_name() && v->name() == "x",
              "replacement returns the argument directly (ret %x)");
    }
}

// udiv by a power of two reduces to a logical shift — a source containing a
// (previously ineligible) div, proved equivalent to a shift. -O3 does this too,
// but it exercises div-family source eligibility + the shift the enumerator
// already emits.
void test_miner_reduces_udiv_to_shift() {
    const char* ir = R"(
define i32 @ud8(i32 %x) {
entry:
  %r = udiv i32 %x, 8
  ret i32 %r
}
)";
    auto fn = parse_fn(ir, "ud8");
    CHECK(fn != nullptr, "parsed udiv source");
    if (!fn) return;

    evaluator::EvaluationEngine engine;
    search::MinerConfig cfg;
    cfg.max_length = 1;
    search::PeepholeMiner miner(&engine, cfg);
    auto win = miner.mine_function(*fn);
    CHECK(win.has_value(), "mined udiv x,8 -> cheaper equivalent");
    if (!win) return;

    // The replacement must be div-free (that's the whole point) and re-verify.
    bool has_div = false;
    for (auto& in : win->replacement->blocks().front()->instructions()) {
        auto o = in->opcode();
        if (o == ir::Opcode::UDiv || o == ir::Opcode::SDiv ||
            o == ir::Opcode::URem || o == ir::Opcode::SRem) has_div = true;
    }
    CHECK(!has_div, "replacement contains no division");
    if (search::SMTVerifier::is_z3_available()) {
        search::SMTVerifier v;
        CHECK(v.verify(*fn, *win->replacement).status ==
                  search::VerificationResult::Equivalent,
              "udiv replacement re-verifies as Equivalent");
    }
}

// Compare-select synthesis: a signed-max written with a redundant `add x, 0`.
// The optimum drops the add, leaving `select(icmp ..., x, y)` — a shape blind
// enumeration can't reach, so this exercises the targeted synthesiser AND
// icmp/select source eligibility.
void test_miner_synthesises_compare_select() {
    const char* ir = R"(
define i32 @maxx(i32 %x, i32 %y) {
entry:
  %t = add i32 %x, 0
  %c = icmp sgt i32 %t, %y
  %r = select i1 %c, i32 %t, i32 %y
  ret i32 %r
}
)";
    auto fn = parse_fn(ir, "maxx");
    CHECK(fn != nullptr, "parsed compare-select source");
    if (!fn) return;

    evaluator::EvaluationEngine engine;
    search::MinerConfig cfg;
    cfg.max_length = 2;
    search::PeepholeMiner miner(&engine, cfg);
    auto win = miner.mine_function(*fn);
    CHECK(win.has_value(), "mined a cheaper equivalent for the redundant smax");
    if (!win) return;

    CHECK(win->replacement_score > win->source_score,
          "compare-select replacement is strictly cheaper");
    // The win must itself be a compare-select (proves the synthesiser produced
    // it, not some coincidental binop form).
    bool has_select = false, has_icmp = false;
    for (auto& in : win->replacement->blocks().front()->instructions()) {
        if (in->opcode() == ir::Opcode::Select) has_select = true;
        if (in->opcode() == ir::Opcode::ICmp) has_icmp = true;
    }
    CHECK(has_select && has_icmp, "replacement is an icmp+select (min/max shape)");
    if (search::SMTVerifier::is_z3_available()) {
        search::SMTVerifier v;
        CHECK(v.verify(*fn, *win->replacement).status ==
                  search::VerificationResult::Equivalent,
              "compare-select replacement re-verifies as Equivalent");
    }
}

// An icmp/select source whose condition is a tautology collapses to a
// pass-through: select(icmp eq x, x, a, b) == a. Exercises ingesting icmp+select
// sources and simplifying them to length 0.
void test_miner_simplifies_redundant_select() {
    const char* ir = R"(
define i32 @seleq(i32 %x, i32 %a, i32 %b) {
entry:
  %c = icmp eq i32 %x, %x
  %r = select i1 %c, i32 %a, i32 %b
  ret i32 %r
}
)";
    auto fn = parse_fn(ir, "seleq");
    CHECK(fn != nullptr, "parsed tautological-select source");
    if (!fn) return;

    evaluator::EvaluationEngine engine;
    search::PeepholeMiner miner(&engine, {});
    auto win = miner.mine_function(*fn);
    CHECK(win.has_value(), "mined select(icmp eq x,x, a, b) -> a");
    if (!win) return;

    auto& rb = win->replacement->blocks().front();
    CHECK(rb->size() == 1, "replacement is length-0 (only the ret)");
    if (rb->size() == 1) {
        auto ret = rb->instruction(0);
        auto v = (ret && ret->num_operands() == 1) ? ret->operand(0) : nullptr;
        CHECK(v && v->has_name() && v->name() == "a",
              "replacement returns %a directly");
    }
}

// Multi-block: a reducible integer slice (x*2*2 -> x<<2) buried in one arm of
// a diamond CFG with a phi. mine_and_rewrite must lift the slice, prove it, and
// splice the win back, leaving the branches / phi intact — and the whole
// rewrite re-verifies Equivalent to the original (the miner's own gate).
void test_miner_rewrites_multiblock_slice() {
    const char* ir = R"(
define i32 @f(i32 %x, i1 %c) {
entry:
  br i1 %c, label %then, label %else
then:
  %a = mul i32 %x, 2
  %b = mul i32 %a, 2
  br label %merge
else:
  %d = add i32 %x, 7
  br label %merge
merge:
  %r = phi i32 [ %b, %then ], [ %d, %else ]
  ret i32 %r
}
)";
    auto fn = parse_fn(ir, "f");
    CHECK(fn != nullptr, "parsed multi-block source");
    if (!fn) return;
    CHECK(fn->blocks().size() == 4, "source is genuinely multi-block");

    evaluator::EvaluationEngine engine;
    search::MinerConfig cfg;
    cfg.max_length = 2;
    search::PeepholeMiner miner(&engine, cfg);

    auto out = miner.mine_and_rewrite(*fn);
    CHECK(out.has_value(), "mine_and_rewrite optimised a multi-block function");
    if (!out) return;

    // Structure preserved: still 4 blocks, phi still there.
    CHECK((*out)->blocks().size() == 4, "rewrite keeps all four blocks");
    CHECK(clunk::ir::validate_function(**out), "rewrite is well-formed SSA");

    // The two muls in %then collapsed to strictly fewer integer ops.
    auto count_ops = [](const std::shared_ptr<ir::Function>& f) {
        size_t muls = 0, total = 0;
        for (auto& b : f->blocks())
            for (auto& in : b->instructions()) {
                if (!in->is_terminator()) ++total;
                if (in->opcode() == ir::Opcode::Mul) ++muls;
            }
        return std::pair<size_t, size_t>{muls, total};
    };
    auto before = count_ops(fn);
    auto after = count_ops(*out);
    CHECK(after.first < before.first, "a multiply was eliminated from the slice");
    CHECK(after.second < before.second, "rewrite has fewer instructions");

    // The returned rewrite is guaranteed by the miner's own gate to be
    // SMT-Equivalent to the original; re-verify independently when Z3 is up.
    if (search::SMTVerifier::is_z3_available()) {
        search::SMTVerifier v;
        CHECK(v.verify(*fn, **out).status == search::VerificationResult::Equivalent,
              "whole rewritten multi-block function re-verifies as Equivalent");
    }
    CHECK(miner.stats().rewrites_applied >= 1, "at least one slice was spliced");
}

// mine_and_rewrite must leave a multi-block function with no reducible slice
// untouched (no false rewrite).
void test_miner_multiblock_no_false_rewrite() {
    const char* ir = R"(
define i32 @g(i32 %x, i1 %c) {
entry:
  br i1 %c, label %t, label %e
t:
  %a = shl i32 %x, 3
  br label %m
e:
  br label %m
m:
  %r = phi i32 [ %a, %t ], [ %x, %e ]
  ret i32 %r
}
)";
    auto fn = parse_fn(ir, "g");
    CHECK(fn != nullptr, "parsed already-tight multi-block source");
    if (!fn) return;
    evaluator::EvaluationEngine engine;
    search::PeepholeMiner miner(&engine, {});
    auto out = miner.mine_and_rewrite(*fn);
    CHECK(!out.has_value(), "no false rewrite on a tight multi-block function");
}

void test_miner_leaves_optimal_alone() {
    auto fn = parse_fn(kOptimal, "sh3");
    CHECK(fn != nullptr, "parsed optimal source");
    if (!fn) return;

    evaluator::EvaluationEngine engine;
    search::PeepholeMiner miner(&engine, {});
    auto win = miner.mine_function(*fn);
    CHECK(!win.has_value(), "miner reports no win on already-optimal x<<3");
}

void test_miner_skips_ineligible() {
    auto fn = parse_fn(kMemory, "ld");
    CHECK(fn != nullptr, "parsed memory function");
    if (!fn) return;

    evaluator::EvaluationEngine engine;
    search::PeepholeMiner miner(&engine, {});
    auto win = miner.mine_function(*fn);
    CHECK(!win.has_value(), "miner skips functions with memory ops");
    CHECK(miner.stats().eligible == 0, "memory function is not counted eligible");
}

// A mined pattern must survive save -> load and still match/apply — the
// payoff path. (Regression for PatternLibrary::load not reparsing IR text
// into functions, and the line-based format not surviving multi-line IR.)
void test_mined_pattern_persists_and_applies() {
    auto fn = parse_fn(kSubopt, "quad");
    CHECK(fn != nullptr, "parsed source for persistence test");
    if (!fn) return;

    evaluator::EvaluationEngine engine;
    search::PeepholeMiner miner(&engine, {});
    auto win = miner.mine_function(*fn);
    CHECK(win.has_value(), "mined a win to persist");
    if (!win) return;

    pattern::ArchDescriptor arch;
    arch.name = "x86_64";
    pattern::PatternLibrary lib;
    lib.add_pattern(search::PeepholeMiner::to_pattern(*win, arch));

    const std::string path = "test_mined_roundtrip.tmp";
    CHECK(lib.save(path), "saved pattern library");

    // Fresh library loaded from disk — the mined pattern must come back as a
    // usable function, not just text.
    pattern::PatternLibrary lib2;
    CHECK(lib2.load(path), "loaded pattern library");

    auto matches = lib2.match(*fn, arch);
    bool found_mined = false;
    for (auto& m : matches)
        if (m.pattern_id.rfind("mined_", 0) == 0) found_mined = true;
    CHECK(found_mined, "loaded MINED pattern matches the source");

    if (found_mined) {
        for (auto& m : matches) {
            if (m.pattern_id.rfind("mined_", 0) != 0) continue;
            auto out = lib2.apply(*fn, m, arch);
            CHECK(out != nullptr, "loaded mined pattern applies");
            if (out)
                CHECK(engine.analyse(*out).score > engine.analyse(*fn).score,
                      "applying the loaded mined pattern is cheaper");
            break;
        }
    }
    std::remove(path.c_str());
}

int main() {
    std::cout << "=== Clunk Peephole Miner Tests ===\n";
    std::cout << "  miner finds & proves a win..." << std::endl;
    test_miner_finds_and_proves_win();
    std::cout << "  miner reduces identity to pass-through..." << std::endl;
    test_miner_reduces_identity_to_passthrough();
    std::cout << "  miner reduces udiv to shift..." << std::endl;
    test_miner_reduces_udiv_to_shift();
    std::cout << "  miner synthesises compare-select (min/max)..." << std::endl;
    test_miner_synthesises_compare_select();
    std::cout << "  miner simplifies redundant select..." << std::endl;
    test_miner_simplifies_redundant_select();
    std::cout << "  miner rewrites a multi-block slice..." << std::endl;
    test_miner_rewrites_multiblock_slice();
    std::cout << "  miner: no false multi-block rewrite..." << std::endl;
    test_miner_multiblock_no_false_rewrite();
    std::cout << "  miner leaves optimal code alone..." << std::endl;
    test_miner_leaves_optimal_alone();
    std::cout << "  miner skips ineligible functions..." << std::endl;
    test_miner_skips_ineligible();
    std::cout << "  mined pattern persists and applies..." << std::endl;
    test_mined_pattern_persists_and_applies();

    std::cout << "\n=== Results: " << g_pass << " passed, " << g_fail << " failed ===\n";
    return g_fail > 0 ? 1 : 0;
}
