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
 * Clunk StrengthReduce Tests — the powers-of-two-driven
 * `x urem y -> x and (y - 1)` rewrite (see
 * include/clunk/IR/StrengthReduce.h).
 */
#include <iostream>
#include <memory>
#include <string>

#include "clunk/Evaluator/Interpreter.h"
#include "clunk/IR/Module.h"
#include "clunk/IR/StrengthReduce.h"
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

// ─────────────────────────────────────────────────────────────────────────
//  Constant power-of-two divisor
// ─────────────────────────────────────────────────────────────────────────

static void test_constant_pow2_divisor_rewritten() {
    auto mod = parse(R"(
define i32 @f(i32 %x) {
entry:
  %r = urem i32 %x, 8
  ret i32 %r
}
)");
    auto fn = mod->function("f");
    CHECK(fn != nullptr, "parses");
    if (!fn) return;

    StrengthReduceStats stats;
    auto out = simplify_pow2_strength_reduce(*fn, &stats);
    CHECK(out != nullptr, "rewrite fires for a constant power-of-two divisor");
    if (!out) return;
    CHECK(stats.urem_to_and == 1, "exactly one rewrite recorded");
    CHECK(count_opcode(*out, Opcode::URem) == 0, "no URem survives");
    CHECK(count_opcode(*out, Opcode::And) == 1, "one And introduced");
    CHECK(count_opcode(*out, Opcode::Sub) == 1, "one Sub (y-1) introduced");

    for (int64_t x : {0, 1, 7, 8, 9, 15, 16, 100, 255, -1, -8}) {
        auto r0 = evaluator::Interpreter::interpret(*fn, {x});
        auto r1 = evaluator::Interpreter::interpret(*out, {x});
        CHECK(r0 && r1 && *r0 == *r1,
              "rewrite preserves semantics for x=" + std::to_string(x));
    }
}

static void test_non_pow2_constant_divisor_untouched() {
    auto mod = parse(R"(
define i32 @f(i32 %x) {
entry:
  %r = urem i32 %x, 6
  ret i32 %r
}
)");
    auto fn = mod->function("f");
    CHECK(fn != nullptr, "parses");
    if (!fn) return;
    auto out = simplify_pow2_strength_reduce(*fn);
    CHECK(out == nullptr, "no rewrite for a non-power-of-two divisor (6)");
}

// ─────────────────────────────────────────────────────────────────────────
//  Runtime (not compile-time-constant) power-of-two-or-zero divisor
// ─────────────────────────────────────────────────────────────────────────

static void test_runtime_masked_divisor_rewritten_and_sound() {
    // %y = %n & 8 is 0-or-8 depending on %n — a genuinely runtime
    // divisor, not a compile-time constant — see PowersOfTwo.h /
    // StrengthReduce.h for why the y==0 case is safe too.
    auto mod = parse(R"(
define i32 @f(i32 %x, i32 %n) {
entry:
  %y = and i32 %n, 8
  %r = urem i32 %x, %y
  ret i32 %r
}
)");
    auto fn = mod->function("f");
    CHECK(fn != nullptr, "parses");
    if (!fn) return;

    StrengthReduceStats stats;
    auto out = simplify_pow2_strength_reduce(*fn, &stats);
    CHECK(out != nullptr, "rewrite fires for a runtime 0-or-pow2 divisor");
    if (!out) return;
    CHECK(stats.urem_to_and == 1, "exactly one rewrite recorded");
    CHECK(count_opcode(*out, Opcode::URem) == 0, "no URem survives");

    for (int64_t n : {0, 1, 2, 3, 4, 7, 8, 9, 15, 16, 100}) {
        for (int64_t x : {0, 1, 5, 7, 8, 9, 100, -1}) {
            auto r1 = evaluator::Interpreter::interpret(*out, {x, n});
            CHECK(r1.has_value(), "rewritten form is always defined (sub+and never traps)");
            if (!r1) continue;

            bool y_is_zero = (n & 8) == 0;
            if (y_is_zero) {
                // The zero case: the SMT verifier's bvurem-by-zero
                // semantics (Z3's SMT-LIB bit-vector theory) equal the
                // dividend, and so does x and (0-1) — confirm directly
                // rather than relying on the original trapping in the
                // interpreter (see StrengthReduce.h's soundness note).
                CHECK(*r1 == x,
                      "y==0 case: rewritten result equals the dividend x");
            } else {
                // y == 8: an ordinary equivalence check against the
                // original urem is meaningful here.
                auto r0 = evaluator::Interpreter::interpret(*fn, {x, n});
                CHECK(r0 && *r0 == *r1,
                      "y==8 case: rewrite matches the original urem");
            }
        }
    }
}

static void test_unconstrained_divisor_untouched() {
    auto mod = parse(R"(
define i32 @f(i32 %x, i32 %y) {
entry:
  %r = urem i32 %x, %y
  ret i32 %r
}
)");
    auto fn = mod->function("f");
    CHECK(fn != nullptr, "parses");
    if (!fn) return;
    auto out = simplify_pow2_strength_reduce(*fn);
    CHECK(out == nullptr, "no rewrite when the divisor carries no powers-of-two guarantee");
}

static void test_srem_is_left_alone() {
    // Signed remainder needs a different identity for negative dividends
    // — deliberately not handled by this pass (see header comment).
    auto mod = parse(R"(
define i32 @f(i32 %x) {
entry:
  %r = srem i32 %x, 8
  ret i32 %r
}
)");
    auto fn = mod->function("f");
    CHECK(fn != nullptr, "parses");
    if (!fn) return;
    auto out = simplify_pow2_strength_reduce(*fn);
    CHECK(out == nullptr, "SRem is never rewritten by this pass");
}

int main() {
    test_constant_pow2_divisor_rewritten();
    test_non_pow2_constant_divisor_untouched();
    test_runtime_masked_divisor_rewritten_and_sound();
    test_unconstrained_divisor_untouched();
    test_srem_is_left_alone();

    std::cout << "\n=== Results: " << g_pass << " passed, " << g_fail << " failed ===\n";
    return g_fail == 0 ? 0 : 1;
}