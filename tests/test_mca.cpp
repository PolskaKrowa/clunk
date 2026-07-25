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
 * Clunk MCA Tests — llvm-mca-backed candidate ranking.
 *
 * These tests need `llc` and `llvm-mca` on PATH. When the toolchain is
 * missing they SKIP (pass vacuously) — graceful degradation is part of
 * the contract, so the availability probe itself is always exercised.
 */
#include <iostream>
#include <memory>
#include <string>

#include "clunk/Evaluator/MCACostModel.h"
#include "clunk/IR/Module.h"
#include "clunk/Parser/IRParser.h"

static int g_pass = 0, g_fail = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { std::cerr << "FAIL: " << msg << " at " << __FILE__ << ":" << __LINE__ << "\n"; g_fail++; } \
    else { g_pass++; } \
} while(0)

using namespace clunk;
using namespace clunk::evaluator;

static std::shared_ptr<ir::Module> parse(const char* src) {
    parser::IRParser p;
    return p.parse_string(src);
}

static void test_measure_basic() {
    auto mod = parse(R"(
define i32 @m8(i32 %x) {
entry:
  %b = mul i32 %x, 8
  ret i32 %b
}
)");
    MCACostModel mca;
    auto m = mca.measure(*mod->function("m8"));
    CHECK(m.ok, "simple integer function measures (error: " + m.error + ")");
    if (m.ok) {
        CHECK(m.total_cycles > 0.0, "cycle count is positive");
        CHECK(m.ipc > 0.0, "IPC parsed");
    }

    // Second measurement of the same function must hit the cache.
    auto before = mca.stats().cache_hits;
    (void)mca.measure(*mod->function("m8"));
    CHECK(mca.stats().cache_hits == before + 1, "repeat measurement hits the cache");
}

static void test_compare_div_vs_shift() {
    // udiv-by-non-constant vs shift: the machine model must agree with
    // reality that the division chain is slower.
    auto slow = parse(R"(
define i32 @f(i32 %x, i32 %y) {
entry:
  %a = udiv i32 %x, %y
  %b = udiv i32 %a, %y
  %c = udiv i32 %b, %y
  ret i32 %c
}
)");
    auto fast = parse(R"(
define i32 @f(i32 %x, i32 %y) {
entry:
  %a = lshr i32 %x, 3
  ret i32 %a
}
)");
    MCACostModel mca;
    double ratio = mca.compare(*slow->function("f"), *fast->function("f"));
    CHECK(ratio > 1.0, "llvm-mca ranks the shift chain faster than the udiv chain (ratio=" +
                        std::to_string(ratio) + ")");
}

static void test_vector_intrinsic_lowering() {
    // A synthesised vector candidate (clunk.vector.reduce.*) must lower
    // through the llvm.vector.reduce.* rename and measure successfully.
    auto mod = parse(R"(
define i32 @dot4(<4 x i32> %a, <4 x i32> %b) {
entry:
  %v = mul <4 x i32> %a, %b
  %r = call i32 @clunk.vector.reduce.add.v4i32(<4 x i32> %v)
  ret i32 %r
}
)");
    MCACostModel mca;
    auto m = mca.measure(*mod->function("dot4"));
    CHECK(m.ok, "vector-intrinsic candidate lowers and measures (error: " + m.error + ")");
}

static void test_unlowerable_fails_gracefully() {
    // A function llc cannot lower must return ok=false, not crash. A
    // REAL intrinsic called with the wrong signature trips the IR
    // verifier ("Intrinsic has incorrect argument type") — calls to
    // unknown plain externals, by contrast, lower fine.
    auto mod = parse(R"(
define i32 @weird(i32 %x) {
entry:
  %r = call i32 @clunk.vector.reduce.add.v4i32(i32 %x)
  ret i32 %r
}
)");
    MCACostModel mca;
    auto m = mca.measure(*mod->function("weird"));
    CHECK(!m.ok, "mis-typed intrinsic fails gracefully");
    CHECK(!m.error.empty(), "failure carries a reason");
}

int main() {
    std::cerr << "test_mca: llvm-mca-backed ranking\n";
    if (!MCACostModel::is_available()) {
        std::cerr << "SKIP: llc / llvm-mca not on PATH — graceful-degradation "
                     "path exercised, nothing else to test\n";
        MCACostModel mca;
        parser::IRParser p;
        auto mod = p.parse_string(
            "define i32 @f(i32 %x) {\nentry:\n  ret i32 %x\n}\n");
        auto m = mca.measure(*mod->function("f"));
        CHECK(!m.ok, "measure degrades gracefully without the toolchain");
        std::cerr << "passed " << g_pass << ", failed " << g_fail << "\n";
        return g_fail == 0 ? 0 : 1;
    }
    test_measure_basic();
    test_compare_div_vs_shift();
    test_vector_intrinsic_lowering();
    test_unlowerable_fails_gracefully();
    std::cerr << "passed " << g_pass << ", failed " << g_fail << "\n";
    return g_fail == 0 ? 0 : 1;
}
