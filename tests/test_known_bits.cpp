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
 * Clunk KnownBits Tests — the "infer known bits" analysis used by
 * dataflow_prune_phase (see include/clunk/Analysis/KnownBits.h).
 */
#include <iostream>
#include <memory>
#include <string>

#include "clunk/Analysis/KnownBits.h"
#include "clunk/IR/Module.h"
#include "clunk/Parser/IRParser.h"

static int g_pass = 0, g_fail = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { std::cerr << "FAIL: " << msg << " at " << __FILE__ << ":" << __LINE__ << "\n"; g_fail++; } \
    else { g_pass++; } \
} while(0)

using namespace clunk;
using namespace clunk::ir;
using namespace clunk::analysis;

static std::shared_ptr<Module> parse(const char* src) {
    parser::IRParser p;
    return p.parse_string(src);
}

// ─────────────────────────────────────────────────────────────────────────
//  KnownBits lattice basics
// ─────────────────────────────────────────────────────────────────────────

static void test_exact_and_unknown() {
    auto kb = KnownBits::exact(8, 5);  // 0b0000_0101
    CHECK(kb.is_fully_known(), "exact() is fully known");
    CHECK(!kb.has_conflict(), "exact() has no conflict");
    CHECK(kb.signed_value() == 5, "exact(8,5) round-trips");

    auto neg = KnownBits::exact(8, -1);  // all-ones byte
    CHECK(neg.signed_value() == -1, "exact(8,-1) sign-extends correctly");
    CHECK(neg.one == 0xFF, "exact(8,-1): all 8 bits known-one");

    auto unk = KnownBits::unknown(32);
    CHECK(!unk.is_fully_known(), "unknown() is not fully known");
    CHECK(unk.zero == 0 && unk.one == 0, "unknown() has no known bits");
}

static void test_meet() {
    auto a = KnownBits::exact(8, 0b0110);
    auto b = KnownBits::exact(8, 0b0100);
    auto m = KnownBits::meet(a, b);
    // Bit 1 (0b0010) disagrees between a and b -> unknown in the meet.
    // Bit 2 (0b0100) agrees (both 1) -> known-one. Bits 3-7 agree (both 0).
    CHECK((m.one & 0b0100) != 0, "meet: agreeing 1-bit stays known-one");
    CHECK((m.zero & 0b1000) != 0, "meet: agreeing 0-bit stays known-zero");
    CHECK(!(m.zero & 0b0010) && !(m.one & 0b0010), "meet: disagreeing bit is unknown");
}

// ─────────────────────────────────────────────────────────────────────────
//  Per-opcode transfer functions, driven through analyse_known_bits() over
//  small parsed functions (closer to how the pass is actually used).
// ─────────────────────────────────────────────────────────────────────────

static KnownBits kb_of(const Function& fn, const std::string& name) {
    auto env = analyse_known_bits(fn);
    auto it = env.find(name);
    if (it == env.end()) return KnownBits::unknown(32);
    return it->second;
}

static void test_and_with_zero_is_fully_known() {
    auto mod = parse(R"(
define i32 @f(i32 %a) {
entry:
  %m = and i32 %a, 0
  ret i32 %m
}
)");
    auto fn = mod->function("f");
    CHECK(fn != nullptr, "parses");
    if (!fn) return;
    auto kb = kb_of(*fn, "m");
    CHECK(kb.is_fully_known(), "and x, 0 is fully known");
    CHECK(kb.signed_value() == 0, "and x, 0 == 0");
}

static void test_shl_then_mask_is_fully_known() {
    auto mod = parse(R"(
define i32 @f(i32 %a) {
entry:
  %s = shl i32 %a, 4
  %m = and i32 %s, 15
  ret i32 %m
}
)");
    auto fn = mod->function("f");
    CHECK(fn != nullptr, "parses");
    if (!fn) return;
    auto kb = kb_of(*fn, "m");
    CHECK(kb.is_fully_known(), "(a<<4) & 15 is fully known (low 4 bits provably 0)");
    CHECK(kb.signed_value() == 0, "(a<<4) & 15 == 0");
    // The shift itself should NOT be fully known (a is opaque), but its
    // low 4 bits should be known-zero.
    auto skb = kb_of(*fn, "s");
    CHECK(!skb.is_fully_known(), "a<<4 alone is not fully known");
    CHECK((skb.zero & 0xF) == 0xF, "a<<4 has 4 known-zero low bits");
}

static void test_two_constants_add_exact() {
    auto mod = parse(R"(
define i32 @f() {
entry:
  %r = add i32 19, 23
  ret i32 %r
}
)");
    auto fn = mod->function("f");
    CHECK(fn != nullptr, "parses");
    if (!fn) return;
    auto kb = kb_of(*fn, "r");
    CHECK(kb.is_fully_known(), "19+23 is fully known");
    CHECK(kb.signed_value() == 42, "19+23 == 42");
}

static void test_mul_trailing_zeros() {
    auto mod = parse(R"(
define i32 @f(i32 %a, i32 %b) {
entry:
  %sa = shl i32 %a, 2
  %sb = shl i32 %b, 3
  %m = mul i32 %sa, %sb
  ret i32 %m
}
)");
    auto fn = mod->function("f");
    CHECK(fn != nullptr, "parses");
    if (!fn) return;
    // (a<<2) has >=2 trailing zero bits, (b<<3) has >=3 -> product has >=5.
    auto kb = kb_of(*fn, "m");
    CHECK((kb.zero & 0x1F) == 0x1F, "product of aligned values has >=5 known-zero low bits");
}

static void test_icmp_constants_fold() {
    auto mod = parse(R"(
define i1 @f() {
entry:
  %eq = icmp eq i32 5, 5
  %ne = icmp eq i32 5, 3
  %slt = icmp slt i32 -1, 0
  %ult = icmp ult i32 -1, 0
  ret i1 %eq
}
)");
    auto fn = mod->function("f");
    CHECK(fn != nullptr, "parses");
    if (!fn) return;
    CHECK(kb_of(*fn, "eq").signed_value() == 1, "icmp eq 5,5 -> true");
    CHECK(kb_of(*fn, "ne").signed_value() == 0, "icmp eq 5,3 -> false");
    CHECK(kb_of(*fn, "slt").signed_value() == 1, "icmp slt -1,0 -> true (signed)");
    CHECK(kb_of(*fn, "ult").signed_value() == 0, "icmp ult -1,0 -> false (-1 is UINT_MAX)");
}

static void test_zext_sext_trunc() {
    auto mod = parse(R"(
define i32 @f(i8 %a) {
entry:
  %m = and i8 %a, 0
  %z = zext i8 %m to i32
  %s = sext i8 %m to i32
  %t = trunc i32 %z to i8
  ret i32 %z
}
)");
    auto fn = mod->function("f");
    CHECK(fn != nullptr, "parses");
    if (!fn) return;
    CHECK(kb_of(*fn, "z").is_fully_known() && kb_of(*fn, "z").signed_value() == 0,
          "zext of known-0 is known-0");
    CHECK(kb_of(*fn, "s").is_fully_known() && kb_of(*fn, "s").signed_value() == 0,
          "sext of known-0 is known-0");
    CHECK(kb_of(*fn, "t").is_fully_known() && kb_of(*fn, "t").signed_value() == 0,
          "trunc of known-0 is known-0");
}

static void test_opaque_add_not_fully_known() {
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
    CHECK(!kb_of(*fn, "r").is_fully_known(), "add of two opaque args is not fully known");
}

static void test_select_meet() {
    auto mod = parse(R"(
define i32 @f(i1 %c) {
entry:
  %r = select i1 %c, i32 6, i32 4
  ret i32 %r
}
)");
    auto fn = mod->function("f");
    CHECK(fn != nullptr, "parses");
    if (!fn) return;
    auto kb = kb_of(*fn, "r");
    // 6 = 0b110, 4 = 0b100: bit 1 disagrees, bits 0 and 2 agree.
    CHECK(!kb.is_fully_known(), "select with unknown condition is not fully known");
    CHECK((kb.one & 0b100) != 0, "select meet: agreeing bit 2 (both have it set) stays known");
    CHECK((kb.zero & 0b001) != 0, "select meet: agreeing bit 0 (both clear) stays known");
}

int main() {
    test_exact_and_unknown();
    test_meet();
    test_and_with_zero_is_fully_known();
    test_shl_then_mask_is_fully_known();
    test_two_constants_add_exact();
    test_mul_trailing_zeros();
    test_icmp_constants_fold();
    test_zext_sext_trunc();
    test_opaque_add_not_fully_known();
    test_select_meet();

    std::cout << "\n=== Results: " << g_pass << " passed, " << g_fail << " failed ===\n";
    return g_fail == 0 ? 0 : 1;
}
