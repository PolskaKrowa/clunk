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
 * Clunk PowersOfTwo Tests — the "is this value 0-or-a-power-of-two?"
 * analysis layered on KnownBits (see
 * include/clunk/Analysis/PowersOfTwo.h).
 */
#include <iostream>
#include <memory>
#include <string>

#include "clunk/Analysis/KnownBits.h"
#include "clunk/Analysis/PowersOfTwo.h"
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

static Pow2Fact pow2_of(const Function& fn, const std::string& name) {
    auto facts = analyse_powers_of_two(fn);
    auto it = facts.find(name);
    if (it == facts.end()) return Pow2Fact::Unknown;
    return it->second;
}

// ─────────────────────────────────────────────────────────────────────────
//  classify_pow2 on raw KnownBits values
// ─────────────────────────────────────────────────────────────────────────

static void test_constant_powers_of_two() {
    for (int64_t v : {1, 2, 4, 8, 16, 1 << 30}) {
        auto kb = KnownBits::exact(32, v);
        CHECK(classify_pow2(kb) == Pow2Fact::PowerOfTwo,
              "constant power of two classified PowerOfTwo: " + std::to_string(v));
    }
}

static void test_constant_non_powers_of_two() {
    for (int64_t v : {0, 3, 5, 6, 7, 100, -1}) {
        auto kb = KnownBits::exact(32, v);
        CHECK(classify_pow2(kb) == Pow2Fact::NotPowerOfTwo,
              "constant non-power-of-two (incl. 0) classified NotPowerOfTwo: " + std::to_string(v));
    }
}

static void test_two_known_one_bits_is_not_pow2_even_if_incomplete() {
    // Bits 0 and 2 known-one, everything else unknown: already >= 2 bits
    // set, so NotPowerOfTwo regardless of what the unknown bits turn out
    // to be.
    KnownBits kb = KnownBits::unknown(8);
    kb.one = 0b0101;
    CHECK(classify_pow2(kb) == Pow2Fact::NotPowerOfTwo,
          "two known-one bits -> NotPowerOfTwo even though not fully known");
}

static void test_single_free_bit_is_zero_or_pow2() {
    // Bit 3 unknown, every other bit known-zero: value is either 0 or
    // exactly 8, but we don't yet know which.
    KnownBits kb = KnownBits::unknown(8);
    kb.zero = 0xFF & ~(uint64_t(1) << 3);
    CHECK(classify_pow2(kb) == Pow2Fact::ZeroOrPowerOfTwo,
          "single free bit, not yet known-one -> ZeroOrPowerOfTwo");
}

static void test_all_zero_is_not_pow2() {
    KnownBits kb = KnownBits::exact(8, 0);
    CHECK(classify_pow2(kb) == Pow2Fact::NotPowerOfTwo,
          "provably-zero value is NotPowerOfTwo (0 is not a power of two)");
}

static void test_conflict_is_unknown() {
    KnownBits kb = KnownBits::unknown(8);
    kb.zero = 0x01;
    kb.one = 0x01;  // bit 0 both known-zero and known-one: unreachable path
    CHECK(kb.has_conflict(), "sanity: this KnownBits is conflicting");
    CHECK(classify_pow2(kb) == Pow2Fact::Unknown,
          "conflicting KnownBits classified Unknown, not trusted");
}

// ─────────────────────────────────────────────────────────────────────────
//  analyse_powers_of_two() over small parsed functions
// ─────────────────────────────────────────────────────────────────────────

static void test_and_with_constant_pow2_mask() {
    auto mod = parse(R"(
define i32 @f(i32 %x) {
entry:
  %m = and i32 %x, 8
  ret i32 %m
}
)");
    auto fn = mod->function("f");
    CHECK(fn != nullptr, "parses");
    if (!fn) return;
    CHECK(pow2_of(*fn, "m") == Pow2Fact::ZeroOrPowerOfTwo,
          "x & 8 is 0-or-a-power-of-two (bit 3 is the only free bit)");
}

static void test_and_with_non_pow2_mask_is_unknown() {
    auto mod = parse(R"(
define i32 @f(i32 %x) {
entry:
  %m = and i32 %x, 12
  ret i32 %m
}
)");
    auto fn = mod->function("f");
    CHECK(fn != nullptr, "parses");
    if (!fn) return;
    // 12 = 0b1100: two free bits, so it's not "at most one free bit" ->
    // Unknown (could end up being 0, 4, 8, or 12 — the last of which is
    // not a power of two, so this can't be proven safe).
    CHECK(pow2_of(*fn, "m") == Pow2Fact::Unknown,
          "x & 12 has two free bits -> Unknown, not provably 0-or-pow2");
}

static void test_opaque_argument_is_unknown() {
    auto mod = parse(R"(
define i32 @f(i32 %x) {
entry:
  %y = add i32 %x, 1
  ret i32 %y
}
)");
    auto fn = mod->function("f");
    CHECK(fn != nullptr, "parses");
    if (!fn) return;
    CHECK(pow2_of(*fn, "y") == Pow2Fact::Unknown,
          "an opaque arithmetic result carries no powers-of-two guarantee");
}

static void test_constant_eight_is_power_of_two() {
    auto mod = parse(R"(
define i32 @f(i32 %x) {
entry:
  %c = add i32 0, 8
  ret i32 %c
}
)");
    auto fn = mod->function("f");
    CHECK(fn != nullptr, "parses");
    if (!fn) return;
    CHECK(pow2_of(*fn, "c") == Pow2Fact::PowerOfTwo,
          "0 + 8 folds to the known constant 8 -> PowerOfTwo");
}

int main() {
    test_constant_powers_of_two();
    test_constant_non_powers_of_two();
    test_two_known_one_bits_is_not_pow2_even_if_incomplete();
    test_single_free_bit_is_zero_or_pow2();
    test_all_zero_is_not_pow2();
    test_conflict_is_unknown();
    test_and_with_constant_pow2_mask();
    test_and_with_non_pow2_mask_is_unknown();
    test_opaque_argument_is_unknown();
    test_constant_eight_is_power_of_two();

    std::cout << "\n=== Results: " << g_pass << " passed, " << g_fail << " failed ===\n";
    return g_fail == 0 ? 0 : 1;
}