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
 * Clunk beat-O3 / diff-test corpus — comparison / icmp chains.
 *
 * Chains of `icmp` + `select` are the dominant hand-rolled 3-way compare
 * pattern in C (qsort comparators, sorting networks, range checks).
 * They are SMT-modelable end-to-end (no memory / FP / loops) and are
 * where clunk's verifier can prove equivalences that -O3's peephole
 * pattern-matcher misses — e.g. collapsing a 5-instruction 3-way
 * compare to a single `sadd_overflow + icmp` form, or the classic
 * `(a > b) - (a < b)` sign function.
 */

/* Three-way sign: -1 / 0 / +1. -O3 lowers this to `icmp slt`, `icmp sgt`,
 * then `zext`/`sub`. clunk can verify the equivalence to a single
 * `(a >> 31) | ((-a) >> 31)` form (with nsw on the negation). */
int sign3(int x)                        { return (x > 0) - (x < 0); }

/* Classic 3-way compare for qsort: returns -1 / 0 / +1. -O3 emits two
 * selects chained; clunk can prove equivalence to a sign-subtract form
 * `((a-b) >> 31) | (((b-a)) >> 31)` (with nsw assumptions). */
int cmp3(int a, int b)                  { return (a < b) ? -1 : (a > b) ? 1 : 0; }

/* Clamp to [lo, hi]: returns x if in range, else lo or hi. -O3 emits
 * two selects. clunk may fuse them. */
int clamp(int x, int lo, int hi)        { return x < lo ? lo : (x > hi ? hi : x); }

/* Median of three without sorting. -O3 emits ~5 selects; clunk may
 * find a shorter arithmetic form. */
int median3(int a, int b, int c) {
    if (a > b) { int t = a; a = b; b = t; }
    if (b > c) { int t = b; b = c; c = t; }
    if (a > b) { int t = a; a = b; b = t; }
    return b;
}

/* Boolean and-of-two-comparisons. -O3 already lowers this to two `icmp`s
 * with a single branch; clunk can verify the no-branch form. */
int in_range_inclusive(int x, int lo, int hi) {
    return (x >= lo) && (x <= hi);
}

/* Short-circuit OR chain: returns 1 if x equals any of 0, 1, 2, 4.
 * -O3 typically uses a switch (jump table); clunk can verify the
 * arithmetic form `(x >= 0) & (x <= 4) & (x != 3)` is equivalent
 * (only for the stated constant set). */
int is_small_pow2_or_zero(int x) {
    return x == 0 || x == 1 || x == 2 || x == 4;
}
