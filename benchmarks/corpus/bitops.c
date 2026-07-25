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
 * Clunk beat-O3 / diff-test corpus — bit-twiddling idioms.
 *
 * Pure bit-manipulation kernels — the historical heartland of
 * superoptimisation (Massalin 1987, Granlund-Kenner PLDI'92). All
 * loop-free / scalar / integer (SMT-modelable). -O3 catches some of
 * these (the trivial ones), but clunk's SMT verifier can prove
 * equivalences that -O3's pattern-matcher misses.
 */

/* Population count by pairwise summation. -O3 turns this into a call to
 * `__builtin_popcount` (lowered as `popcnt` on x86 with -mpopcnt); the
 * raw shift-add form is what clunk sees if compiled without -mpopcnt. */
int popcount_byte(unsigned char x) {
    x = (x & 0x55) + ((x >> 1) & 0x55);
    x = (x & 0x33) + ((x >> 2) & 0x33);
    x = (x & 0x0F) + ((x >> 4) & 0x0F);
    return x;
}

/* Count trailing zeros via De Bruijn sequence. -O3 does NOT fold this
 * (the table lookup defeats pattern matching) — clunk can verify the
 * De Bruijn equivalence to a `ctz` intrinsic on architectures that
 * expose one. */
int ctz_debruijn(unsigned x) {
    static const unsigned char table[32] = {
         0,  1, 28,  2, 29, 14, 24,  3, 30, 22, 20, 15, 25, 17,  4,  8,
        31, 27, 13, 23, 21, 19, 16,  7, 26, 12, 18,  6, 11,  5, 10,  9
    };
    if (x == 0) return 32;
    /* Isolate lowest set bit, then multiply by the De Bruijn constant
     * 0x077CB531 and use the top 5 bits as a table index. */
    unsigned v = x & (0u - x);
    return table[(v * 0x077CB531u) >> 27];
}

/* Bit-reverse a byte using the standard 4-stage swap. -O3 emits the
 * same swaps; the equivalence to a single `rbit` (ARM) is provable. */
unsigned char bitreverse_byte(unsigned char x) {
    x = ((x & 0xF0) >> 4) | ((x & 0x0F) << 4);
    x = ((x & 0xCC) >> 2) | ((x & 0x33) << 2);
    x = ((x & 0xAA) >> 1) | ((x & 0x55) << 1);
    return x;
}

/* Next power of two (round up to the smallest power of 2 >= x). The
 * straight-line cascade of shifts fills all bits below the topmost set
 * bit, then `+1` produces the next power. -O3 keeps it; clunk may find
 * a shorter form for inputs known to be small. */
unsigned next_pow2(unsigned x) {
    if (x == 0) return 1;
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return x + 1;
}

/* Bit-swap (XOR-swap): no temporary. Equivalent to `tmp=a; a=b; b=tmp;`
 * when &a != &b, and a no-op when &a == &b. -O3 catches the no-op form;
 * clunk can verify the value equivalence. */
void xor_swap(int* a, int* b) {
    *a ^= *b;
    *b ^= *a;
    *a ^= *b;
}

/* Conditional-set via arithmetic: produces 1 if (x & mask) != 0, else 0.
 * -O3 typically lowers to a CMP + SETNE; the arithmetic form lets clunk
 * prove the equivalence without a branch. */
int has_bit(unsigned x, unsigned mask) {
    return (x & mask) != 0;
}
