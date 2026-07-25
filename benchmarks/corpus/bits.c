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
 * Clunk beat-O3 corpus — bit-twiddling kernels.
 *
 * Classic superoptimizer territory: short bit-manipulation sequences where a
 * cheaper equivalent sometimes exists that a greedy compiler pass leaves on
 * the table. All loop-free / integer / scalar (SMT-modelable).
 */

unsigned avg(unsigned a, unsigned b)  { return (a & b) + ((a ^ b) >> 1); }  /* no-overflow average */
int is_pow2(unsigned x)               { return x && !(x & (x - 1)); }
unsigned clear_low(unsigned x)        { return x & (x - 1); }               /* clear lowest set bit */
unsigned low_bit(unsigned x)          { return x & (0u - x); }              /* isolate lowest set bit */
int same_sign(int a, int b)           { return (a ^ b) >= 0; }
unsigned swap_nib(unsigned char x)    { return (x << 4) | (x >> 4); }
int xor_id(int x)                     { return (x ^ 0x55) ^ 0x55; }         /* -O3: x */
unsigned or_and(unsigned x)           { return (x & 1) | (x & 2); }         /* x & 3 */
int abs_branchless(int x)             { int m = x >> 31; return (x + m) ^ m; }
