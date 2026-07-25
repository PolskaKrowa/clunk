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
 * Clunk beat-O3 corpus — integer arithmetic kernels.
 *
 * Small, loop-free, scalar integer functions. Their `clang -O3` IR stays
 * inside clunk's parser + SMT-modelable subset (no loops, no memory, no FP),
 * so the harness (scripts/beat_o3.sh) can measure whether clunk finds a
 * strictly-cheaper, SMT-equivalent rewrite of the ALREADY-optimised code.
 */

int mul5(int x)            { return x * 2 + x * 3; }      /* -O3: mul x,5  */
int addsub(int a, int b)   { return (a + b) - b; }        /* -O3: a        */
int times9(int x)          { return x * 9; }              /* lea/shift-add */
int neg_neg(int x)         { return -(-x); }              /* -O3: x        */
int dist(int a, int b, int c) { return a * c + b * c; }   /* (a+b)*c       */
int shift_mask(unsigned x) { return (x << 4) >> 4; }      /* and x,0x0FFFFFFF-ish */
int diff_sq(int a, int b)  { return a * a - b * b; }      /* (a-b)*(a+b)?  */
int poly(int x)            { return x * x + x; }          /* x*(x+1)       */
