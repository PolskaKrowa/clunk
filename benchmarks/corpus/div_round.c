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
 * Clunk beat-O3 / diff-test corpus — integer division + rounding idioms.
 *
 * Division is the canonical superoptimiser target: it's the slowest scalar
 * integer op on every modern CPU (latency 20-40 cycles vs ~1 for add/and),
 * so a single division-elimination rewrite dominates the speedup envelope.
 * These idioms are loop-free / scalar / integer (SMT-modelable) and come
 * from real code: Linux kernel round-up macros, fixed-point audio, the
 * `(a+b-1)/b` ceil-div idiom, etc.
 */

/* Round-to-nearest with ties-to-even (banker's rounding). -O3 leaves a
 * div+rem pair; clunk may be able to fold to a single div via the
 * "div + rem => div * (b+1) / b" pattern, but only when b is known. */
int round_div(int a, int b)             { return (a + b / 2) / b; }

/* Classic ceil-div: smallest k with k*b >= a. Linux kernel macro
 * DIV_ROUND_UP. -O3 folds to `((a+b-1)/b)`, but cannot eliminate the
 * division (clunk could if `b` is provably a power of two). */
int ceil_div(int a, int b)              { return (a + b - 1) / b; }

/* Floor-div: for non-negative a this is just `a / b`. -O3 already
 * simplifies this — included as a regression-baseline. */
int floor_div(unsigned a, unsigned b)   { return a / b; }

/* "Div-then-mul" recovers the dividend minus the remainder. -O3 lowers
 * this to `a - (a % b)`, a div+rem pair. clunk could fuse to a single
 * div (the x86 `div` instruction yields both quotient and remainder). */
int div_mul(int a, int b)               { return a / b * b; }

/* Average of two unsigned values without overflow — classic superoptimiser
 * target. -O3 emits ` (a & b) + ((a ^ b) >> 1)`. clunk can verify the
 * equivalence of the two forms directly. */
unsigned avg_u(unsigned a, unsigned b)  { return (a + b) / 2; }

/* Modulo with constant divisor — the canonical "magic-number" rewrite.
 * -O3 turns this into a multiply-high + shift, but clunk may find a
 * shorter equivalent when the constant is a small power-of-two. */
int mod_const(int x)                    { return x % 7; }
