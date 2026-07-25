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
 * Clunk beat-O3 corpus — comparison / select kernels.
 *
 * icmp + select chains, which the SMT verifier models directly. Loop-free /
 * integer / scalar. Note: some of these -O3-lower to llvm.smax/smin/abs
 * intrinsics (calls), which clunk treats as opaque — the harness reports
 * those as unverifiable rather than pretending to optimise them.
 */

int sign(int x)                 { return (x > 0) - (x < 0); }
int clamp0(int x)               { return x < 0 ? 0 : x; }
int sel_chain(int a, int b, int c) { return a ? b : c ? b : a; }
int min3(int a, int b, int c)   { int m = a < b ? a : b; return m < c ? m : c; }
int in_range(int x, int lo, int hi) { return x >= lo && x <= hi; }
int cmp_eq_zero(int a, int b)   { return (a - b) == 0; }         /* a == b */
int not_not(int x)              { return !!x; }
