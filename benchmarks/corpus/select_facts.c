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
 * Select-condition redundancy — patterns clang -O3 misses (checked against
 * clang 16): instcombine does not use dominating/select conditions for
 * bit-level facts, and CVP only tracks value ranges. After -O3 each of
 * these keeps a redundant instruction in a select arm that is provably
 * simplifiable UNDER the select's condition:
 *
 *   clear_known_bit: `and x, -2`  ==  x       when (x & 1) == 0
 *   redundant_or:    `and x, -5`  ==  x       when (x & 4) == 0
 *                    (clang first folds (x|4)^4 to x&-5, still redundant)
 *   guarded_mask:    `and x, 5`   ==  5       when (x & 7) == 5
 *
 * These are the select-arm conditional-mining targets (R1.B, select form).
 */

int clear_known_bit(int x) {
    if ((x & 1) == 0) return x & ~1;
    return -x;
}

unsigned redundant_or(unsigned x, unsigned y) {
    if (x & 4) return y;
    return (x | 4) ^ 4;
}

int guarded_mask(int x) {
    if ((x & 7) == 5) return x & 5;
    return 0;
}
