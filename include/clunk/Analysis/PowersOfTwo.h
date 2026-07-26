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

#pragma once
/*
 * Clunk PowersOfTwo — a lightweight "is this value a power of two?"
 * analysis, layered directly on top of clunk::analysis::KnownBits (see
 * KnownBits.h) rather than duplicating a separate abstract interpreter.
 *
 * A value has at most one set bit — i.e. it is either 0 or a power of
 * two — for every concretisation consistent with its known bits iff at
 * most one bit is NOT already known-zero. Combined with KnownBits'
 * existing "is this value nonzero?" fact (any bit known-one rules out
 * zero), that gives a cheap, sound classification:
 *
 *   - NotPowerOfTwo:     two or more bits are provably set, OR the value
 *                        is provably exactly 0 (0 is not a power of two
 *                        by convention).
 *   - PowerOfTwo:        exactly one bit is provably set and no other bit
 *                        could possibly be set (fully pinned down).
 *   - ZeroOrPowerOfTwo:  at most one bit could possibly be set, but it
 *                        isn't yet known whether that bit IS set (so the
 *                        value is 0 or a power of two — still useful,
 *                        e.g. this is exactly the guarantee `x & -x`
 *                        or a validated alignment/mask value gives you).
 *   - Unknown:           none of the above provable.
 *
 * Like KnownBits, this is deliberately a quick, sound-but-incomplete
 * analysis (single forward pass, no fixed-point iteration across loop
 * back-edges) rather than a maximally-precise one — see KnownBits.h for
 * the same caveats, which apply here unchanged since this analysis is
 * purely a per-value classification of KnownBits' output.
 */
#include <unordered_map>

#include "clunk/Analysis/KnownBits.h"
#include "clunk/IR/Function.h"
#include "clunk/IR/Instruction.h"
#include "clunk/IR/Value.h"

namespace clunk::analysis {

enum class Pow2Fact {
    Unknown,
    NotPowerOfTwo,
    PowerOfTwo,
    ZeroOrPowerOfTwo,
};

// Classify a single KnownBits fact. See the file comment above for the
// exact soundness argument. has_conflict() facts (provably-unreachable
// code) are conservatively reported Unknown rather than trusted.
Pow2Fact classify_pow2(const KnownBits& kb);

// Powers-of-two fact for a single operand: exact for a ConstantInt,
// classified from `known_bits_env` for a named value, Unknown otherwise.
Pow2Fact operand_pow2_fact(const std::shared_ptr<ir::Value>& v,
                           const std::unordered_map<std::string, KnownBits>& known_bits_env);

// Classify every named integer instruction in `fn`, reusing an
// already-computed known-bits environment (e.g. one the caller is also
// using for constant folding, so it's only computed once per round).
std::unordered_map<std::string, Pow2Fact> classify_powers_of_two(
    const ir::Function& fn,
    const std::unordered_map<std::string, KnownBits>& known_bits_env);

// Convenience one-shot entry point: runs analyse_known_bits() internally
// and classifies the result. Prefer classify_powers_of_two() when you
// already have a KnownBits environment lying around.
std::unordered_map<std::string, Pow2Fact> analyse_powers_of_two(const ir::Function& fn);

} // namespace clunk::analysis