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
 * Clunk StrengthReduce — sound-by-construction strength reductions driven
 * by the PowersOfTwo analysis (see clunk/Analysis/PowersOfTwo.h).
 *
 * Currently one rewrite:
 *
 *   x urem y   ->   x and (y - 1)
 *
 * applied whenever y is provably 0-or-a-power-of-two
 * (analysis::Pow2Fact::PowerOfTwo or ::ZeroOrPowerOfTwo). This is the
 * standard unsigned "remainder by a power of two" identity, and —
 * crucially — it doesn't need to know WHICH power of two y is: it fires
 * just as well when y is a runtime value (a validated alignment, a
 * bucket count known to be rounded up to a power of two, the result of
 * `n & mask`, ...) as when y is a compile-time constant.
 *
 * Why the zero case is safe too: this project's SMT verifier encodes
 * URem/UDiv as Z3's bvurem/bvudiv (see SMTVerifier.cpp), whose SMT-LIB
 * semantics define bvurem-by-zero to equal the dividend itself — i.e.
 * under the exact equivalence notion this codebase already treats as
 * ground truth, `x urem 0 == x`. And `x and (0 - 1)` is `x and
 * 0xFFFF...F`, which is also exactly `x`. So the rewrite is an EXACT
 * equivalence in the y=0 case too, not merely a "poison can refine to
 * anything" hand-wave — ZeroOrPowerOfTwo alone is enough to fire safely,
 * ::PowerOfTwo doesn't need to be proven separately.
 *
 * Deliberately URem-only: the equivalent SRem identity needs an extra
 * correction term for negative dividends (`x - y * (x/y)` isn't just a
 * mask once sign is in play), so it's left for a future, more careful
 * rewrite rather than risking a subtly-wrong "quick" version here.
 *
 * Sound by construction — no SMT proof required, same trust tier as
 * LoopOpt/MemOpt/DataflowPrune (see Pipeline's Candidate::sound).
 */
#include <memory>

#include "clunk/IR/Function.h"

namespace clunk::ir {

struct StrengthReduceStats {
    size_t urem_to_and = 0;  // `x urem y` -> `x and (y - 1)`, y provably pow2-or-0

    bool changed() const { return urem_to_and != 0; }
};

// Rewrites every `x urem y` where y is provably 0-or-a-power-of-two into
// `x and (y - 1)`. Returns nullptr if nothing was rewritten.
std::shared_ptr<Function> simplify_pow2_strength_reduce(
    const Function& fn, StrengthReduceStats* stats = nullptr);

} // namespace clunk::ir