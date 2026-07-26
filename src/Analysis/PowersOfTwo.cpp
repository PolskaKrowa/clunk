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
 * Clunk PowersOfTwo — implementation.
 * See include/clunk/Analysis/PowersOfTwo.h for the design/scope notes.
 */
#include "clunk/Analysis/PowersOfTwo.h"

namespace clunk::analysis {

Pow2Fact classify_pow2(const KnownBits& kb) {
    // A conflicting KnownBits fact means the transfer function derived
    // something only possible on an unreachable path — same convention as
    // KnownBits itself: don't trust it, report Unknown.
    if (kb.has_conflict()) return Pow2Fact::Unknown;

    const uint64_t mask = KnownBits::mask_for_width(kb.width);
    const unsigned known_ones = __builtin_popcountll(kb.one & mask);

    // Two or more bits are already provably set: no concretisation can
    // have popcount 1, regardless of the remaining unknown bits.
    if (known_ones >= 2) return Pow2Fact::NotPowerOfTwo;

    if (kb.is_fully_known()) {
        // Nothing left unknown, so known_ones IS the exact bit count.
        return known_ones == 1 ? Pow2Fact::PowerOfTwo : Pow2Fact::NotPowerOfTwo;
    }

    // Not fully known. The value can only be 0-or-a-power-of-two if at
    // most one bit is NOT already known-zero (every other bit is pinned
    // to 0, so at most the one free bit could ever be set).
    const unsigned maybe_set = __builtin_popcountll(mask & ~kb.zero);
    if (maybe_set == 0) return Pow2Fact::NotPowerOfTwo;   // provably == 0
    if (maybe_set == 1) {
        // known_ones can't be 1 here — if it were, that single known-one
        // bit plus every other bit known-zero would make kb fully known,
        // which the branch above already handled. So this is genuinely
        // "0 or a power of two", not yet narrowed to nonzero.
        return Pow2Fact::ZeroOrPowerOfTwo;
    }
    return Pow2Fact::Unknown;
}

Pow2Fact operand_pow2_fact(const std::shared_ptr<ir::Value>& v,
                           const std::unordered_map<std::string, KnownBits>& known_bits_env) {
    return classify_pow2(operand_known_bits(v, known_bits_env));
}

std::unordered_map<std::string, Pow2Fact> classify_powers_of_two(
    const ir::Function& fn,
    const std::unordered_map<std::string, KnownBits>& known_bits_env) {
    std::unordered_map<std::string, Pow2Fact> out;
    for (auto& block : fn.blocks()) {
        if (!block) continue;
        for (auto& inst : block->instructions()) {
            if (!inst || !inst->has_name()) continue;
            if (!inst->type() || !inst->type()->is_integer()) continue;
            auto it = known_bits_env.find(inst->name());
            if (it == known_bits_env.end()) continue;
            out[inst->name()] = classify_pow2(it->second);
        }
    }
    return out;
}

std::unordered_map<std::string, Pow2Fact> analyse_powers_of_two(const ir::Function& fn) {
    return classify_powers_of_two(fn, analyse_known_bits(fn));
}

} // namespace clunk::analysis