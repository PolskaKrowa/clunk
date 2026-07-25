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
 * Clunk KnownBits — a lightweight known-bits abstract-interpretation
 * analysis, in the spirit of Souper's "infer known bits" / LLVM's
 * computeKnownBits: for each integer SSA value, track which bits are
 * provably 0, provably 1, or unknown.
 *
 * Scope (deliberately "quick", not a full compiler-grade analysis):
 *   - Single forward pass over the function in block order. Each named
 *     instruction's known bits are computed once, from the already-computed
 *     bits of its operands (constants are exact; earlier-defined named
 *     values are looked up; anything not yet known is treated as fully
 *     unknown).
 *   - Loop-carried Phi operands (back-edges — the incoming value is
 *     defined LATER in block order than the Phi itself) are therefore
 *     conservatively unknown rather than iterated to a fixed point. This
 *     is sound (unknown is always a safe answer) but not maximally
 *     precise across loops. Phis merging only forward-defined values
 *     (the common straight-line if/else diamond shape) ARE precise.
 *   - Transfer functions favour cheap-but-sound rules over maximal
 *     precision — e.g. Add/Sub/Mul propagate only trailing-known-zero
 *     bits rather than doing full carry-lattice reasoning. This still
 *     catches the common, high-value cases (masks, shifts, alignment,
 *     zero/sign extension, redundant comparisons) cheaply.
 *
 * Only integer-typed (including i1) values get known-bits facts;
 * pointers, floats, aggregates, and vectors are always unknown.
 */
#include <cstdint>
#include <optional>
#include <unordered_map>

#include "clunk/IR/Function.h"
#include "clunk/IR/Instruction.h"
#include "clunk/IR/Value.h"

namespace clunk::analysis {

// ── The KnownBits lattice ──────────────────────────────────────────────
// Invariant: (zero & one) == 0 outside of a temporarily-conflicting
// intermediate (has_conflict() detects that — it means the transfer
// function derived a provably-unreachable fact, e.g. from a mismatched
// icmp; conservatively treat as fully unknown rather than trusting it).
struct KnownBits {
    unsigned width = 32;
    uint64_t zero = 0;  // bit i set  =>  bit i of the value is provably 0
    uint64_t one = 0;   // bit i set  =>  bit i of the value is provably 1

    static uint64_t mask_for_width(unsigned w) {
        if (w == 0) return 0;
        return (w >= 64) ? ~uint64_t(0) : ((uint64_t(1) << w) - 1);
    }

    static KnownBits unknown(unsigned w) {
        KnownBits kb;
        kb.width = w;
        return kb;
    }

    // Exact known bits for a concrete value (e.g. a ConstantInt payload,
    // which clunk stores sign-extended in an int64_t — see
    // StochasticSearch.cpp's wrap_to_width/as_unsigned convention, mirrored
    // here via the unsigned-truncate-to-width cast below).
    static KnownBits exact(unsigned w, int64_t value) {
        KnownBits kb;
        kb.width = w;
        const uint64_t mask = mask_for_width(w);
        const uint64_t bits = static_cast<uint64_t>(value) & mask;
        kb.one = bits;
        kb.zero = (~bits) & mask;
        return kb;
    }

    bool has_conflict() const { return (zero & one) != 0; }
    bool is_fully_known() const { return (zero | one) == mask_for_width(width); }

    // Valid only when is_fully_known(): the value's bit pattern. For
    // width > 1 this sign-extends to int64_t per clunk's ConstantInt
    // convention (see StochasticSearch.cpp's wrap_to_width). i1 is a
    // special case: this codebase treats it as a plain 0/1 boolean, never
    // as a signed 1-bit integer — Interpreter::apply_icmp returns 0/1
    // (not 0/-1), and every consumer of an i1 (Br/Select conditions)
    // checks `!= 0`. Sign-extending bit-pattern 1 to -1 here would still
    // satisfy those `!= 0` checks, but would silently disagree with the
    // literal ConstantInt value everything else in the pipeline expects
    // for "true", so it's special-cased out.
    int64_t signed_value() const {
        uint64_t bits = one & mask_for_width(width);
        if (width == 1) return static_cast<int64_t>(bits);
        if (width > 0 && width < 64 && (bits & (uint64_t(1) << (width - 1)))) {
            bits |= ~mask_for_width(width);
        }
        return static_cast<int64_t>(bits);
    }

    // Meet (lattice join towards "less known"): bits known-equal in both
    // operands stay known; anything else becomes unknown. Used for Select
    // (both arms) and Phi (all incoming values).
    static KnownBits meet(const KnownBits& a, const KnownBits& b) {
        KnownBits kb;
        kb.width = a.width;
        kb.zero = a.zero & b.zero;
        kb.one = a.one & b.one;
        return kb;
    }
};

// Known bits for a single operand: exact for a ConstantInt, looked up by
// name in `env` for a named value, unknown otherwise (including for
// non-integer types).
KnownBits operand_known_bits(const std::shared_ptr<ir::Value>& v,
                              const std::unordered_map<std::string, KnownBits>& env);

// Per-instruction transfer function: computes `inst`'s known bits from its
// operands' known bits (resolved through `env`). Returns unknown() for
// opcodes with no known-bits rule (memory ops, calls, float ops, casts to
// pointer, aggregates, vectors, ...).
KnownBits compute_known_bits(const ir::Instruction& inst,
                              const std::unordered_map<std::string, KnownBits>& env);

// Run compute_known_bits over every named instruction in `fn`, in block /
// program order, building the environment incrementally. See the
// single-forward-pass caveat above.
std::unordered_map<std::string, KnownBits> analyse_known_bits(const ir::Function& fn);

// If `kb` is fully known AND `ty` is an integer type, returns the
// equivalent ConstantInt. Otherwise nullopt.
std::optional<int64_t> known_bits_constant(const KnownBits& kb, const ir::Type* ty);

} // namespace clunk::analysis
