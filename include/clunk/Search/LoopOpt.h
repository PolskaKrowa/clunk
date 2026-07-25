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
 * Clunk LoopOpt — loop-aware, sound-by-construction rewrites.
 *
 * The SMT verifier (soundly) refuses functions with back-edges, so loop
 * rewrites cannot be SMT-gated — every transform here must be exact by
 * construction, like the stochastic search's sound mutation family:
 *
 *  1. LICM (`hoist_invariants`). Pure, non-trapping instructions whose
 *     operands are loop-invariant move to the loop's preheader. Poison
 *     flags (nuw/nsw/exact) are STRIPPED from hoisted instructions —
 *     hoisting executes them speculatively (possibly on iterations that
 *     never happen), and a speculated flag violation must not introduce
 *     poison the original never produced. Divisions never move (trap on
 *     zero); loads never move (trap on bad addresses, and would need
 *     alias analysis besides).
 *
 *  2. Constant-trip full unrolling (`unroll_constant_loops`). Single-block
 *     loops whose exit condition folds to a constant on every iteration
 *     (found by abstract interpretation of the loop-carried values, NOT
 *     by induction-variable pattern matching — so any constant-bounded
 *     shape unrolls, not just canonical `i = i + 1` counters) are
 *     replaced by their straight-line expansion. Body instructions are
 *     cloned per iteration in order, so memory ops and calls keep their
 *     exact count and sequence. The payoff is compound: the unrolled
 *     function has no back-edge, so the SMT-verified peephole miner and
 *     the rest of the pipeline can attack it in subsequent rounds —
 *     unrolling converts an unverifiable function into a verifiable one.
 *
 * The cost model arbitrates the unrolling TRADEOFF: an unrolled candidate
 * is only adopted if it scores cheaper (branch/loop overhead removed vs.
 * code growth), exactly like any other candidate.
 */
#include <cstddef>
#include <memory>

#include "clunk/IR/Function.h"

namespace clunk::search {

struct LoopOptConfig {
    size_t max_trip = 8;                    // widest loop to fully unroll
    size_t max_unrolled_instructions = 256; // trip * body-size cap
    size_t max_simulation_steps = 4096;     // abstract-interpretation fuel
};

class LoopOptimizer {
public:
    explicit LoopOptimizer(const LoopOptConfig& config = {});

    // LICM. Returns the rewritten function, or nullptr if nothing moved.
    std::shared_ptr<ir::Function> hoist_invariants(const ir::Function& fn);

    // Full unrolling of constant-trip single-block loops. Returns the
    // rewritten function, or nullptr if no loop qualified.
    std::shared_ptr<ir::Function> unroll_constant_loops(const ir::Function& fn);

    struct Stats {
        size_t instructions_hoisted = 0;
        size_t loops_unrolled = 0;
        size_t iterations_expanded = 0;
    };
    const Stats& stats() const { return stats_; }

    LoopOptConfig& config() { return config_; }

private:
    LoopOptConfig config_;
    Stats stats_{};
};

} // namespace clunk::search
