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
 * Clunk MCACostModel — llvm-mca-backed candidate ranking.
 *
 * The built-in CostModel tables are hand-written Agner-Fog-style
 * approximations; modern superoptimisers (Minotaur §4.3) rank final
 * candidates with LLVM-MCA instead, which models the actual dispatch
 * width, port contention, and scheduler of the target CPU. This class
 * shells out to the system toolchain:
 *
 *     candidate IR text  ─llc→  native asm  ─llvm-mca→  Total Cycles
 *
 * and exposes the measured cycle count as a ranking signal. It is an
 * EXTERNAL FINAL RANKER, not a per-opcode table: use it to arbitrate
 * between a handful of near-final candidates (each measurement costs two
 * subprocess launches, ~20-50 ms), never inside a search loop. Results
 * are cached by structural hash, so re-measuring an unchanged baseline
 * across rounds is free.
 *
 * clunk intrinsics (`clunk.vector.reduce.*`) are rewritten to their
 * `llvm.vector.reduce.*` equivalents on emission, so synthesised vector
 * candidates measure as the real SIMD sequences they lower to.
 *
 * Graceful degradation: if llc / llvm-mca are missing, the IR does not
 * lower (unsupported constructs), or parsing fails, measure() returns
 * ok=false and callers fall back to the built-in cost model. The ranker
 * REFINES scoring only — soundness never depends on it.
 */
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>

#include "clunk/IR/Function.h"

namespace clunk::evaluator {

class MCACostModel {
public:
    struct Measurement {
        bool ok = false;
        double total_cycles = 0.0;   // per --iterations=100 of the block
        double uops = 0.0;
        double ipc = 0.0;
        std::string error;           // populated when !ok
    };

    // True iff both `llc` and `llvm-mca` are runnable on this machine.
    // Cached after the first probe.
    static bool is_available();

    // Measure `fn` through llc + llvm-mca. Thread-safe; cached by
    // structural content.
    Measurement measure(const ir::Function& fn) const;

    // Ranking helper: > 1.0 means `candidate` measures FASTER (fewer
    // cycles) than `original`; returns 0.0 when either side failed to
    // measure (caller falls back to the built-in model).
    double compare(const ir::Function& original,
                   const ir::Function& candidate) const;

    struct Stats {
        size_t measurements = 0;   // external tool invocations
        size_t cache_hits = 0;
        size_t failures = 0;
    };
    const Stats& stats() const { return stats_; }

private:
    // Render `fn` as self-contained LLVM IR (declares for called
    // functions, clunk intrinsics renamed to llvm ones).
    static std::string render_module(const ir::Function& fn);

    mutable std::mutex mutex_;
    mutable std::unordered_map<std::string, Measurement> cache_;
    mutable Stats stats_{};
};

} // namespace clunk::evaluator
