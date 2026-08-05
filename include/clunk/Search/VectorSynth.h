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
 * Clunk Vector-Intrinsic Synthesis — width-aware SIMD superoptimisation.
 *
 * Two-pass design:
 *
 *  Pass A — Width cascade (NEW):
 *    For each in-scope function (one that consumes vector values OR
 *    contains a load chain we can pack into vectors), try the widest
 *    available SIMD width first:
 *
 *      AVX-512  (512-bit:  <16 x i32>, <8 x i64>, ...)
 *        ↓ if not worth it: no saturation, or CPU lacks AVX-512
 *      AVX2     (256-bit:  <8 x i32>,  <4 x i64>, ...)
 *        ↓ if not worth it: not enough lanes
 *      AVX      (128-bit:  <4 x i32>,  <2 x i64>, ...)
 *        ↓ if no SIMD win exists at any width
 *      Scalar fallback (no rewrite — original is preserved)
 *
 *    "Worth it" is decided by the cost model: a width is worth keeping
 *    iff (a) the rewritten function scores strictly cheaper than the
 *    original AND (b) the lanes are saturated (≥75% of lanes carry
 *    useful work). When the input vector is wider than the chosen
 *    width, lane decomposition splits it into multiple narrower
 *    vectors via shufflevector.
 *
 *    When the function has SCALAR loads that can be packed (e.g.
 *    N consecutive `load i32` from contiguous pointers), the pass
 *    rewrites the surrounding code: scalar loads become a single
 *    `load <N x i32>`, the computation becomes a vector binop, and
 *    the scalar extracts are reintroduced only where the consumer
 *    actually needs a single lane.
 *
 *  Pass B — Lane idiom synthesis (UNCHANGED from v0.1):
 *    The original three rewrites (lane-wise fusion, reduction
 *    synthesis, shuffle algebra) run as a fixpoint on the chosen
 *    width. These are the primitives that turn a fully scalarised
 *    dot product into `mul <4 x i32>` + `reduce.add`.
 *
 * Soundness: synthesize() only returns a rewrite that (a) the evaluation
 * engine scores strictly cheaper than the input and (b) the SMT verifier
 * proves equivalent to the input (`proven` reports this). Without a
 * working prover, rewrites are returned only if `trust_unverified` is
 * set, and `proven` stays false.
 */
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "clunk/IR/Function.h"
#include "clunk/Evaluator/EvaluationEngine.h"
#include "clunk/Search/SMTVerifier.h"

namespace clunk::search {

// ── Width tier ────────────────────────────────────────────────────────────
// The SIMD widths the synthesiser will consider, in cascade order
// (widest first). The synthesiser tries each tier in turn and keeps
// the first one that yields a profitable, SMT-proven rewrite.
enum class VectorWidth : uint8_t {
    AVX512 = 0,   // 512-bit vectors: <16 x i32>, <8 x i64>
    AVX2   = 1,   // 256-bit:         <8 x i32>,  <4 x i64>
    AVX    = 2,   // 128-bit:         <4 x i32>,  <2 x i64>
    Scalar = 3,   // no SIMD — the fallback
};

// Convert a width tier to a lane count for a given element bit-width.
//   AVX-512: 512 / elem_bits
//   AVX2:    256 / elem_bits
//   AVX:     128 / elem_bits
//   Scalar:  1
inline size_t width_lanes(VectorWidth w, unsigned elem_bits) {
    if (w == VectorWidth::Scalar) return 1;
    unsigned total = (w == VectorWidth::AVX512) ? 512 :
                     (w == VectorWidth::AVX2)   ? 256 : 128;
    return std::max<size_t>(1, total / std::max(1u, elem_bits));
}

struct VectorSynthConfig {
    // Widest tier the pass will attempt. Default AVX-512 — the cascade
    // naturally falls back to narrower tiers when the cost model says
    // no profitable rewrite exists at the wider tier.
    VectorWidth widest_tier = VectorWidth::AVX512;

    // ── Legacy lane-idiom config (unchanged) ──────────────────────────
    size_t max_lanes = 16;             // widest vector the lane-idiom pass touches
    bool enable_lane_fusion = true;
    bool enable_reduction_synthesis = true;
    bool enable_shuffle_fusion = true;

    // ── New: surrounding-code rewrite ─────────────────────────────────
    // When true, the pass promotes scalar load chains into vector
    // loads. Default ON.
    bool enable_load_promotion = true;

    // ── New: lane decomposition ───────────────────────────────────────
    // When true and the chosen width is narrower than the input vector
    // width, split the input via shufflevector into multiple narrower
    // vectors. Default ON.
    bool enable_lane_decomposition = true;

    // Lane saturation threshold (0..1). A width is considered "worth
    // keeping" only if at least this fraction of lanes carry useful
    // work. Default 0.5 — i.e. <8 x i32> with 4 useful lanes is OK
    // for AVX2 but <16 x i32> with 4 useful lanes is not worth AVX-512.
    double saturation_threshold = 0.5;

    // Return unproven rewrites (score-gated only) when SMT is unavailable
    // or returns Unknown. Off by default: no proof, no rewrite.
    bool trust_unverified = false;
    unsigned smt_timeout_ms = 5000;
};

class VectorSynthesizer {
public:
    explicit VectorSynthesizer(evaluator::EvaluationEngine* engine,
                               const VectorSynthConfig& config = {});

    // Try to synthesise a cheaper vector-intrinsic form of `fn`. Returns
    // the rewritten function, or nullptr if no rewrite was found, the
    // rewrite did not score cheaper, or it could not be proven (and
    // trust_unverified is off). `proven` (optional out) is set true iff
    // the returned rewrite carries an SMT equivalence proof.
    //
    // The new width-cascade pass runs FIRST. If it produces a verified
    // rewrite, that's returned immediately. Otherwise the legacy
    // lane-idiom pass runs as a fallback (so existing test cases that
    // relied on the v0.1 lane-fusion behaviour continue to work).
    std::shared_ptr<ir::Function> synthesize(const ir::Function& fn,
                                             bool* proven = nullptr);

    struct Stats {
        size_t functions_seen = 0;
        // ── Width-cascade stats (new) ─────────────────────────────────
        size_t width_cascade_attempts = 0;
        size_t width_cascade_wins = 0;       // verified cheaper rewrite at some width
        size_t load_promotions = 0;          // scalar loads → vector loads
        size_t lane_decompositions = 0;      // wide vector split into narrower
        size_t rejected_by_saturation = 0;   // width tier abandoned: not enough useful lanes
        size_t rejected_by_cost = 0;         // width tier abandoned: cost model says no win
        size_t rejected_by_smt = 0;          // width tier abandoned: SMT refuted
        // ── Lane-idiom stats (legacy) ────────────────────────────────
        size_t lane_fusions = 0;       // vector binops synthesised
        size_t reductions = 0;         // reduce intrinsics synthesised
        size_t shuffle_folds = 0;      // shuffle-algebra folds applied
        size_t rejected_by_score = 0;  // rewrites the cost model rejected
        size_t proven = 0;             // rewrites returned with a proof
        // ── The width tier that produced the returned rewrite ─────────
        // (AVX512 / AVX2 / AVX / Scalar / "legacy lane-idiom")
        std::string winning_tier;
    };
    const Stats& stats() const { return stats_; }

    VectorSynthConfig& config() { return config_; }
    const VectorSynthConfig& config() const { return config_; }

private:
    // ── Width cascade (new) ───────────────────────────────────────────
    std::shared_ptr<ir::Function> synthesize_with_width_cascade(
        const ir::Function& fn, bool* proven);
    // ── Lane-idiom pass (legacy, v0.1) ────────────────────────────────
    std::shared_ptr<ir::Function> synthesize_with_lane_idioms(
        const ir::Function& fn, bool* proven);
    bool apply_lane_fusion(ir::Function& fn);
    bool apply_reduction_synthesis(ir::Function& fn);
    bool apply_shuffle_folds(ir::Function& fn);

    evaluator::EvaluationEngine* engine_;
    VectorSynthConfig config_;
    ir::TypeContext type_ctx_;
    Stats stats_{};
};

} // namespace clunk::search
