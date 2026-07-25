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
 * Clunk Vector-Intrinsic Synthesis — Minotaur-style SIMD superoptimisation.
 *
 * Recognises scalar idioms over vector values and re-expresses them as
 * vector instructions and target-independent vector intrinsics, then
 * gates every rewrite behind the cost model AND an SMT equivalence proof
 * (via SMTVerifier, whose verify() lane-blasts vector functions through
 * ir::Scalarizer — see IR/Scalarizer.h).
 *
 * Synthesised forms:
 *
 *  1. Lane-wise fusion. N independent scalar binops
 *         %r_k = op (extractelement %a, k), (extractelement %b, k)
 *     covering every lane k = 0..N-1 of two same-typed vectors become
 *         %v   = op <N x iM> %a, %b
 *         %r_k = extractelement <N x iM> %v, k        (names preserved)
 *     The per-lane extracts usually die in DCE once rewrite 2 fires.
 *
 *  2. Reduction synthesis. A single-use tree of one associative-
 *     commutative opcode (add/mul/and/or/xor) whose leaves are exactly
 *     {extractelement %v, k : k = 0..N-1} becomes
 *         %root = call iM @clunk.vector.reduce.<op>.v<N>i<M>(<N x iM> %v)
 *     — the horizontal-reduction intrinsic (priced by the cost model as
 *     its log2(N) shuffle+op lowering, and lane-blasted back into a binop
 *     tree by the scalarizer for verification/interpretation).
 *
 *  3. Shuffle algebra. extractelement-of-shufflevector folds to an
 *     extract of the shuffle source; shufflevector-of-shufflevector
 *     composes into a single shuffle (constant, non-undef masks only).
 *
 *  Together, 1+2 turn e.g. a fully scalarised dot product (8 extracts,
 *  4 muls, 3 adds) into `mul <4 x i32>` + `reduce.add` — the class of
 *  SIMD win a scalar peephole miner cannot reach.
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

#include "clunk/IR/Function.h"
#include "clunk/Evaluator/EvaluationEngine.h"
#include "clunk/Search/SMTVerifier.h"

namespace clunk::search {

struct VectorSynthConfig {
    size_t max_lanes = 16;             // widest vector the pass will touch
    bool enable_lane_fusion = true;
    bool enable_reduction_synthesis = true;
    bool enable_shuffle_fusion = true;
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
    std::shared_ptr<ir::Function> synthesize(const ir::Function& fn,
                                             bool* proven = nullptr);

    struct Stats {
        size_t functions_seen = 0;
        size_t lane_fusions = 0;       // vector binops synthesised
        size_t reductions = 0;         // reduce intrinsics synthesised
        size_t shuffle_folds = 0;      // shuffle-algebra folds applied
        size_t rejected_by_score = 0;  // rewrites the cost model rejected
        size_t rejected_by_smt = 0;    // rewrites SMT refuted / couldn't prove
        size_t proven = 0;             // rewrites returned with a proof
    };
    const Stats& stats() const { return stats_; }

    VectorSynthConfig& config() { return config_; }
    const VectorSynthConfig& config() const { return config_; }

private:
    bool apply_lane_fusion(ir::Function& fn);
    bool apply_reduction_synthesis(ir::Function& fn);
    bool apply_shuffle_folds(ir::Function& fn);

    evaluator::EvaluationEngine* engine_;
    VectorSynthConfig config_;
    ir::TypeContext type_ctx_;
    Stats stats_{};
};

} // namespace clunk::search
