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
// along with this program.  If not, see <https://www.gnu.org/licenses>.

#pragma once
/*
 * Clunk Algorithmic Preprocessor.
 *
 * A module-level pre-pass (same shape as CrossFunctionPasses) that runs
 * BEFORE per-function superoptimisation. It walks the call graph in
 * reverse-topological order (leaves first) and looks for functions (or
 * compositions of functions) whose output is a *predictable* closed-form
 * function of the input:
 *
 *   1. Constant function:        f(x) ≡ C          for all x
 *   2. Multiplicative scaling:   f(x) ≡ c * x      for some constant c
 *   3. Affine function:          f(x) ≡ c * x + b  for constants c, b
 *   4. Composition collapse:     g(f(x)) ≡ h(x)    where h is one of the
 *                                above, or a one-instruction binop over x
 *
 * When such a pattern is detected and SMT-proven, the original function
 * body is rewritten to the minimal closed form (e.g. `ret i32 5` for a
 * constant function, `ret i32 (shl i32 %x, 1)` for c*x with c=2). The
 * rewritten function is then a much smaller target for the per-function
 * pipeline (and often the search converges immediately).
 *
 * Why this is sound:
 *   - Every rewrite is SMT-proven via SMTVerifier::verify() against the
 *     ORIGINAL function (Alive2 may then second-opinion veto it via the
 *     pipeline's standard machinery).
 *   - When Z3 is unavailable, the pass returns false (it cannot prove
 *     anything without a real prover).
 *   - When the candidate fails SMT (NotEquivalent), the original is
 *     preserved unchanged.
 *
 * Detection strategy:
 *   - Constant: probe f(0), f(1), f(-1), f(2), f(-2), f(small_primes).
 *     If all return the same value AND that value is reproducible on
 *     the random-probe set, conjecture f(x) ≡ c, then SMT-prove.
 *   - Affine: probe f(0) → b, f(1) → c+b, f(2) → 2c+b, f(-1) → -c+b.
 *     If a consistent (c, b) emerges across ≥4 probes, conjecture
 *     f(x) ≡ c*x + b, then SMT-prove.
 *   - Composition: for a caller that calls exactly one callee in
 *     sequence (or a small chain of single-arg callees), probe the
 *     composed behaviour and check whether it matches one of the above
 *     closed forms. If so, SMT-prove the composition-equivalence and
 *     inline the closed form into the caller.
 *
 * Scope (intentionally narrow):
 *   - Single-block integer functions only (SMT verifier refuses
 *     everything else anyway).
 *   - Up to 4 integer arguments.
 *   - Function instruction count ≤ max_function_instructions (default 64).
 *   - Composition chains up to 4 deep.
 */
#include <memory>
#include <string>
#include <vector>

#include "clunk/IR/Module.h"
#include "clunk/Search/SMTVerifier.h"

namespace clunk::search {

struct AlgoPreConfig {
    // SMT timeout per proof attempt, in milliseconds.
    unsigned smt_timeout_ms = 5000;

    // Functions above this instruction count are skipped.
    size_t max_function_instructions = 64;

    // Cap on the number of probe values used for pattern detection.
    // (16 is enough to detect any reasonable affine / constant pattern.)
    size_t max_probes = 16;

    // Cap on composition-chain depth (caller → callee → callee → ...).
    size_t max_composition_depth = 4;

    // Return unproven rewrites when SMT is unavailable. Off by default:
    // no proof, no rewrite.
    bool trust_unverified = false;
};

class AlgoPreprocessor {
public:
    explicit AlgoPreprocessor(const AlgoPreConfig& config = {})
        : config_(config) {}

    // Run on `module` IN PLACE. Returns true iff the module was modified
    // (at least one function was rewritten to a minimal closed form).
    bool run(ir::Module& module);

    struct Stats {
        size_t functions_scanned = 0;
        size_t functions_skipped = 0;        // out of scope
        size_t constants_detected = 0;       // f(x) ≡ C
        size_t scalars_detected = 0;         // f(x) ≡ c*x
        size_t affines_detected = 0;         // f(x) ≡ c*x + b
        size_t compositions_detected = 0;    // g(f(x)) collapsed
        size_t rewrites_proven = 0;          // SMT-proven and applied
        size_t rewrites_rejected_by_smt = 0; // SMT refuted
        size_t rewrites_rejected_by_score = 0; // proven but not cheaper
    };
    const Stats& stats() const { return stats_; }

    AlgoPreConfig& config() { return config_; }
    const AlgoPreConfig& config() const { return config_; }

private:
    AlgoPreConfig config_;
    Stats stats_{};
};

} // namespace clunk::search
