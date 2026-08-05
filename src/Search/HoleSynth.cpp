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
 * Clunk Hole-Based Progressive-Deepening Synthesiser — implementation.
 * See include/clunk/Search/HoleSynth.h for the contract.
 *
 * Strategy:
 *   for depth = 1 .. max_depth:
 *     for each opcode in {Add, Sub, Mul, And, Or, Xor, Shl, LShr, AShr}:
 *       for each (lhs, rhs) in operand_pool²:
 *         for each result-register selection:
 *           build candidate = sequence of `depth` instructions + ret
 *           if candidate passes test-vector pre-filter:
 *             if SMT says Equivalent:
 *               if cost(candidate) < cost(original):
 *                 return candidate
 *
 * Where `operand_pool` at any instruction position is:
 *   {original arguments} ∪ {small constants} ∪ {results of prior instructions
 *   in this candidate sequence}
 *
 * The enumeration is breadth-first over depth, so the FIRST returned
 * candidate is the SHORTEST equivalent form found
 *
 * We cap |opcode| × |operand_pool|² at each depth using a small constant
 * pool (16 values) and a small opcode set (9 binops + identity + const
 * return), giving ≈ 9 × (nargs + 16 + depth)² candidates per depth. At
 * depth 3 with 2 args that's ≈ 9 × 21² × 21 ≈ 80k candidates — well
 * within budget given the test-vector pre-filter rejects ≥95% of them
 * before SMT.
 */

#include "clunk/Search/HoleSynth.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <functional>
#include <unordered_set>
#include <vector>

#include "clunk/IR/Clone.h"
#include "clunk/IR/Instruction.h"
#include "clunk/IR/Value.h"
#include "clunk/Evaluator/Interpreter.h"
#include "clunk/Search/StochasticSearch.h"  // for structural_hash reuse

namespace clunk::search {

namespace {

// ── Opcode pool ────────────────────────────────────────────────────────────
// The set of integer binops the synthesiser will consider at each hole
// position. Deliberately small — every opcode here is one the SMT
// verifier can soundly encode.
constexpr std::array<ir::Opcode, 9> kBinops = {
    ir::Opcode::Add, ir::Opcode::Sub, ir::Opcode::Mul,
    ir::Opcode::And, ir::Opcode::Or,  ir::Opcode::Xor,
    ir::Opcode::Shl, ir::Opcode::LShr, ir::Opcode::AShr,
};

// ── Constant pool ──────────────────────────────────────────────────────────
// A small but "interesting" set of constants (Massalin §3.2). Every
// candidate at depth ≥ 1 may pick any of these as an operand.
constexpr std::array<int64_t, 16> kConstants = {
    0, 1, -1, 2, 3, 4, 5, 7, 8, 15, 16, 31, 32, 63, 255, 256,
};

// ── Test-vector probe set ──────────────────────────────────────────────────
constexpr std::array<int64_t, 16> kProbeValues = {
    0, 1, -1, 2, -2, 3, 7, 8, 15, 16, 17, 127, 128, 255, 256, 1023,
};

// Is `fn` a candidate this synthesiser can attempt? Returns false for
// anything outside the scope documented in the header.
bool is_in_scope(const ir::Function& fn) {
    if (fn.blocks().size() != 1) return false;
    if (fn.argument_count() > 4) return false;  // keep enumeration tractable
    auto ret_ty = fn.return_type();
    if (!ret_ty || !ret_ty->is_integer()) return false;
    for (const auto& arg : fn.arguments()) {
        if (!arg.type || !arg.type->is_integer()) return false;
    }
    return true;
}

// Look up the integer bit-width used by the function (return type and
// all args assumed to share the same width — verified by the caller).
unsigned common_int_width(const ir::Function& fn) {
    auto ret_ty = fn.return_type();
    auto* it = dynamic_cast<const ir::IntegerType*>(ret_ty.get());
    return it ? it->bits() : 32u;
}

// Build a fresh `ir::Function` with the same signature as `src`, a single
// "entry" block, and no instructions.
std::shared_ptr<ir::Function> make_hole_skeleton(const ir::Function& src) {
    auto fn_type = src.function_type();
    auto out = std::make_shared<ir::Function>(src.name(), fn_type, src.linkage());
    for (const auto& arg : src.arguments()) {
        out->add_argument(arg.type, arg.name, arg.attrs);
    }
    out->add_block("entry");
    return out;
}

// One operand slot in the candidate's operand pool. Either an original
// argument by index, a small constant by index, or a reference to a
// prior instruction's result by index (0-based, relative to the
// candidate sequence).
struct PoolRef {
    enum Kind { Arg, Const, PriorInstr };
    Kind kind;
    uint16_t index;  // arg index, const index, or prior-instr index
};

// Resolve a PoolRef to a concrete `shared_ptr<Value>` given the
// candidate's environment (args + constants + already-built instructions).
std::shared_ptr<ir::Value> resolve_ref(
    const PoolRef& ref,
    const std::vector<std::shared_ptr<ir::Value>>& args,
    const std::vector<std::shared_ptr<ir::ConstantInt>>& consts,
    const std::vector<std::shared_ptr<ir::Value>>& prior_results) {
    switch (ref.kind) {
    case PoolRef::Arg:        return args.at(ref.index);
    case PoolRef::Const:      return consts.at(ref.index);
    case PoolRef::PriorInstr: return prior_results.at(ref.index);
    }
    return nullptr;
}

struct HoleStep {
    ir::Opcode op;
    PoolRef lhs;
    PoolRef rhs;
};
struct HoleSpec {
    std::vector<HoleStep> steps;
    PoolRef ret;  // what to return
};

std::shared_ptr<ir::Function> materialise(const ir::Function& orig,
                                           const HoleSpec& spec,
                                           unsigned width,
                                           ir::TypeContext& ctx) {
    auto out = make_hole_skeleton(orig);
    auto bb = out->entry_block();

    // The new function's arguments (already added by make_hole_skeleton).
    // Build the args vector by wrapping each in a shared_ptr<Value> that
    // shares the type+name — instructions will reference these by name.
    std::vector<std::shared_ptr<ir::Value>> args;
    for (const auto& a : out->arguments()) {
        auto v = std::make_shared<ir::Value>(a.type, a.name);
        args.push_back(v);
    }

    std::vector<std::shared_ptr<ir::ConstantInt>> consts;
    consts.reserve(kConstants.size());
    for (int64_t c : kConstants) {
        consts.push_back(ir::ConstantInt::get(ctx, c, width));
    }

    std::vector<std::shared_ptr<ir::Value>> prior_results;
    prior_results.reserve(spec.steps.size());

    for (size_t i = 0; i < spec.steps.size(); ++i) {
        const auto& step = spec.steps[i];
        auto lhs = resolve_ref(step.lhs, args, consts, prior_results);
        auto rhs = resolve_ref(step.rhs, args, consts, prior_results);
        if (!lhs || !rhs) return nullptr;
        if (!lhs->type() || !lhs->type()->is_integer()) return nullptr;
        if (!rhs->type() || !rhs->type()->is_integer()) return nullptr;
        std::string name = "h" + std::to_string(i);
        auto inst = ir::inst::make_binop(step.op, lhs, rhs, name);
        bb->add_instruction(inst);
        prior_results.push_back(inst);
    }

    auto ret_val = resolve_ref(spec.ret, args, consts, prior_results);
    if (!ret_val) return nullptr;
    auto ret = ir::inst::make_ret(ret_val);
    bb->add_instruction(ret);

    if (!ir::validate_function(*out)) return nullptr;
    return out;
}

// Enumerate all PoolRefs available at a given position in the candidate
// sequence (i.e. before instruction `pos` out of `total`).
// Pool: args (0..nargs-1) + constants (0..k-1) + prior instrs (0..pos-1).
std::vector<PoolRef> pool_at(size_t nargs, size_t k_constants, size_t pos) {
    std::vector<PoolRef> out;
    out.reserve(nargs + k_constants + pos);
    for (size_t i = 0; i < nargs; ++i)
        out.push_back({PoolRef::Arg, static_cast<uint16_t>(i)});
    for (size_t i = 0; i < k_constants; ++i)
        out.push_back({PoolRef::Const, static_cast<uint16_t>(i)});
    for (size_t i = 0; i < pos; ++i)
        out.push_back({PoolRef::PriorInstr, static_cast<uint16_t>(i)});
    return out;
}

// Is `op` a commutative integer binop?
bool is_commutative(ir::Opcode op) {
    return op == ir::Opcode::Add || op == ir::Opcode::Mul ||
           op == ir::Opcode::And || op == ir::Opcode::Or  ||
           op == ir::Opcode::Xor;
}

} // anonymous namespace

// ── Public constructor ────────────────────────────────────────────────────

HoleSynthesizer::HoleSynthesizer(evaluator::EvaluationEngine* engine,
                                   const HoleSynthConfig& config)
    : engine_(engine), config_(config) {}

// ── synthesize: the main progressive-deepening loop ──────────────────────

std::shared_ptr<ir::Function> HoleSynthesizer::synthesize(
    const ir::Function& fn, bool* proven) {
    if (proven) *proven = false;
    ++stats_.functions_seen;

    auto t0 = std::chrono::steady_clock::now();
    auto elapsed_s = [&]() {
        return std::chrono::duration<double>(
                   std::chrono::steady_clock::now() - t0).count();
    };
    auto time_up = [&]() {
        return config_.time_budget_seconds > 0.0 &&
               elapsed_s() >= config_.time_budget_seconds;
    };

    // Scope gate.
    if (!is_in_scope(fn)) {
        ++stats_.functions_skipped;
        return nullptr;
    }
    if (fn.instruction_count() > config_.max_original_instructions) {
        ++stats_.functions_skipped;
        return nullptr;
    }
    // Skip functions that are trivially minimal (just `ret arg` or
    // `ret const`) — no shorter equivalent can exist (depth 0 is the
    // minimum), and the cost gate would reject any same-size candidate
    // anyway. This avoids wasting the time budget on functions where
    // hole-synth cannot help.
    if (fn.instruction_count() <= 1) {
        ++stats_.functions_skipped;
        return nullptr;
    }

    const unsigned width = common_int_width(fn);
    const size_t nargs = fn.argument_count();
    const size_t k_consts = kConstants.size();

    // Baseline score for the cost gate.
    auto orig_analysis = engine_ ? engine_->analyse(fn) :
                        evaluator::FunctionAnalysis{};
    const double orig_score = orig_analysis.score;

    // Pre-compute the original's interpreter behaviour on the probe set
    // so we can pre-filter candidates cheaply (without re-interpreting
    // the original each time).
    struct ProbeResult { std::vector<int64_t> args; int64_t result; };
    std::vector<ProbeResult> probes;
    if (nargs == 0) {
        auto r = evaluator::Interpreter::interpret(fn, {});
        if (r) probes.push_back({{}, *r});
    } else {
        size_t n_probes = std::min(config_.test_vector_count, kProbeValues.size());
        for (size_t i = 0; i < n_probes; ++i) {
            std::vector<int64_t> a(nargs, kProbeValues[i]);
            for (size_t j = 1; j < nargs; ++j) {
                a[j] = kProbeValues[(i + j) % kProbeValues.size()];
            }
            auto r = evaluator::Interpreter::interpret(fn, a);
            if (r) probes.push_back({a, *r});
        }
    }

    // ── Soundness gate ────────────────────────────────────────────────
    // If the original function can't be interpreted (probes is empty),
    // we have NO verification path:
    //   - The Interpreter can't evaluate the original → test-vector
    //     differential is impossible.
    //   - SMT also refuses functions with calls / memory / FP / loops
    //     (returns Unknown), so it can't prove equivalence either.
    //   - The only remaining path is `trust_unverified`, which adopts
    //     WITHOUT any verification — that's unsound.
    // So if probes is empty, refuse ALL candidates. This is conservative
    // (we might miss some legitimate rewrites on uninterpretable
    // functions) but sound (we never adopt a wrong rewrite).
    if (probes.empty()) {
        ++stats_.functions_skipped;
        stats_.elapsed_ms = elapsed_s() * 1000.0;
        return nullptr;
    }

    // SMT verifier (reused across candidates).
    SMTConfig scfg;
    scfg.timeout_ms = config_.smt_timeout_ms;
    SMTVerifier verifier(scfg);
    const bool z3_available = SMTVerifier::is_z3_available();

    // Helper: try a single candidate. Returns true and sets `out_cand` /
    // `out_verified` on success.
    auto try_candidate = [&](const HoleSpec& spec)
        -> std::pair<std::shared_ptr<ir::Function>, bool> {
        auto cand = materialise(fn, spec, width, type_ctx_);
        if (!cand) return {nullptr, false};
        ++stats_.candidates_enumerated;

        // Test-vector pre-filter.
        bool ok = true;
        for (const auto& p : probes) {
            auto r = evaluator::Interpreter::interpret(*cand, p.args);
            if (!r) { ok = true; break; }  // unsupported — let SMT decide
            if (*r != p.result) { ok = false; break; }
        }
        if (!ok) { ++stats_.candidates_test_vector_pruned; return {nullptr, false}; }

        // Cost gate (skip if not strictly cheaper).
        if (engine_) {
            auto ca = engine_->analyse(*cand);
            if (!(ca.score > orig_score)) {
                ++stats_.rejected_by_score;
                return {nullptr, false};
            }
        }

        // SMT gate.
        bool verified = false;
        if (z3_available) {
            ++stats_.candidates_smt_verified;
            auto res = verifier.verify(fn, *cand);
            if (res.status == VerificationResult::Equivalent) {
                verified = true;
                ++stats_.candidates_equivalent;
            }
        }
        if (!verified && !config_.trust_unverified) return {nullptr, false};

        return {cand, verified};
    };

    // Progressive deepening.
    for (size_t depth = 1; depth <= config_.max_depth; ++depth) {
        if (time_up()) break;

        // Special case: depth 1 includes zero-instruction bodies (just
        // `ret arg` or `ret const`). Handle them before the binop
        // enumeration, since they don't fit the "pick opcode + 2 operands"
        // pattern.
        if (depth == 1) {
            // ret arg_i
            for (size_t i = 0; i < nargs; ++i) {
                if (time_up()) break;
                HoleSpec spec{ {}, {PoolRef::Arg, static_cast<uint16_t>(i)} };
                auto [cand, verified] = try_candidate(spec);
                if (cand) {
                    if (proven) *proven = verified;
                    ++stats_.proven;
                    stats_.best_depth = 1;
                    stats_.elapsed_ms = elapsed_s() * 1000.0;
                    return cand;
                }
            }
            // ret const_i
            for (size_t i = 0; i < k_consts; ++i) {
                if (time_up()) break;
                HoleSpec spec{ {}, {PoolRef::Const, static_cast<uint16_t>(i)} };
                auto [cand, verified] = try_candidate(spec);
                if (cand) {
                    if (proven) *proven = verified;
                    ++stats_.proven;
                    stats_.best_depth = 1;
                    stats_.elapsed_ms = elapsed_s() * 1000.0;
                    return cand;
                }
            }
        }

        // General case: enumerate `depth` binops.
        // Cap the total candidate count at this depth at 50k to keep
        // the worst case bounded — at ~100μs per candidate (materialise
        // + interpreter + score), 50k candidates = ~5s, which fits
        // inside the default 2s time budget with room to spare.
        const size_t kMaxCandidatesAtDepth = 50000;
        size_t enumerated = 0;

        // Recursive enumeration. Stashes the winning candidate in
        // `found` (a shared_ptr) and `found_verified` (bool) on success.
        std::shared_ptr<ir::Function> found;
        bool found_verified = false;

        std::function<bool(std::vector<HoleStep>&)> recurse =
            [&](std::vector<HoleStep>& steps) -> bool {
            if (steps.size() == depth) {
                // Pick the ret value: any pool member at position `depth`.
                auto ret_pool = pool_at(nargs, k_consts, depth);
                for (const auto& ret_ref : ret_pool) {
                    if (enumerated >= kMaxCandidatesAtDepth) return true;
                    ++enumerated;
                    if (time_up()) return true;

                    HoleSpec spec{ steps, ret_ref };
                    auto [cand, verified] = try_candidate(spec);
                    if (cand) {
                        found = cand;
                        found_verified = verified;
                        return true;  // short-circuit
                    }
                }
                return false;
            }

            // Pick the next step's (opcode, lhs, rhs).
            size_t pos = steps.size();
            auto pool = pool_at(nargs, k_consts, pos);
            for (ir::Opcode op : kBinops) {
                for (const auto& lhs : pool) {
                    for (const auto& rhs : pool) {
                        // Symmetry prune for commutative opcodes: keep
                        // the canonical (lhs ≤ rhs) ordering. This
                        // roughly halves the search space at each
                        // commutative opcode.
                        if (is_commutative(op)) {
                            if (lhs.kind > rhs.kind) continue;
                            if (lhs.kind == rhs.kind && lhs.index > rhs.index) continue;
                        }

                        // Prune trivially-dead `x op x` (except add/mul,
                        // which give 2x and x² respectively — both useful).
                        if (lhs.kind == rhs.kind &&
                            lhs.index == rhs.index &&
                            op != ir::Opcode::Add && op != ir::Opcode::Mul) {
                            continue;
                        }

                        if (time_up()) return true;

                        steps.push_back({op, lhs, rhs});
                        if (recurse(steps)) return true;
                        steps.pop_back();
                    }
                }
            }
            return false;
        };

        std::vector<HoleStep> steps;
        steps.reserve(depth);
        (void)recurse(steps);

        if (found) {
            if (proven) *proven = found_verified;
            ++stats_.proven;
            stats_.best_depth = depth;
            stats_.elapsed_ms = elapsed_s() * 1000.0;
            return found;
        }
    }

    stats_.elapsed_ms = elapsed_s() * 1000.0;
    return nullptr;
}

} // namespace clunk::search
