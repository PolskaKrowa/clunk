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
 * candidate is the SHORTEST equivalent form found.
 *
 * ── Parallel mode (per-function multithreading, work-stealing) ──────
 * When a ThreadPool is attached via set_thread_pool(), each depth's
 * search space is split into independent "first-step" work items. Each
 * work item is one (opcode, lhs, rhs) triple for the FIRST instruction
 * in the candidate sequence; the remaining depth-1 instructions are
 * enumerated recursively INSIDE the worker. Each worker collects ALL
 * verified candidates in its subtree (NOT short-circuiting on the first
 * hit), and the main thread picks the verified candidate with the
 * lowest score at the end of the depth. This implements the user's
 * "keep searching the allocated 2-instruction space" requirement: a
 * thread that finds a short equivalent early does NOT stop the other
 * threads; they continue searching their own subtrees in case a faster
 * equivalent exists at the same depth.
 *
 * The parallel path uses the pool's run_until() helper to wait on
 * subtask futures — this lets a pool worker that submitted subtasks
 * participate in draining the queue, avoiding the classic nested-
 * parallelism deadlock (all N workers blocked on subtasks, no one left
 * to run them).
 *
 * Soundness is unchanged: every returned candidate still passes the
 * test-vector pre-filter, the cost gate, and (when Z3 is available) the
 * SMT equivalence gate. The parallel path merely widens the per-depth
 * search to find a LOWER-score equivalent than the sequential
 * first-hit-wins path would have returned.
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
#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <mutex>
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

// ── Per-thread SMT verifier ────────────────────────────────────────────────
//
// SMTVerifier is NOT thread-safe across verify() calls on the same
// instance (it caches a Z3 context as `mutable void* z3_ctx_`). The
// sequential synthesiser creates one verifier per synthesize() call;
// the parallel synthesiser needs one verifier PER WORKER THREAD to
// avoid the cache race.
//
// We use a thread_local verifier keyed by the synthesiser's config.
// thread_local ensures one verifier per OS thread, and the lazy init
// reuses the verifier across all tasks run by that thread (so the Z3
// context — which costs 1-10ms to create — is built once per worker,
// not once per task). The config is re-applied on each call to handle
// the case where different HoleSynthesizer instances share the same
// thread pool with different SMT timeouts.
//
// NOTE: we deliberately LEAK the verifier (via a thread_local raw
// pointer that's never deleted). This avoids calling Z3_del_context
// during thread exit, which can race with other threads' Z3 usage and
// cause occasional hangs during pool destruction. The leaked context
// is bounded (one per thread that ever called this function) and is
// cleaned up by the OS when the process exits.
SMTVerifier& tls_verifier(unsigned smt_timeout_ms) {
    thread_local SMTVerifier* verifier = [&] {
        SMTConfig c;
        c.timeout_ms = smt_timeout_ms;
        return new SMTVerifier(c);
    }();
    // Re-apply the timeout in case a different synthesiser is using
    // this thread now. Cheap (just a struct copy).
    verifier->config().timeout_ms = smt_timeout_ms;
    return *verifier;
}

// ── Shared candidate-evaluation context ────────────────────────────────────
//
// Both the sequential and parallel paths share this. It encapsulates
// everything stateful about evaluating a single candidate: the original
// function, the precomputed probe results, the cost gate, and the SMT
// gate. The sequential path calls `try_candidate` directly; the
// parallel path calls it from inside worker tasks.
//
// Each task accumulates its OWN local Stats (so no atomics are needed
// for the hot path); the parallel runner merges them under a mutex at
// the end of each depth.
struct EvalCtx {
    const ir::Function& orig;
    unsigned width;
    size_t nargs;
    size_t k_consts;
    double orig_score;
    bool z3_available;
    bool trust_unverified;
    unsigned smt_timeout_ms;
    ir::TypeContext& type_ctx;
    evaluator::EvaluationEngine* engine;
    std::vector<std::pair<std::vector<int64_t>, int64_t>> probes;  // (args, result)

    // Evaluate one candidate spec. Returns the rewritten function (and
    // whether it was SMT-proven) if it passed all gates; nullptr
    // otherwise. `local` accumulates per-task statistics.
    // `smt_timeout_override` (if non-zero) replaces `smt_timeout_ms` for
    // the SMT call — used by the parallel path to cap each Z3 call at a
    // fraction of the time budget so in-flight calls can't block
    // termination past the budget.
    std::pair<std::shared_ptr<ir::Function>, bool> try_candidate(
        const HoleSpec& spec,
        HoleSynthesizer::Stats& local,
        unsigned smt_timeout_override = 0) const
    {
        auto cand = materialise(orig, spec, width, type_ctx);
        if (!cand) return {nullptr, false};
        ++local.candidates_enumerated;

        // Test-vector pre-filter.
        bool ok = true;
        for (const auto& [args, result] : probes) {
            auto r = evaluator::Interpreter::interpret(*cand, args);
            if (!r) { ok = true; break; }  // unsupported — let SMT decide
            if (*r != result) { ok = false; break; }
        }
        if (!ok) {
            ++local.candidates_test_vector_pruned;
            return {nullptr, false};
        }

        // Cost gate (skip if not strictly cheaper).
        if (engine) {
            auto ca = engine->analyse(*cand);
            if (!(ca.score > orig_score)) {
                ++local.rejected_by_score;
                return {nullptr, false};
            }
        }

        // SMT gate.
        bool verified = false;
        if (z3_available) {
            ++local.candidates_smt_verified;
            unsigned t = smt_timeout_override ? smt_timeout_override : smt_timeout_ms;
            auto& verifier = tls_verifier(t);
            auto res = verifier.verify(orig, *cand);
            if (res.status == VerificationResult::Equivalent) {
                verified = true;
                ++local.candidates_equivalent;
            }
        }
        if (!verified && !trust_unverified) return {nullptr, false};

        return {cand, verified};
    }
};

// One "first-step" work item for the parallel search. Each work item
// fixes the FIRST instruction's (opcode, lhs, rhs) and the worker
// enumerates all completions of the remaining depth-1 steps.
//
// `is_special` is set for depth-1 ret-arg / ret-const candidates (no
// instructions, just a ret). In that case `special_ret` holds the ret
// PoolRef and `first_step` is unused.
struct WorkItem {
    bool is_special = false;
    PoolRef special_ret{};
    HoleStep first_step{};
};

// Build the list of work items for a given depth. Returns the list,
// along with the size of the search space each item covers (for stats).
std::vector<WorkItem> build_work_items(size_t depth, size_t nargs, size_t k_consts) {
    std::vector<WorkItem> items;

    // Special case at depth 1: ret arg_i, ret const_i (zero instructions).
    if (depth == 1) {
        for (size_t i = 0; i < nargs; ++i) {
            items.push_back(WorkItem{true, {PoolRef::Arg, static_cast<uint16_t>(i)}, {}});
        }
        for (size_t i = 0; i < k_consts; ++i) {
            items.push_back(WorkItem{true, {PoolRef::Const, static_cast<uint16_t>(i)}, {}});
        }
    }

    // General case: pick the first step's (opcode, lhs, rhs).
    // The remaining depth-1 steps are enumerated inside the worker.
    auto pool0 = pool_at(nargs, k_consts, 0);
    for (ir::Opcode op : kBinops) {
        for (const auto& lhs : pool0) {
            for (const auto& rhs : pool0) {
                if (is_commutative(op)) {
                    if (lhs.kind > rhs.kind) continue;
                    if (lhs.kind == rhs.kind && lhs.index > rhs.index) continue;
                }
                if (lhs.kind == rhs.kind &&
                    lhs.index == rhs.index &&
                    op != ir::Opcode::Add && op != ir::Opcode::Mul) {
                    continue;
                }
                WorkItem wi{};
                wi.is_special = false;
                wi.first_step = {op, lhs, rhs};
                items.push_back(wi);
            }
        }
    }
    return items;
}

// ── Parallel synthesize (forward declaration) ─────────────────────────────
// Defined below; declared here so HoleSynthesizer::synthesize() can call it.
// Free function in the anonymous namespace — takes the synthesiser's
// Stats and config by reference so it can update them.
struct EvalCtx;  // defined below

std::shared_ptr<ir::Function> synthesize_parallel(
    ThreadPool& pool,
    const HoleSynthConfig& config,
    HoleSynthesizer::Stats& stats,
    bool* proven,
    const EvalCtx& ctx,
    std::chrono::steady_clock::time_point t0,
    const std::function<bool()>& time_up,
    size_t kMaxCandidatesAtDepth);

} // anonymous namespace

// ── Public constructor ────────────────────────────────────────────────────

HoleSynthesizer::HoleSynthesizer(evaluator::EvaluationEngine* engine,
                                   const HoleSynthConfig& config)
    : engine_(engine), config_(config) {}

// ── synthesize: dispatch to parallel or sequential path ───────────────────

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
    if (fn.instruction_count() <= 1) {
        ++stats_.functions_skipped;
        return nullptr;
    }

    const unsigned width = common_int_width(fn);
    const size_t nargs = fn.argument_count();
    const size_t k_consts = kConstants.size();

    auto orig_analysis = engine_ ? engine_->analyse(fn) :
                        evaluator::FunctionAnalysis{};
    const double orig_score = orig_analysis.score;

    // Pre-compute the original's interpreter behaviour on the probe set.
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

    if (probes.empty()) {
        ++stats_.functions_skipped;
        stats_.elapsed_ms = elapsed_s() * 1000.0;
        return nullptr;
    }

    const bool z3_available = SMTVerifier::is_z3_available();
    const bool trust_unverified = config_.trust_unverified;

    // Build the shared evaluation context.
    EvalCtx ctx{
        fn, width, nargs, k_consts, orig_score,
        z3_available, trust_unverified, config_.smt_timeout_ms,
        type_ctx_, engine_, {}
    };
    for (const auto& p : probes) {
        ctx.probes.push_back({p.args, p.result});
    }

    // Decide whether to engage the parallel path.
    const bool can_parallel =
        config_.parallel_search &&
        pool_ != nullptr &&
        pool_->worker_count() > 1 &&
        !pool_->is_shutdown();

    // Build the per-depth work items once (the structure is the same
    // regardless of parallel/sequential; the parallel path dispatches
    // them as tasks, the sequential path just iterates).
    // We build them lazily per depth inside the loop, since the work
    // items depend on depth.

    // Cap on candidates per depth (sequential path's historical limit).
    const size_t kMaxCandidatesAtDepth = 50000;

    // ── Parallel path ────────────────────────────────────────────────
    if (can_parallel) {
        return synthesize_parallel(*pool_, config_, stats_, proven, ctx, t0,
                                    time_up, kMaxCandidatesAtDepth);
    }

    // ── Sequential path (existing behaviour) ─────────────────────────
    // First-hit-wins: returns the FIRST verified cheaper candidate at
    // the smallest depth. This preserves the historical semantics
    // (deterministic, lowest-depth-first) for callers that don't attach
    // a pool.
    for (size_t depth = 1; depth <= config_.max_depth; ++depth) {
        if (time_up()) break;

        // Depth-1 special cases (ret arg / ret const).
        if (depth == 1) {
            for (size_t i = 0; i < nargs; ++i) {
                if (time_up()) break;
                HoleSpec spec{ {}, {PoolRef::Arg, static_cast<uint16_t>(i)} };
                auto [cand, verified] = ctx.try_candidate(spec, stats_);
                if (cand) {
                    if (proven) *proven = verified;
                    ++stats_.proven;
                    stats_.best_depth = 1;
                    stats_.elapsed_ms = elapsed_s() * 1000.0;
                    return cand;
                }
            }
            for (size_t i = 0; i < k_consts; ++i) {
                if (time_up()) break;
                HoleSpec spec{ {}, {PoolRef::Const, static_cast<uint16_t>(i)} };
                auto [cand, verified] = ctx.try_candidate(spec, stats_);
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
        size_t enumerated = 0;
        std::shared_ptr<ir::Function> found;
        bool found_verified = false;

        std::function<bool(std::vector<HoleStep>&)> recurse =
            [&](std::vector<HoleStep>& steps) -> bool {
            if (steps.size() == depth) {
                auto ret_pool = pool_at(nargs, k_consts, depth);
                for (const auto& ret_ref : ret_pool) {
                    if (enumerated >= kMaxCandidatesAtDepth) return true;
                    ++enumerated;
                    if (time_up()) return true;

                    HoleSpec spec{ steps, ret_ref };
                    auto [cand, verified] = ctx.try_candidate(spec, stats_);
                    if (cand) {
                        found = cand;
                        found_verified = verified;
                        return true;  // short-circuit
                    }
                }
                return false;
            }

            size_t pos = steps.size();
            auto pool = pool_at(nargs, k_consts, pos);
            for (ir::Opcode op : kBinops) {
                for (const auto& lhs : pool) {
                    for (const auto& rhs : pool) {
                        if (is_commutative(op)) {
                            if (lhs.kind > rhs.kind) continue;
                            if (lhs.kind == rhs.kind && lhs.index > rhs.index) continue;
                        }
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

// ── synthesize_parallel: per-function multithreading via work-stealing ───
//
// See the file-header comment for the design. Each depth's search space
// is split into "first-step" work items; each work item is dispatched
// to the pool as a task. Each task enumerates the full subtree under
// its first step, collecting ALL verified candidates. After all tasks
// for a depth complete, the main thread picks the verified candidate
// with the lowest score.
//
// We use pool.run_until() to wait on task futures — this lets a pool
// worker that submitted tasks participate in draining the queue,
// avoiding the nested-parallelism deadlock (all N workers blocked on
// subtasks, no one left to run them).
//
// This is a free function inside the anonymous namespace — declared
// above and called from HoleSynthesizer::synthesize(). It takes the
// synthesiser's Stats and config by reference so it can update them.
namespace {

std::shared_ptr<ir::Function> synthesize_parallel(
    ThreadPool& pool,
    const HoleSynthConfig& config,
    HoleSynthesizer::Stats& stats,
    bool* proven,
    const EvalCtx& ctx,
    std::chrono::steady_clock::time_point t0,
    const std::function<bool()>& time_up,
    size_t kMaxCandidatesAtDepth) {

    auto elapsed_s = [&]() {
        return std::chrono::duration<double>(
                   std::chrono::steady_clock::now() - t0).count();
    };

    const size_t nargs = ctx.nargs;
    const size_t k_consts = ctx.k_consts;

    // Cap the per-candidate SMT timeout at a fraction of the remaining
    // time budget. Without this, a single Z3 call could run for the
    // full smt_timeout_ms (default 30s) — much longer than the
    // synthesiser's own time_budget_seconds (default 2s). When the
    // time budget expires, in-flight SMT calls can't be interrupted,
    // so the parallel path would have to wait for them to finish.
    // Capping the SMT timeout at 1/4 of the time budget (or 500ms,
    // whichever is smaller) ensures the parallel path terminates
    // promptly after the time budget expires.
    unsigned effective_smt_timeout = ctx.smt_timeout_ms;
    if (config.time_budget_seconds > 0.0) {
        unsigned budget_cap_ms = static_cast<unsigned>(
            config.time_budget_seconds * 1000.0 / 4.0);
        if (budget_cap_ms < 500) budget_cap_ms = 500;
        if (effective_smt_timeout > budget_cap_ms) {
            effective_smt_timeout = budget_cap_ms;
        }
    }

    // Per-depth results: a verified candidate and its score. Updated
    // by workers under `results_mutex`.
    struct VerifiedCandidate {
        std::shared_ptr<ir::Function> function;
        bool proven = false;
        double score = 0.0;  // lower is better
    };

    // Atomic shared state across tasks at a given depth.
    std::atomic<size_t> global_enumerated{0};
    std::atomic<bool> global_time_up{false};

    for (size_t depth = 1; depth <= config.max_depth; ++depth) {
        if (time_up() || global_time_up.load(std::memory_order_relaxed)) break;

        global_enumerated.store(0, std::memory_order_relaxed);

        // Build the work items for this depth.
        auto items = build_work_items(depth, nargs, k_consts);
        if (items.empty()) continue;

        // If the work-item count is too small to justify parallel
        // dispatch, run them sequentially inline (still collecting ALL
        // verified candidates and picking the best — so the semantics
        // match the parallel path, just without the pool overhead).
        const bool use_pool =
            items.size() >= config.parallel_min_work_items &&
            pool.worker_count() > 1;

        std::mutex results_mutex;
        std::vector<VerifiedCandidate> results;
        std::mutex stats_mutex;
        HoleSynthesizer::Stats merged_local{};

        // Worker lambda: process one work item. Each task gets its own
        // local Stats; we merge them under stats_mutex at the end.
        auto process_item = [&](const WorkItem& wi) {
            HoleSynthesizer::Stats local{};

            // Special case: depth-1 ret-arg / ret-const.
            if (wi.is_special) {
                if (global_time_up.load(std::memory_order_relaxed)) return;
                HoleSpec spec{ {}, wi.special_ret };
                auto [cand, verified] = ctx.try_candidate(spec, local, effective_smt_timeout);
                if (cand) {
                    VerifiedCandidate vc;
                    vc.function = cand;
                    vc.proven = verified;
                    vc.score = ctx.engine ? ctx.engine->analyse(*cand).score : 0.0;
                    std::lock_guard<std::mutex> lk(results_mutex);
                    results.push_back(std::move(vc));
                }
                {
                    std::lock_guard<std::mutex> lk(stats_mutex);
                    merged_local.candidates_enumerated += local.candidates_enumerated;
                    merged_local.candidates_test_vector_pruned += local.candidates_test_vector_pruned;
                    merged_local.candidates_smt_verified += local.candidates_smt_verified;
                    merged_local.candidates_equivalent += local.candidates_equivalent;
                    merged_local.rejected_by_score += local.rejected_by_score;
                }
                return;
            }

            // General case: enumerate the remaining depth-1 steps under
            // this fixed first step. Collect ALL verified candidates
            // (do NOT short-circuit on first hit — the user explicitly
            // wants other threads to keep searching their allocated
            // space at the same depth in case a faster equivalent
            // exists).
            std::vector<HoleStep> base_steps;
            base_steps.reserve(depth);
            base_steps.push_back(wi.first_step);

            std::function<void(std::vector<HoleStep>&)> recurse =
                [&](std::vector<HoleStep>& steps) {
                    if (steps.size() == depth) {
                        auto ret_pool = pool_at(nargs, k_consts, depth);
                        for (const auto& ret_ref : ret_pool) {
                            size_t enumerated = global_enumerated.fetch_add(1);
                            if (enumerated >= kMaxCandidatesAtDepth) return;
                            if (global_time_up.load(std::memory_order_relaxed)) return;
                            if (time_up()) {
                                global_time_up.store(true, std::memory_order_relaxed);
                                return;
                            }

                            HoleSpec spec{ steps, ret_ref };
                            auto [cand, verified] = ctx.try_candidate(spec, local, effective_smt_timeout);
                            if (cand) {
                                VerifiedCandidate vc;
                                vc.function = cand;
                                vc.proven = verified;
                                vc.score = ctx.engine ? ctx.engine->analyse(*cand).score : 0.0;
                                std::lock_guard<std::mutex> lk(results_mutex);
                                results.push_back(std::move(vc));
                            }
                        }
                        return;
                    }

                    size_t pos = steps.size();
                    auto pool = pool_at(nargs, k_consts, pos);
                    for (ir::Opcode op : kBinops) {
                        for (const auto& lhs : pool) {
                            for (const auto& rhs : pool) {
                                if (is_commutative(op)) {
                                    if (lhs.kind > rhs.kind) continue;
                                    if (lhs.kind == rhs.kind && lhs.index > rhs.index) continue;
                                }
                                if (lhs.kind == rhs.kind &&
                                    lhs.index == rhs.index &&
                                    op != ir::Opcode::Add && op != ir::Opcode::Mul) {
                                    continue;
                                }
                                if (global_time_up.load(std::memory_order_relaxed)) return;

                                steps.push_back({op, lhs, rhs});
                                recurse(steps);
                                steps.pop_back();
                            }
                        }
                    }
                };

            recurse(base_steps);

            {
                std::lock_guard<std::mutex> lk(stats_mutex);
                merged_local.candidates_enumerated += local.candidates_enumerated;
                merged_local.candidates_test_vector_pruned += local.candidates_test_vector_pruned;
                merged_local.candidates_smt_verified += local.candidates_smt_verified;
                merged_local.candidates_equivalent += local.candidates_equivalent;
                merged_local.rejected_by_score += local.rejected_by_score;
            }
        };

        if (use_pool) {
            // Dispatch work items to the pool. We submit them in batches
            // and use pool.run_until() to drain the queue while we
            // wait — this is what prevents the nested-parallelism
            // deadlock when this thread is itself a pool worker.
            //
            // We check global_time_up BEFORE each submission so we don't
            // flood the pool with subtasks when the time budget has
            // expired. Without this, the pool workers could be stuck in
            // long SMT calls (up to smt_timeout_ms each) for minutes
            // after the time budget expires, because the already-
            // submitted subtasks can't be cancelled mid-flight.
            // Dispatch work items in BATCHES of pool.worker_count().
            // This limits the number of in-flight subtasks and ensures
            // fair scheduling when multiple functions are searching in
            // parallel (each function gets at most worker_count subtasks
            // in the queue at a time, rather than flooding the queue
            // with thousands of subtasks that starve other functions).
            const size_t batch_size = std::max(pool.worker_count(), (size_t)1);
            for (size_t start = 0; start < items.size(); start += batch_size) {
                if (global_time_up.load(std::memory_order_relaxed)) break;
                if (time_up()) {
                    global_time_up.store(true, std::memory_order_relaxed);
                    break;
                }
                size_t end = std::min(start + batch_size, items.size());
                std::vector<std::future<void>> batch_futs;
                batch_futs.reserve(end - start);
                for (size_t i = start; i < end; ++i) {
                    batch_futs.push_back(pool.submit(process_item, items[i]));
                }
                // Wait for this batch to complete. Steal work while
                // waiting to prevent the nested-parallelism deadlock.
                for (auto& f : batch_futs) {
                    pool.run_until(f);
                    try { f.get(); } catch (...) {}
                }
            }
        } else {
            // Sequential inline execution (small work-item count).
            for (const auto& wi : items) {
                process_item(wi);
            }
        }

        // Merge the per-task stats into the synthesiser's stats.
        stats.candidates_enumerated += merged_local.candidates_enumerated;
        stats.candidates_test_vector_pruned += merged_local.candidates_test_vector_pruned;
        stats.candidates_smt_verified += merged_local.candidates_smt_verified;
        stats.candidates_equivalent += merged_local.candidates_equivalent;
        stats.rejected_by_score += merged_local.rejected_by_score;
        ++stats.parallel_depths_run;
        stats.parallel_work_items_dispatched += items.size();
        if (items.size() > 0) {
            stats.parallel_candidates_per_work_item_avg =
                static_cast<double>(merged_local.candidates_enumerated) /
                static_cast<double>(items.size());
        }

        // Pick the best (highest-score = lowest-cost) verified candidate
        // at this depth. Note: score = -cost, so HIGHER score is BETTER
        // (less negative = lower cost). We use max_element to pick the
        // candidate with the highest score.
        if (!results.empty()) {
            auto best_it = std::max_element(results.begin(), results.end(),
                [](const VerifiedCandidate& a, const VerifiedCandidate& b) {
                    return a.score < b.score;
                });
            if (best_it != results.end()) {
                if (proven) *proven = best_it->proven;
                ++stats.proven;
                stats.best_depth = depth;
                stats.elapsed_ms = elapsed_s() * 1000.0;
                return best_it->function;
            }
        }
        // No verified candidate at this depth — advance to depth+1.
    }

    stats.elapsed_ms = elapsed_s() * 1000.0;
    return nullptr;
}

} // anonymous namespace

} // namespace clunk::search
