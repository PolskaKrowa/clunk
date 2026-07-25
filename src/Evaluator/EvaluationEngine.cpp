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
 * Clunk Evaluation Engine — static semantic analysis using weighted heuristics.
 * The heart of Clunk: reasons about code properties without executing it.
 *
 * ────────────────────────────────────────────────────────────────────────────
 * SIGN / RATIO CONVENTION:
 *
 *   score = -cost,  cost ≥ 0    ("more negative = worse")
 *
 *   score_candidate_with_cached_orig() returns:
 *       ratio > 1.0  → candidate BETTER (cheaper) than original
 *       ratio == 1.0 → equivalent
 *       ratio < 1.0  → candidate WORSE  (more expensive) than original
 *
 *   Implementation: ratio = orig_abs / cand_abs  (both abs values are
 *   strictly positive after the no-NaN/Inf guard).
 *
 *   Callers in StochasticSearch.cpp (`if (candidate_score > baseline_score)`)
 *   and EvolutionarySearch.cpp rely on this `higher-ratio == better`
 *   convention.
 * ────────────────────────────────────────────────────────────────────────────
 */
#include "clunk/Evaluator/EvaluationEngine.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>
#include <unordered_map>

#include "clunk/IR/Instruction.h"
#include "clunk/IR/LoopAnalysis.h"
#include "clunk/IR/Scalarizer.h"
#include "clunk/IR/Type.h"

namespace clunk::evaluator {

namespace {

// Numerical safety: return a finite, non-NaN value. If `v` is NaN/Inf
// or otherwise unsuitable for arithmetic, returns 0.0.
double safe_finite(double v) {
    if (!std::isfinite(v)) return 0.0;
    return v;
}

// ── §2.1 block-frequency weighting ──────────────────────────────────────
//
// Estimate how many times a basic block executes per call, relative to a
// block outside any loop (frequency 1.0). This is the single frequency
// estimate shared by task-weight estimation, critical-path computation,
// and the per-block cost accumulation in analyse().
//
// Heuristic: 8x per level of loop nesting containing the block, capped at
// 64x. This is LLVM BlockFrequencyInfo's classic default for an
// unprofiled back-edge ("loop body is likely taken relative to its
// preheader") — we don't have real profile data or a general variable
// trip-count estimator (that's §1.4/bounded model checking territory),
// so a fixed geometric heuristic keyed to nesting depth is the accurate,
// tractable middle ground: it distinguishes a doubly-nested inner loop
// from a single loop from straight-line code, without pretending to know
// an exact trip count we don't have.
double block_loop_frequency(const std::string& block_name,
                             const std::vector<ir::NaturalLoop>& loops) {
    double freq = 1.0;
    for (const auto& loop : loops) {
        if (loop.contains(block_name)) freq *= 8.0;
    }
    return std::min(freq, 64.0);
}

// Price a `clunk.vector.*` intrinsic call as the hardware sequence it
// lowers to, instead of the CostModel's opaque-call cost (20-30 cycles,
// which would make every synthesised intrinsic look like a regression).
// A horizontal reduction over N lanes lowers to ceil(log2 N) shuffle+op
// level pairs plus one final lane extract. Returns false if `inst` is
// not a priced intrinsic (caller falls back to cost(Opcode::Call)).
bool vector_intrinsic_cost(const ir::Instruction& inst,
                           const CostModel& cm, OpCost& out) {
    if (inst.opcode() != ir::Opcode::Call) return false;
    auto it = inst.metadata().find("callee");
    if (it == inst.metadata().end()) return false;
    ir::Opcode lane_op;
    if (!ir::parse_reduce_intrinsic(it->second, &lane_op)) return false;

    // Lane count from the operand's vector type (fallback: 4).
    uint64_t lanes = 4;
    if (inst.num_operands() >= 1 && inst.operand(0) &&
        inst.operand(0)->type() && inst.operand(0)->type()->is_vector()) {
        lanes = static_cast<const ir::VectorType&>(*inst.operand(0)->type()).count();
    }
    unsigned levels = 0;
    for (uint64_t n = lanes; n > 1; n = (n + 1) / 2) ++levels;

    const OpCost op_c = cm.cost(lane_op);
    const OpCost shuf_c = cm.cost(ir::Opcode::ShuffleVector);
    const OpCost extract_c = cm.cost(ir::Opcode::ExtractElement);

    out.latency_cycles = levels * (op_c.latency_cycles + shuf_c.latency_cycles) +
                         extract_c.latency_cycles;
    out.throughput_cycles =
        levels * (op_c.throughput_cycles + shuf_c.throughput_cycles) +
        extract_c.throughput_cycles;
    out.uops = static_cast<uint16_t>(levels * (op_c.uops + shuf_c.uops) +
                                     extract_c.uops);
    out.port_mask = static_cast<uint16_t>(op_c.port_mask | shuf_c.port_mask);
    return true;
}

} // namespace

// ── Constructors ────────────────────────────────────────────────────────────

EvaluationEngine::EvaluationEngine()
    : EvaluationEngine(HeuristicWeights{}) {}

EvaluationEngine::EvaluationEngine(const HeuristicWeights& weights)
    : EvaluationEngine(weights, make_cost_model(Arch::X86_64)) {}

EvaluationEngine::EvaluationEngine(const HeuristicWeights& weights,
                                   std::shared_ptr<const CostModel> cost_model)
    : weights_(weights),
      cost_model_(cost_model ? std::move(cost_model)
                             : make_cost_model(Arch::X86_64)),
      cache_(1024) {
    validate_weights();
}

void EvaluationEngine::validate_weights() {
    // Replace any NaN/Inf weights with 0.0 to keep arithmetic sane.
    auto fix = [](double& v) {
        if (!std::isfinite(v)) v = 0.0;
    };
    fix(weights_.register_penalty);
    fix(weights_.l1_penalty);
    fix(weights_.l2_penalty);
    fix(weights_.l3_penalty);
    fix(weights_.dram_penalty);
    fix(weights_.gpu_global_penalty);
    fix(weights_.gpu_shared_penalty);
    fix(weights_.gpu_local_penalty);
    fix(weights_.nvme_penalty);
    fix(weights_.network_penalty);
    fix(weights_.task_weight_multiplier);
    fix(weights_.ecore_cache_penalty);
    fix(weights_.branch_mispredict_penalty);
    fix(weights_.load_latency_base);
    fix(weights_.store_latency_base);
    fix(weights_.div_latency_penalty);
    fix(weights_.loop_unroll_bonus);
    fix(weights_.vectorize_bonus);
    fix(weights_.register_reuse_bonus);
    // Clamp the task-weight multiplier to a non-negative value to keep
    // the weight in [0,1] after the estimate_task_weight clamp.
    if (weights_.task_weight_multiplier < 0.0) {
        weights_.task_weight_multiplier = 0.0;
    }
}

// ── Cache control ───────────────────────────────────────────────────────────

void EvaluationEngine::clear_cache() const { cache_.clear(); }

CacheStats EvaluationEngine::cache_stats() const { return cache_.stats(); }

void EvaluationEngine::set_cache_capacity(size_t capacity) const {
    cache_.set_capacity(capacity);
}

// ── Public API ─────────────────────────────────────────────────────────────

FunctionAnalysis EvaluationEngine::analyse(const ir::Function& fn) const {
    // Cache lookup: structural hash is canonical for the function's
    // opcode stream + operand types + control structure.
    const uint64_t hash = structural_hash(fn);

    if (auto cached = cache_.get(hash)) {
        // The cache stores a shared_ptr<const FunctionAnalysis>.
        // Return a *copy* so callers can mutate freely.
        return *cached;
    }

    FunctionAnalysis result;
    result.function_name = fn.name();
    result.is_gpu_kernel = fn.is_gpu_kernel();
    result.structural_hash = hash;

    // Phase 1: Count instructions by category
    analyse_instructions(fn, result);

    // Phase 2: Estimate memory access patterns
    analyse_memory_access(fn, result);

    // Phase 3: Control flow analysis (branches, loops)
    analyse_control_flow(fn, result);

    // Phase 4: Identify hot values
    identify_hot_values(fn, result);

    // Loop-frequency analysis (§2.1): computed once, reused by task-weight
    // estimation (Phase 5), critical-path computation (Phase 7), and the
    // per-block cost accumulation below (Phase 8), so all three signals
    // agree on how hot each block is instead of each guessing separately.
    const auto loop_info = ir::find_natural_loops(fn);
    result.max_block_frequency = compute_max_block_frequency(fn, loop_info);

    // Phase 5: Estimate task weight
    estimate_task_weight(result);

    // Phase 6: Decide core placement
    decide_core_placement(result);

    // Phase 7: Critical-path analysis using the CostModel, frequency-weighted
    result.critical_path_cycles = compute_critical_path(fn, loop_info);

    // Phase 8: Compute overall score
    //
    // `score` accumulates a POSITIVE cost and is negated at the end
    // (result.score = -score). The HeuristicWeights penalty fields are
    // declared NEGATIVE ("penalty"), so they must enter this accumulator
    // by magnitude — std::abs keeps user-supplied weights working under
    // either sign convention.
    double score = 0.0;

    // Instruction penalties
    for (auto& [layer, count] : result.memory_accesses) {
        score += std::abs(memory_penalty(layer)) * static_cast<double>(count);
    }
    score += std::abs(weights_.branch_mispredict_penalty) *
             static_cast<double>(result.branch_instructions);

    // Division penalty — count division instructions specifically
    for (auto& block : fn.blocks()) {
        for (auto& inst : block->instructions()) {
            auto op = inst->opcode();
            if (op == ir::Opcode::UDiv || op == ir::Opcode::SDiv ||
                op == ir::Opcode::FDiv || op == ir::Opcode::URem ||
                op == ir::Opcode::SRem || op == ir::Opcode::FRem) {
                score += std::abs(weights_.div_latency_penalty);
            }
            // Load/store base latencies are already accounted for in
            // memory_penalty, but we add the base latency on top of the
            // layer penalty.
            if (op == ir::Opcode::Load) {
                score += std::abs(weights_.load_latency_base);
            }
            if (op == ir::Opcode::Store) {
                score += std::abs(weights_.store_latency_base);
            }
        }
    }

    // Per-basic-block scoring.
    //
    // `loop_info` was already computed above (§2.1) and is reused here so
    // the per-block cost weighting agrees with task-weight and
    // critical-path (analyse() results are cached by structural hash, so
    // this small CFG pass is not hot regardless).
    //
    // The base cost is the per-instruction latency from the CostModel.
    // Without this, a function with no memory ops, no branches, and no
    // divisions would score 0 — making it impossible for the search to
    // distinguish any candidates. We use the CostModel's latency_cycles
    // (capped so an unknown opcode doesn't silently contribute 0) and
    // add the weighted penalty bonuses on top.
    for (auto& block : fn.blocks()) {
        double bb_score = 0.0;
        for (auto& inst : block->instructions()) {
            auto op = inst->opcode();
            // Base cost from the CostModel (always > 0 for known ops).
            if (cost_model_) {
                if (op == ir::Opcode::BitCast) {
                    // A bitcast is a pure reinterpretation of the same bits in
                    // the same register — LLVM/TTI treat it as free, and it
                    // never lowers to a machine instruction. Charge nothing so
                    // the search sees no phantom cost (and never "optimises"
                    // one away for a fake win).
                } else {
                    OpCost oc;
                    if (!vector_intrinsic_cost(*inst, *cost_model_, oc)) {
                        oc = cost_model_->cost(op);
                    }
                    // Use latency_cycles as the base cost. This is the
                    // minimum cost of the instruction on the critical path.
                    // Add throughput_cycles as well to model port pressure
                    // for independent instructions.
                    bb_score += oc.latency_cycles + oc.throughput_cycles;
                }
            } else {
                // Fallback: every instruction costs at least 1 cycle.
                bb_score += 1.0;
            }
            // Constant materialisation (LLVM/TTI knowledge): a ConstantInt
            // operand that does not fit a signed 32-bit immediate field cannot
            // ride along inline — it needs a separate mov / constant-pool load.
            // Charge the materialisation cost per such operand so the evaluator
            // prefers code that avoids giant literals, as real codegen and
            // constant hoisting do. Small constants inline for free.
            for (auto& op_v : inst->operands()) {
                if (auto ci = dynamic_cast<const ir::ConstantInt*>(op_v.get())) {
                    const int64_t v = ci->value();
                    if (v < -2147483648LL || v > 2147483647LL) {
                        bb_score += weights_.constant_materialisation_cost;
                    }
                }
            }
            // Memory ops. Weights enter by MAGNITUDE — bb_score is a
            // positive cost accumulator, and the penalty fields default
            // negative (see the Phase-8 comment above).
            if (op == ir::Opcode::Load) {
                bb_score += std::abs(weights_.load_latency_base);
            } else if (op == ir::Opcode::Store) {
                bb_score += std::abs(weights_.store_latency_base);
            } else if (op == ir::Opcode::Alloca) {
                bb_score += std::abs(weights_.load_latency_base) * 0.5;
            }
            // Branches
            if (op == ir::Opcode::Br || op == ir::Opcode::SwitchInst) {
                bb_score += std::abs(weights_.branch_mispredict_penalty);
            }
            // Division
            if (op == ir::Opcode::UDiv || op == ir::Opcode::SDiv ||
                op == ir::Opcode::FDiv || op == ir::Opcode::URem ||
                op == ir::Opcode::SRem || op == ir::Opcode::FRem) {
                bb_score += std::abs(weights_.div_latency_penalty);
            }
        }
        // ── Loop-frequency weighting ─────────────────────────────────────
        // A block inside a loop executes ~trip-count times, not once. A
        // purely static per-block sum can never see that, which makes
        // loop rewrites invisible: an unrolled loop (same total work,
        // fewer branches) looked strictly WORSE because its instructions
        // appear trip-count times statically. Weight each block by the
        // shared block_loop_frequency() estimate (§2.1) so per-iteration
        // costs — and per-iteration savings from LICM/unrolling — scale
        // the way execution does.
        bb_score *= block_loop_frequency(block->name(), loop_info);

        result.block_scores[block->name()] = bb_score;
        // The block score is a COST — add it to the total cost (which
        // becomes the negative score: score = -cost, less negative = better).
        score += bb_score;
    }

    // Optimisation bonuses
    double bonus_total = 0.0;
    bonus_total += weights_.loop_unroll_bonus *
                   static_cast<double>(result.estimated_loop_count);

    // Register reuse bonus: heuristic — if values are used as operands many
    // times, register allocation can keep them live.
    //
    // We compute both the SIMD type-use histogram and the operand use-count
    // map in a single pass over instructions.
    {
        std::unordered_map<std::string, size_t> type_use_count;
        std::unordered_map<std::string, size_t> operand_use_count;
        for (auto& block : fn.blocks()) {
            for (auto& inst : block->instructions()) {
                if (inst->is_binary_op() && inst->type()) {
                    type_use_count[inst->type()->to_string()]++;
                }
                for (auto& op : inst->operands()) {
                    if (op && op->has_name()) {
                        operand_use_count[op->name()]++;
                    }
                }
            }
        }

        // SIMD pattern: any type with ≥3 same-typed binary ops counts toward
        // the SIMD candidate pool.
        size_t same_type_arith = 0;
        for (auto& [ty_name, count] : type_use_count) {
            if (count >= 3) {
                same_type_arith += count;
            }
        }
        if (same_type_arith >= 4) {
            bonus_total += weights_.vectorize_bonus;
        }

        // Register reuse bonus: count values used ≥2 times, capped.
        size_t high_reuse_count = 0;
        for (auto& [name, count] : operand_use_count) {
            if (count >= 2) {
                high_reuse_count++;
            }
        }
        size_t capped = std::min(high_reuse_count, static_cast<size_t>(10));
        bonus_total += weights_.register_reuse_bonus *
                       static_cast<double>(capped) * 0.2;
    }

    // Numerical safety: ensure bonus_total never pushes score positive,
    // which would invert the "less negative = better" convention.
    if (bonus_total > 0.0) {
        // Allow bonuses only up to 50% of the absolute penalty so the
        // score can't flip sign on a net-positive function.
        double max_bonus = std::abs(score) * 0.5;
        if (bonus_total > max_bonus) {
            bonus_total = max_bonus;
        }
    }
    score += bonus_total;

    // The score is the NEGATIVE of the cost — less negative = better.
    // (Convention documented at the top of this file and relied on by
    // score_candidate_with_cached_orig, which uses abs values so the
    // ratio is robust to sign, and by StochasticSearch::accept which
    // treats "higher score = better".)
    result.score = -score;

    // Numerical safety: never let score be NaN/Inf.
    if (!std::isfinite(result.score)) {
        result.score = 0.0;
    }

    // GPU-specific estimates
    if (result.is_gpu_kernel) {
        // Estimate register pressure from the number of simultaneously
        // live values
        size_t max_live = 0;
        for (auto& block : fn.blocks()) {
            std::unordered_map<std::string, bool> live;
            for (auto& inst : block->instructions()) {
                if (inst->has_name()) {
                    live[inst->name()] = true;
                }
            }
            max_live = std::max(max_live, live.size());
        }
        result.estimated_register_pressure = max_live;

        // Estimate warp divergence from conditional branches
        size_t cond_branches = 0;
        for (auto& block : fn.blocks()) {
            for (auto& inst : block->instructions()) {
                if (inst->opcode() == ir::Opcode::Br && inst->num_operands() > 0) {
                    cond_branches++;
                }
            }
        }
        result.estimated_warp_divergence =
            std::min(static_cast<double>(cond_branches) * 0.15, 1.0);
    }

    // Insert into cache (shared, const FunctionAnalysis).
    cache_.put(hash, std::make_shared<const FunctionAnalysis>(result));
    return result;
}

std::map<std::string, FunctionAnalysis>
EvaluationEngine::analyse_module(const ir::Module& mod) const {
    std::map<std::string, FunctionAnalysis> results;
    for (auto& fn : mod.functions()) {
        if (fn) {
            results[fn->name()] = analyse(*fn);
        }
    }
    return results;
}

double EvaluationEngine::score_candidate(
    const ir::Function& original,
    const ir::Function& candidate) const {
    auto orig_analysis = analyse(original);
    return score_candidate_with_cached_orig(orig_analysis, candidate);
}

double EvaluationEngine::score_candidate_with_cached_orig(
    const FunctionAnalysis& orig_analysis,
    const ir::Function& candidate) const {
    auto cand_analysis = analyse(candidate);

    double orig_score = safe_finite(orig_analysis.score);
    double cand_score = safe_finite(cand_analysis.score);

    // Both scores are negative (less negative = better) under the
    // convention documented at the top of this file. We use abs values
    // so that the ratio is robust to the rare positive-score edge case
    // (which is itself prevented by the bonus clamp in analyse()).
    //
    // ratio = orig_abs / cand_abs
    //   |cand| < |orig|  →  ratio > 1   → candidate is BETTER ✓
    //   |cand| > |orig|  →  ratio < 1   → candidate is WORSE  ✓
    //   |cand| == |orig| →  ratio == 1  → equivalent          ✓
    //
    double orig_abs = std::abs(orig_score);
    double cand_abs = std::abs(cand_score);

    // Guard BOTH sides against near-zero values.
    if (cand_abs < 1e-9) {
        // Candidate has zero observable cost.
        // If orig is also ~0, ratio is 1.0 (equivalent).
        // Otherwise the candidate is unboundedly better.
        return (orig_abs < 1e-9) ? 1.0
                                 : std::numeric_limits<double>::max() / 2.0;
    }
    if (orig_abs < 1e-9) {
        // Original has zero cost; any non-zero candidate is a regression.
        return (cand_abs < 1e-9) ? 1.0
                                 : std::numeric_limits<double>::min() * 2.0;
    }
    double ratio = orig_abs / cand_abs;
    // Final safety: never let the ratio be NaN/Inf — clamp into a
    // finite positive range.
    if (!std::isfinite(ratio) || ratio < 0.0) {
        ratio = 1.0;
    }
    assert(ratio >= 0.0 && "improvement ratio must be non-negative");
    return ratio;
}

double EvaluationEngine::memory_penalty(MemoryLayer layer) const {
    switch (layer) {
        case MemoryLayer::Register:   return weights_.register_penalty;
        case MemoryLayer::L1:         return weights_.l1_penalty;
        case MemoryLayer::L2:         return weights_.l2_penalty;
        case MemoryLayer::L3:         return weights_.l3_penalty;
        case MemoryLayer::DRAM:       return weights_.dram_penalty;
        case MemoryLayer::GPU_Global: return weights_.gpu_global_penalty;
        case MemoryLayer::GPU_Shared: return weights_.gpu_shared_penalty;
        case MemoryLayer::GPU_Local:  return weights_.gpu_local_penalty;
        case MemoryLayer::NVMe:       return weights_.nvme_penalty;
        case MemoryLayer::Network:    return weights_.network_penalty;
    }
    return 0.0;
}

// ── Private: Critical-path analysis ─────────────────────────────────────────
//
// Builds a per-basic-block def-use DAG and computes the longest
// weighted dependency chain (in cycles), using the CostModel. The
// function-level critical path is the max over blocks.
//
// Edges: an instruction `I` depends on a prior instruction `D` in the
// same block if `D` defines a value that `I` uses as an operand. We
// resolve operands by name (this IR is register-by-name; constants and
// cross-block values have no in-block def and contribute 0 latency).
//
// Latency of an edge = D's latency_cycles (the cost the consumer pays
// for waiting on D's result). Critical path through a block =
// max over instructions of (path_length[I]).

double EvaluationEngine::compute_critical_path(
    const ir::Function& fn,
    const std::vector<ir::NaturalLoop>& loop_info) const {
    if (!cost_model_) return 0.0;
    double max_path = 0.0;

    for (auto& block : fn.blocks()) {
        // §2.1: weight this block's latency contributions by its estimated
        // execution frequency. A dependency chain inside a loop body is
        // walked ~freq times per call, so its contribution to the *total*
        // dynamic critical path scales with freq — exactly like the
        // per-instruction cost accumulation in analyse() (Phase 8).
        // Without this, a loop body's critical path counted for exactly
        // one iteration no matter how many times it actually ran, which
        // made the metric blind to "this dependency chain is the hot
        // path" vs "this dependency chain runs once on a cold branch."
        const double freq = block_loop_frequency(block->name(), loop_info);

        // Map from value name → (def-index, weighted-latency-of-def).
        std::unordered_map<std::string, std::pair<size_t, double>> defs;
        // Longest path ending at instruction i (in cycles).
        std::vector<double> path_len(block->instructions().size(), 0.0);

        for (size_t i = 0; i < block->instructions().size(); ++i) {
            const auto& inst = block->instructions()[i];
            double max_pred_path = 0.0;

            for (auto& op : inst->operands()) {
                if (!op || !op->has_name()) continue;
                auto it = defs.find(op->name());
                if (it == defs.end()) continue; // constant / arg / cross-block
                const auto& [def_idx, def_latency] = it->second;
                double candidate = path_len[def_idx] + def_latency;
                if (candidate > max_pred_path) {
                    max_pred_path = candidate;
                }
            }

            // A bitcast is a pure reinterpretation, never lowers to a
            // machine instruction, and is charged nothing in the Phase-8
            // cost accumulation either — keep the two consistent so the
            // search can't see a phantom critical-path cost from a
            // bitcast that has no runtime cost at all.
            double inst_latency = 0.0;
            if (inst->opcode() != ir::Opcode::BitCast) {
                OpCost oc;
                if (!vector_intrinsic_cost(*inst, *cost_model_, oc)) {
                    oc = cost_model_->cost(inst->opcode());
                }
                inst_latency = oc.latency_cycles * freq;
            }
            path_len[i] = max_pred_path + inst_latency;
            if (path_len[i] > max_path) {
                max_path = path_len[i];
            }

            // Record this def for use by later instructions.
            if (inst->has_name()) {
                defs[inst->name()] = {i, inst_latency};
            }
        }
    }

    return max_path;
}

double EvaluationEngine::compute_max_block_frequency(
    const ir::Function& fn,
    const std::vector<ir::NaturalLoop>& loop_info) const {
    double max_freq = 1.0;
    for (auto& block : fn.blocks()) {
        const double freq = block_loop_frequency(block->name(), loop_info);
        if (freq > max_freq) max_freq = freq;
    }
    return max_freq;
}

// ── Private: Instruction-level analysis ────────────────────────────────────

void EvaluationEngine::analyse_instructions(
    const ir::Function& fn, FunctionAnalysis& result) const {
    result.total_instructions = 0;
    result.memory_instructions = 0;
    result.branch_instructions = 0;
    result.compute_instructions = 0;
    result.call_instructions = 0;

    for (auto& block : fn.blocks()) {
        for (auto& inst : block->instructions()) {
            ++result.total_instructions;

            auto op = inst->opcode();

            // Memory instructions
            if (op == ir::Opcode::Alloca || op == ir::Opcode::Load ||
                op == ir::Opcode::Store || op == ir::Opcode::GetElementPtr ||
                op == ir::Opcode::Fence) {
                ++result.memory_instructions;
            }
            // Branch instructions
            else if (op == ir::Opcode::Br || op == ir::Opcode::SwitchInst ||
                     op == ir::Opcode::Invoke || op == ir::Opcode::Resume) {
                ++result.branch_instructions;
            }
            // Call instructions
            else if (op == ir::Opcode::Call) {
                ++result.call_instructions;
            }
            // Compute instructions (binary ops, comparisons, conversions)
            else if (inst->is_binary_op() || inst->is_cmp() || inst->is_cast() ||
                     op == ir::Opcode::Select || op == ir::Opcode::Phi ||
                     op == ir::Opcode::ExtractValue ||
                     op == ir::Opcode::InsertValue ||
                     op == ir::Opcode::ExtractElement ||
                     op == ir::Opcode::InsertElement ||
                     op == ir::Opcode::ShuffleVector) {
                ++result.compute_instructions;
            }
            // Ret and Unreachable are terminators but not branches for counting
        }
    }
}

// ── Private: Memory access pattern analysis ────────────────────────────────

void EvaluationEngine::analyse_memory_access(
    const ir::Function& fn, FunctionAnalysis& result) const {
    // Estimate which memory layer each access targets based on heuristics:
    //
    // - alloca: stack allocation → L1 (fast, small)
    // - load from alloca result: L1
    // - load from argument pointer (noalias): L1-L2
    // - load from GEP of argument: L2-L3
    // - load from global: DRAM
    // - store: similar to load
    // - GPU kernel + shared memory: GPU_Shared
    // - GPU kernel + global memory: GPU_Global

    // Collect alloca names to identify stack accesses
    std::unordered_map<std::string, MemoryLayer> value_layers;

    for (auto& block : fn.blocks()) {
        for (auto& inst : block->instructions()) {
            auto op = inst->opcode();

            if (op == ir::Opcode::Alloca) {
                // Stack allocations live in L1
                if (inst->has_name()) {
                    value_layers[inst->name()] = MemoryLayer::L1;
                }
                // alloca is an allocation, not an access — don't
                // double-count it in memory_accesses.
            }
        }
    }

    // Now analyse load/store
    for (auto& block : fn.blocks()) {
        for (auto& inst : block->instructions()) {
            auto op = inst->opcode();

            if (op == ir::Opcode::Load) {
                MemoryLayer layer = MemoryLayer::DRAM; // default assumption

                if (inst->num_operands() > 0) {
                    auto ptr = inst->operand(0);
                    if (ptr && ptr->has_name()) {
                        auto it = value_layers.find(ptr->name());
                        if (it != value_layers.end()) {
                            // Loading from a stack allocation → same layer
                            layer = it->second;
                        } else if (ptr->name()[0] == '@') {
                            // Global variable → DRAM
                            layer = MemoryLayer::DRAM;
                        } else {
                            // Function argument or computed pointer → L2/L3
                            layer = MemoryLayer::L2;
                        }
                    }
                    // Prefer PointerType::address_space() over
                    // name-based detection when available.
                    if (ptr && ptr->type() && ptr->type()->is_pointer()) {
                        auto& pty =
                            static_cast<const ir::PointerType&>(*ptr->type());
                        unsigned as = pty.address_space();
                        if (as == 1)      layer = MemoryLayer::GPU_Global;
                        else if (as == 3) layer = MemoryLayer::GPU_Shared;
                        else if (as == 5) layer = MemoryLayer::GPU_Local;
                    }
                }

                if (result.is_gpu_kernel) {
                    // For GPU kernels, default loads are from global memory
                    if (layer == MemoryLayer::DRAM) {
                        layer = MemoryLayer::GPU_Global;
                    }
                }

                result.memory_accesses[layer]++;
            }

            if (op == ir::Opcode::Store) {
                MemoryLayer layer = MemoryLayer::DRAM; // default assumption

                if (inst->num_operands() > 1) {
                    auto ptr = inst->operand(1);
                    if (ptr && ptr->has_name()) {
                        auto it = value_layers.find(ptr->name());
                        if (it != value_layers.end()) {
                            layer = it->second;
                        } else if (ptr->name()[0] == '@') {
                            layer = MemoryLayer::DRAM;
                        } else {
                            layer = MemoryLayer::L2;
                        }
                    }
                    if (ptr && ptr->type() && ptr->type()->is_pointer()) {
                        auto& pty =
                            static_cast<const ir::PointerType&>(*ptr->type());
                        unsigned as = pty.address_space();
                        if (as == 1)      layer = MemoryLayer::GPU_Global;
                        else if (as == 3) layer = MemoryLayer::GPU_Shared;
                        else if (as == 5) layer = MemoryLayer::GPU_Local;
                    }
                }

                if (result.is_gpu_kernel) {
                    if (layer == MemoryLayer::DRAM) {
                        layer = MemoryLayer::GPU_Global;
                    }
                }

                result.memory_accesses[layer]++;
            }

            if (op == ir::Opcode::GetElementPtr) {
                // GEP itself doesn't access memory, but it computes
                // addresses that will be used by load/store. We still
                // count it as a Register operation since it's purely
                // address computation.
                result.memory_accesses[MemoryLayer::Register]++;
            }
        }
    }

    // Arguments (incoming values) — they're in registers or L1
    for (size_t i = 0; i < fn.argument_count(); ++i) {
        result.memory_accesses[MemoryLayer::Register]++;
    }
}

// ── Private: Control flow analysis ─────────────────────────────────────────

void EvaluationEngine::analyse_control_flow(
    const ir::Function& fn, FunctionAnalysis& result) const {
    result.estimated_loop_count = 0;
    result.has_nested_loops = false;

    // Simple loop detection: a basic block that branches back to a dominator
    // or to a block with a smaller index is likely a loop back-edge.
    //
    // We use a simple heuristic: if a block's successors include a block that
    // appears earlier in the function's block list, it's a loop.

    std::unordered_map<std::string, size_t> block_order;
    for (size_t i = 0; i < fn.blocks().size(); ++i) {
        block_order[fn.blocks()[i]->name()] = i;
    }

    size_t loop_count = 0;
    bool has_nested = false;
    std::unordered_map<std::string, int> loop_depth;

    for (size_t i = 0; i < fn.blocks().size(); ++i) {
        auto& block = fn.blocks()[i];
        auto succs = block->successors();

        for (auto& succ : succs) {
            auto it = block_order.find(succ);
            if (it != block_order.end() && it->second <= i) {
                // Back-edge detected → loop
                loop_count++;
                loop_depth[succ]++;

                // Nested loop: if the target block already has a loop depth > 0
                if (loop_depth[succ] > 1) {
                    has_nested = true;
                }
            }
        }
    }

    result.estimated_loop_count = loop_count;
    result.has_nested_loops = has_nested;
}

// ── Private: Task weight estimation ────────────────────────────────────────

void EvaluationEngine::estimate_task_weight(FunctionAnalysis& result) const {
    // Task weight is a 0..1 value representing computational heaviness.
    // Factors:
    //   - More compute instructions → heavier
    //   - More memory instructions → heavier (but less than compute)
    //   - More call instructions → heavier
    //   - Loops → much heavier
    //   - GPU kernels → heavier by default

    if (result.total_instructions == 0) {
        result.task_weight = 0.0;
        return;
    }

    double weight = 0.0;

    // Base weight from instruction mix. Normalised coefficients
    // sum to 1.0 (0.4 + 0.3 + 0.3).
    double compute_ratio =
        static_cast<double>(result.compute_instructions) /
        static_cast<double>(result.total_instructions);
    double memory_ratio =
        static_cast<double>(result.memory_instructions) /
        static_cast<double>(result.total_instructions);
    double call_ratio =
        static_cast<double>(result.call_instructions) /
        static_cast<double>(result.total_instructions);

    // Coefficients now sum to 1.0: 0.4 + 0.3 + 0.3 = 1.0
    weight += compute_ratio * 0.4;
    weight += memory_ratio * 0.3;
    weight += call_ratio * 0.3;

    // Loop multiplier (§2.1): derived from `max_block_frequency`, the same per-block
    // execution-frequency estimate used for critical-path and
    // per-instruction cost weighting below, log-scaled so a doubly-nested
    // loop (freq 64) doesn't saturate the weight outright but still reads
    // meaningfully heavier than a single shallow loop (freq 8).
    double loop_multiplier =
        1.0 + std::log2(std::max(1.0, result.max_block_frequency)) * 0.15;
    weight *= loop_multiplier;

    // GPU kernels start with higher base weight
    if (result.is_gpu_kernel) {
        weight *= 1.2;
    }

    // Clamp to [0.0, 1.0]
    result.task_weight =
        std::max(0.0, std::min(1.0, weight)) * weights_.task_weight_multiplier;
    // Numerical safety.
    if (!std::isfinite(result.task_weight)) {
        result.task_weight = 0.0;
    }
}

// ── Private: Core placement decision ───────────────────────────────────────

void EvaluationEngine::decide_core_placement(FunctionAnalysis& result) const {
    // P-core: heavy compute, memory-bound, latency-sensitive
    // E-core: lightweight, compute-light, throughput-oriented
    // Any: moderate workloads

    if (result.is_gpu_kernel) {
        // GPU kernels don't run on CPU cores
        result.core_preference = FunctionAnalysis::CorePreference::Any;
        return;
    }

    double task_weight = result.task_weight;

    // Count how much memory work involves slow layers (L3, DRAM+)
    size_t slow_memory_accesses = 0;
    for (auto& [layer, count] : result.memory_accesses) {
        if (layer == MemoryLayer::L3 || layer == MemoryLayer::DRAM ||
            layer == MemoryLayer::NVMe || layer == MemoryLayer::Network) {
            slow_memory_accesses += count;
        }
    }

    // Heavy task weight → P-core (needs high single-thread performance)
    if (task_weight >= 0.6) {
        result.core_preference = FunctionAnalysis::CorePreference::P_Core;
    }
    // Light task weight with few slow memory accesses → E-core
    else if (task_weight <= 0.25 && slow_memory_accesses <= 2) {
        result.core_preference = FunctionAnalysis::CorePreference::E_Core;
    }
    // Moderate → Any
    else {
        result.core_preference = FunctionAnalysis::CorePreference::Any;
    }

    // Override: if there are many slow memory accesses, prefer P-core
    // (E-cores have smaller caches and suffer more from cache misses)
    if (slow_memory_accesses >= 5 &&
        result.core_preference == FunctionAnalysis::CorePreference::E_Core) {
        result.core_preference = FunctionAnalysis::CorePreference::Any;
    }
}

// ── Private: Identify hot values ───────────────────────────────────────────

void EvaluationEngine::identify_hot_values(
    const ir::Function& fn, FunctionAnalysis& result) const {
    // Count how many times each value is used as an operand
    auto use_count = compute_use_counts(fn);

    // Sort by use count and collect the "hot" values (used ≥ 2 times)
    std::vector<std::pair<size_t, std::string>> sorted;
    sorted.reserve(use_count.size());
    for (auto& [name, count] : use_count) {
        sorted.push_back({count, name});
    }
    std::sort(sorted.begin(), sorted.end(), std::greater<>());

    for (auto& [count, name] : sorted) {
        if (count >= 2) {
            result.hot_values.push_back(name);
        }
        // Cap at a reasonable number
        if (result.hot_values.size() >= 20) break;
    }
}

// ── Private: Use-count helper (shared by identify_hot_values and the ────────
// ── register-reuse bonus block inside analyse()) ────────────────────────────

std::unordered_map<std::string, size_t>
EvaluationEngine::compute_use_counts(const ir::Function& fn) const {
    std::unordered_map<std::string, size_t> use_count;
    for (auto& block : fn.blocks()) {
        for (auto& inst : block->instructions()) {
            for (auto& op : inst->operands()) {
                if (op && op->has_name()) {
                    use_count[op->name()]++;
                }
            }
        }
    }
    return use_count;
}

} // namespace clunk::evaluator