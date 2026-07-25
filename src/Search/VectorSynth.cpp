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
 * Clunk Vector-Intrinsic Synthesis — implementation.
 * See include/clunk/Search/VectorSynth.h for the contract.
 */
#include "clunk/Search/VectorSynth.h"

#include <algorithm>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "clunk/IR/Clone.h"
#include "clunk/IR/Instruction.h"
#include "clunk/IR/Scalarizer.h"
#include "clunk/IR/Value.h"

namespace clunk::search {

namespace {

bool is_assoc_comm(ir::Opcode op) {
    switch (op) {
    case ir::Opcode::Add: case ir::Opcode::Mul:
    case ir::Opcode::And: case ir::Opcode::Or: case ir::Opcode::Xor:
        return true;
    default:
        return false;
    }
}

const char* reduce_op_name(ir::Opcode op) {
    switch (op) {
    case ir::Opcode::Add: return "add";
    case ir::Opcode::Mul: return "mul";
    case ir::Opcode::And: return "and";
    case ir::Opcode::Or:  return "or";
    case ir::Opcode::Xor: return "xor";
    default:              return nullptr;
    }
}

// Integer binops that are valid lane-wise on integer vectors.
bool is_int_binop(ir::Opcode op) {
    switch (op) {
    case ir::Opcode::Add: case ir::Opcode::Sub: case ir::Opcode::Mul:
    case ir::Opcode::UDiv: case ir::Opcode::SDiv:
    case ir::Opcode::URem: case ir::Opcode::SRem:
    case ir::Opcode::And: case ir::Opcode::Or: case ir::Opcode::Xor:
    case ir::Opcode::Shl: case ir::Opcode::LShr: case ir::Opcode::AShr:
        return true;
    default:
        return false;
    }
}

int64_t const_int_or(const std::shared_ptr<ir::Value>& v, int64_t fallback) {
    auto* ci = dynamic_cast<const ir::ConstantInt*>(v.get());
    return ci ? ci->value() : fallback;
}

// Count operand uses of every named value across the whole function
// (including ret / branch conditions).
std::unordered_map<std::string, size_t> use_counts(const ir::Function& fn) {
    std::unordered_map<std::string, size_t> uses;
    for (auto& block : fn.blocks()) {
        for (auto& inst : block->instructions()) {
            if (!inst) continue;
            for (auto& op : inst->operands()) {
                if (op && op->has_name()) ++uses[op->name()];
            }
        }
    }
    return uses;
}

// All value names in the function (for fresh-name generation).
std::unordered_set<std::string> all_names(const ir::Function& fn) {
    std::unordered_set<std::string> names;
    for (auto& arg : fn.arguments()) {
        if (!arg.name.empty()) names.insert(arg.name);
    }
    for (auto& block : fn.blocks()) {
        for (auto& inst : block->instructions()) {
            if (inst && inst->has_name()) names.insert(inst->name());
        }
    }
    return names;
}

std::string fresh_name(std::unordered_set<std::string>& names,
                       const std::string& base) {
    for (size_t i = 0;; ++i) {
        std::string n = base + std::to_string(i);
        if (names.insert(n).second) return n;
    }
}

// Is `inst` an extractelement of a NAMED integer vector with a constant
// in-range lane index? Fills the out-params on success.
bool match_lane_extract(const ir::Instruction& inst,
                        std::string& vec_name,
                        std::shared_ptr<ir::Value>& vec_val,
                        uint64_t& lane, uint64_t& lane_count) {
    if (inst.opcode() != ir::Opcode::ExtractElement) return false;
    if (inst.num_operands() < 2) return false;
    auto vec = inst.operand(0);
    if (!vec || !vec->has_name() || !vec->type() || !vec->type()->is_vector()) {
        return false;
    }
    auto& vt = static_cast<const ir::VectorType&>(*vec->type());
    if (!vt.element_type()->is_integer()) return false;
    int64_t idx = const_int_or(inst.operand(1), -1);
    if (idx < 0 || static_cast<uint64_t>(idx) >= vt.count()) return false;
    vec_name = vec->name();
    vec_val = vec;
    lane = static_cast<uint64_t>(idx);
    lane_count = vt.count();
    return true;
}

// Remove trivially-dead pure instructions (no named uses). Iterates to a
// fixpoint so a fused group's extract chains fully disappear.
bool run_dce(ir::Function& fn) {
    bool any = false;
    for (;;) {
        auto uses = use_counts(fn);
        bool changed = false;
        for (auto& block : fn.blocks()) {
            auto& instrs = block->instructions();
            for (size_t i = instrs.size(); i-- > 0;) {
                auto& inst = instrs[i];
                if (!inst || inst->is_terminator() || inst->is_memory_op()) continue;
                if (inst->is_volatile()) continue;
                if (!inst->has_name() || uses.count(inst->name())) continue;
                // Calls: only the pure clunk.vector.* intrinsics are safe
                // to delete.
                if (inst->opcode() == ir::Opcode::Call) {
                    auto it = inst->metadata().find("callee");
                    if (it == inst->metadata().end() ||
                        it->second.rfind("clunk.vector.", 0) != 0) {
                        continue;
                    }
                }
                instrs.erase(instrs.begin() + static_cast<std::ptrdiff_t>(i));
                changed = true;
            }
        }
        if (!changed) break;
        any = true;
    }
    return any;
}

} // anonymous namespace

VectorSynthesizer::VectorSynthesizer(evaluator::EvaluationEngine* engine,
                                     const VectorSynthConfig& config)
    : engine_(engine), config_(config) {}

// ── Rewrite 1: lane-wise fusion ─────────────────────────────────────────────

bool VectorSynthesizer::apply_lane_fusion(ir::Function& fn) {
    bool changed = false;
    auto names = all_names(fn);

    for (auto& block : fn.blocks()) {
        auto& instrs = block->instructions();

        // Where each named value is defined in this block.
        std::unordered_map<std::string, size_t> def_index;
        for (size_t i = 0; i < instrs.size(); ++i) {
            if (instrs[i] && instrs[i]->has_name()) def_index[instrs[i]->name()] = i;
        }

        // Group candidate scalar binops by (opcode, flags, vecA, vecB).
        struct Member { size_t inst_index; uint64_t lane; };
        struct Group {
            std::shared_ptr<ir::Value> vec_a, vec_b;
            uint64_t lane_count = 0;
            std::vector<Member> members;
        };
        std::map<std::string, Group> groups;

        for (size_t i = 0; i < instrs.size(); ++i) {
            auto& inst = instrs[i];
            if (!inst || !inst->has_name()) continue;
            if (!is_int_binop(inst->opcode())) continue;
            if (!inst->type() || !inst->type()->is_integer()) continue;
            if (inst->num_operands() < 2) continue;

            auto resolve = [&](const std::shared_ptr<ir::Value>& v)
                -> const ir::Instruction* {
                if (!v || !v->has_name()) return nullptr;
                auto it = def_index.find(v->name());
                if (it == def_index.end()) return nullptr;
                return instrs[it->second].get();
            };
            const auto* ea = resolve(inst->operand(0));
            const auto* eb = resolve(inst->operand(1));
            if (!ea || !eb) continue;

            std::string an, bn;
            std::shared_ptr<ir::Value> av, bv;
            uint64_t la = 0, lb = 0, na = 0, nb = 0;
            if (!match_lane_extract(*ea, an, av, la, na) ||
                !match_lane_extract(*eb, bn, bv, lb, nb)) {
                continue;
            }
            if (la != lb || na != nb || na < 2 || na > config_.max_lanes) continue;
            if (!(*av->type() == *bv->type())) continue;

            std::string key = std::string(ir::Instruction::opcode_name(inst->opcode())) +
                              inst->binop_flags().to_string() + "|" + an + "|" + bn;
            auto& g = groups[key];
            if (g.members.empty()) {
                g.vec_a = av;
                g.vec_b = bv;
                g.lane_count = na;
            }
            g.members.push_back({i, la});
        }

        // Rewrite every group that covers each lane exactly once.
        // Collected first, applied in one pass (indices stay valid because
        // rewrites replace in place, never insert/erase — the new vector op
        // replaces the group's first member's slot... it must precede all
        // members, so insert there and shift bookkeeping is avoided by
        // replacing the FIRST member with the vector op and each remaining
        // member with its extract.
        for (auto& [key, g] : groups) {
            (void)key;
            if (g.members.size() != g.lane_count) continue;
            std::vector<bool> seen(g.lane_count, false);
            bool exact = true;
            for (auto& m : g.members) {
                if (seen[m.lane]) { exact = false; break; }
                seen[m.lane] = true;
            }
            if (!exact) continue;

            // First member (lowest instruction index) hosts the vector op.
            size_t first = g.members[0].inst_index;
            for (auto& m : g.members) first = std::min(first, m.inst_index);

            const auto& proto = instrs[g.members[0].inst_index];
            auto vec_op = ir::inst::make_binop(
                proto->opcode(), g.vec_a, g.vec_b,
                fresh_name(names, "vsynth"), proto->binop_flags());

            auto idx_ty = type_ctx_.int32();
            for (auto& m : g.members) {
                auto& scalar = instrs[m.inst_index];
                auto ext = ir::inst::make_extractelement(
                    vec_op,
                    std::make_shared<ir::ConstantInt>(idx_ty,
                                                      static_cast<int64_t>(m.lane)),
                    scalar->name());
                if (m.inst_index == first) {
                    // The vector op takes the first slot; the extract that
                    // used to live here is re-inserted right after it.
                    block->replace_instruction(first, vec_op);
                    block->insert_instruction(first + 1, ext);
                    // Shift every later member index by one.
                    for (auto& m2 : g.members) {
                        if (m2.inst_index > first) ++m2.inst_index;
                    }
                    // def_index is stale from here on, but all remaining
                    // group members were located before any rewriting and
                    // their (shifted) indices are still exact.
                } else {
                    block->replace_instruction(m.inst_index, ext);
                }
            }
            ++stats_.lane_fusions;
            changed = true;
            // def_index positions after `first` are stale now; rebuilding
            // per group keeps the remaining groups' member indices honest.
            // (Groups collected earlier hold indices from before this
            // insertion — one insertion shifts them all by one.)
            for (auto& [k2, g2] : groups) {
                (void)k2;
                if (&g2 == &g) continue;
                for (auto& m2 : g2.members) {
                    if (m2.inst_index >= first) ++m2.inst_index;
                }
            }
        }
    }
    return changed;
}

// ── Rewrite 2: reduction synthesis ──────────────────────────────────────────

bool VectorSynthesizer::apply_reduction_synthesis(ir::Function& fn) {
    bool changed = false;
    auto uses = use_counts(fn);

    for (auto& block : fn.blocks()) {
        auto& instrs = block->instructions();

        std::unordered_map<std::string, size_t> def_index;
        for (size_t i = 0; i < instrs.size(); ++i) {
            if (instrs[i] && instrs[i]->has_name()) def_index[instrs[i]->name()] = i;
        }

        for (size_t root_i = 0; root_i < instrs.size(); ++root_i) {
            auto& root = instrs[root_i];
            if (!root || !root->has_name()) continue;
            const auto op = root->opcode();
            if (!is_assoc_comm(op)) continue;
            if (!root->type() || !root->type()->is_integer()) continue;
            if (root->num_operands() != 2) continue;

            // Walk the single-use same-opcode tree under `root`, collecting
            // extractelement leaves. The root itself is the first internal
            // node: expand its operands.
            std::vector<const ir::Instruction*> leaves;
            bool ok = true;
            std::vector<std::shared_ptr<ir::Value>> expand;
            expand.push_back(root->operand(0));
            expand.push_back(root->operand(1));
            while (!expand.empty() && ok) {
                auto v = expand.back();
                expand.pop_back();
                if (!v || !v->has_name()) { ok = false; break; }
                auto it = def_index.find(v->name());
                if (it == def_index.end()) { ok = false; break; }
                const auto* def = instrs[it->second].get();
                if (def->opcode() == op && def->type() &&
                    def->type()->is_integer() && def->num_operands() == 2 &&
                    uses[def->name()] == 1) {
                    expand.push_back(def->operand(0));
                    expand.push_back(def->operand(1));
                } else if (def->opcode() == ir::Opcode::ExtractElement) {
                    leaves.push_back(def);
                } else {
                    ok = false;
                }
            }
            if (!ok || leaves.size() < 2) continue;

            // Leaves must be exactly {extract %v, k : k = 0..N-1} of ONE
            // vector.
            std::string vec_name;
            std::shared_ptr<ir::Value> vec_val;
            uint64_t lane_count = 0;
            std::vector<bool> seen;
            bool exact = true;
            for (const auto* leaf : leaves) {
                std::string vn;
                std::shared_ptr<ir::Value> vv;
                uint64_t lane = 0, n = 0;
                if (!match_lane_extract(*leaf, vn, vv, lane, n)) { exact = false; break; }
                if (vec_name.empty()) {
                    vec_name = vn;
                    vec_val = vv;
                    lane_count = n;
                    if (n < 2 || n > config_.max_lanes || n != leaves.size()) {
                        exact = false;
                        break;
                    }
                    seen.assign(n, false);
                }
                if (vn != vec_name || seen[lane]) { exact = false; break; }
                seen[lane] = true;
            }
            if (!exact) continue;

            // Synthesise: %root = call iM @clunk.vector.reduce.<op>.vNiM(%v)
            const auto& vt = static_cast<const ir::VectorType&>(*vec_val->type());
            if (root->type()->bit_width() != vt.element_type()->bit_width()) {
                continue; // widened/truncated tree — not a pure reduction
            }
            std::string callee = std::string("clunk.vector.reduce.") +
                                 reduce_op_name(op) + ".v" +
                                 std::to_string(lane_count) + "i" +
                                 std::to_string(vt.element_type()->bit_width());
            auto call = ir::inst::make_call(root->type(), callee, {vec_val},
                                            root->name());
            block->replace_instruction(root_i, call);
            ++stats_.reductions;
            changed = true;

            // The internal nodes are now dead; DCE (run by synthesize())
            // removes them. Rebuild use counts so later roots in this block
            // see the post-rewrite world.
            uses = use_counts(fn);
        }
    }
    return changed;
}

// ── Rewrite 3: shuffle algebra ──────────────────────────────────────────────

bool VectorSynthesizer::apply_shuffle_folds(ir::Function& fn) {
    bool changed = false;

    for (auto& block : fn.blocks()) {
        auto& instrs = block->instructions();

        std::unordered_map<std::string, size_t> def_index;
        for (size_t i = 0; i < instrs.size(); ++i) {
            if (instrs[i] && instrs[i]->has_name()) def_index[instrs[i]->name()] = i;
        }

        auto shuffle_def = [&](const std::shared_ptr<ir::Value>& v)
            -> const ir::Instruction* {
            if (!v || !v->has_name()) return nullptr;
            auto it = def_index.find(v->name());
            if (it == def_index.end()) return nullptr;
            const auto* def = instrs[it->second].get();
            return def->opcode() == ir::Opcode::ShuffleVector ? def : nullptr;
        };
        auto mask_lanes = [](const ir::Instruction& shuf,
                             std::vector<int64_t>& out) -> bool {
            auto* cv = dynamic_cast<const ir::ConstantVector*>(shuf.operand(2).get());
            if (!cv) return false;
            out.clear();
            for (auto& e : cv->elements()) {
                auto* ci = dynamic_cast<const ir::ConstantInt*>(e.get());
                if (!ci || ci->value() < 0) return false; // undef lane
                out.push_back(ci->value());
            }
            return !out.empty();
        };

        for (size_t i = 0; i < instrs.size(); ++i) {
            auto& inst = instrs[i];
            if (!inst) continue;

            // extractelement (shufflevector %a, %b, mask), k
            //   -> extractelement %a-or-%b, mask[k]
            if (inst->opcode() == ir::Opcode::ExtractElement &&
                inst->num_operands() >= 2) {
                const auto* shuf = shuffle_def(inst->operand(0));
                if (!shuf) continue;
                std::vector<int64_t> mask;
                if (!mask_lanes(*shuf, mask)) continue;
                int64_t k = const_int_or(inst->operand(1), -1);
                if (k < 0 || static_cast<size_t>(k) >= mask.size()) continue;
                auto src_ty = shuf->operand(0)->type();
                if (!src_ty || !src_ty->is_vector()) continue;
                const auto n = static_cast<int64_t>(
                    static_cast<const ir::VectorType&>(*src_ty).count());
                auto src = mask[static_cast<size_t>(k)] < n ? shuf->operand(0)
                                                            : shuf->operand(1);
                int64_t lane = mask[static_cast<size_t>(k)] < n
                                   ? mask[static_cast<size_t>(k)]
                                   : mask[static_cast<size_t>(k)] - n;
                auto folded = ir::inst::make_extractelement(
                    src, std::make_shared<ir::ConstantInt>(type_ctx_.int32(), lane),
                    inst->name());
                block->replace_instruction(i, folded);
                ++stats_.shuffle_folds;
                changed = true;
                continue;
            }

            // shufflevector (shufflevector %a, %b, m1), %z, m2  with m2
            // referencing only its first operand -> shufflevector %a, %b, m1∘m2
            if (inst->opcode() == ir::Opcode::ShuffleVector &&
                inst->num_operands() >= 3) {
                const auto* inner = shuffle_def(inst->operand(0));
                if (!inner) continue;
                std::vector<int64_t> m1, m2;
                if (!mask_lanes(*inner, m1) || !mask_lanes(*inst, m2)) continue;
                auto inner_src_ty = inner->operand(0)->type();
                if (!inner_src_ty || !inner_src_ty->is_vector()) continue;
                const auto outer_n = static_cast<int64_t>(m1.size());
                bool lhs_only = true;
                for (int64_t m : m2) {
                    if (m >= outer_n) { lhs_only = false; break; }
                }
                if (!lhs_only) continue;
                std::vector<int64_t> composed;
                composed.reserve(m2.size());
                for (int64_t m : m2) composed.push_back(m1[static_cast<size_t>(m)]);
                const auto elem_bits = static_cast<unsigned>(32);
                auto mask_cv = ir::ConstantVector::get_int_lanes(
                    type_ctx_, composed, elem_bits);
                auto folded = ir::inst::make_shufflevector(
                    inner->operand(0), inner->operand(1), mask_cv, inst->name());
                block->replace_instruction(i, folded);
                ++stats_.shuffle_folds;
                changed = true;
            }
        }
    }
    return changed;
}

// ── Driver ──────────────────────────────────────────────────────────────────

std::shared_ptr<ir::Function> VectorSynthesizer::synthesize(
    const ir::Function& fn, bool* proven) {
    if (proven) *proven = false;
    ++stats_.functions_seen;

    // Every synthesised form consumes vector values, so a vector-free
    // function has nothing to offer this pass.
    if (!ir::function_has_vector_ops(fn)) return nullptr;

    auto work = ir::deep_copy_function(fn);
    bool changed = false;

    // Fusion feeds reduction (a fused vector op's extracts become reduce
    // leaves), and shuffle folds can expose both — iterate to a small
    // fixpoint.
    for (int pass = 0; pass < 4; ++pass) {
        bool any = false;
        if (config_.enable_shuffle_fusion)       any |= apply_shuffle_folds(*work);
        if (config_.enable_lane_fusion)          any |= apply_lane_fusion(*work);
        if (config_.enable_reduction_synthesis)  any |= apply_reduction_synthesis(*work);
        if (!any) break;
        run_dce(*work);
        changed = true;
    }
    if (!changed) return nullptr;
    if (!ir::validate_function(*work)) return nullptr;

    // Cost gate: strictly cheaper or bust.
    if (engine_) {
        const double ratio = engine_->score_candidate(fn, *work);
        if (!(ratio > 1.0)) {
            ++stats_.rejected_by_score;
            return nullptr;
        }
    }

    // SMT gate.
    if (SMTVerifier::is_z3_available()) {
        SMTConfig scfg;
        scfg.timeout_ms = config_.smt_timeout_ms;
        SMTVerifier verifier(scfg);
        auto res = verifier.verify(fn, *work);
        if (res.status == VerificationResult::Equivalent) {
            if (proven) *proven = true;
            ++stats_.proven;
            return work;
        }
        ++stats_.rejected_by_smt;
        return config_.trust_unverified &&
                       res.status != VerificationResult::NotEquivalent
                   ? work
                   : nullptr;
    }
    return config_.trust_unverified ? work : nullptr;
}

} // namespace clunk::search
