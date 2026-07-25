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
 * Clunk LoopOpt — LICM + constant-trip full unrolling.
 * See include/clunk/Search/LoopOpt.h for the soundness contract.
 */
#include "clunk/Search/LoopOpt.h"

#include <cstdint>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "clunk/IR/Clone.h"
#include "clunk/IR/Instruction.h"
#include "clunk/IR/LoopAnalysis.h"
#include "clunk/IR/Value.h"

namespace clunk::search {

namespace {

// ── Pure, non-trapping opcodes (safe to execute speculatively) ─────────────
bool is_speculatable(const ir::Instruction& inst) {
    switch (inst.opcode()) {
    case ir::Opcode::Add: case ir::Opcode::Sub: case ir::Opcode::Mul:
    case ir::Opcode::And: case ir::Opcode::Or: case ir::Opcode::Xor:
    case ir::Opcode::Shl: case ir::Opcode::LShr: case ir::Opcode::AShr:
    case ir::Opcode::FAdd: case ir::Opcode::FSub: case ir::Opcode::FMul:
    case ir::Opcode::FDiv:  // IEEE default env: no traps, inf/NaN results
    case ir::Opcode::ICmp: case ir::Opcode::FCmp:
    case ir::Opcode::Select:
    case ir::Opcode::Trunc: case ir::Opcode::ZExt: case ir::Opcode::SExt:
    case ir::Opcode::BitCast:
    case ir::Opcode::GetElementPtr:  // pure address arithmetic
    case ir::Opcode::ExtractElement: case ir::Opcode::InsertElement:
    case ir::Opcode::ShuffleVector:
        return true;
    // UDiv/SDiv/URem/SRem trap on zero; Load traps on bad addresses;
    // calls/stores/phis/terminators are not pure. All stay put.
    default:
        return false;
    }
}

// ── Tiny constant evaluator for the unroll simulation ──────────────────────

uint64_t mask_to(uint64_t v, unsigned bits) {
    if (bits == 0 || bits >= 64) return v;
    return v & ((uint64_t(1) << bits) - 1);
}

int64_t sign_extend_from(uint64_t v, unsigned bits) {
    if (bits == 0 || bits >= 64) return static_cast<int64_t>(v);
    const uint64_t sign_bit = uint64_t(1) << (bits - 1);
    v = mask_to(v, bits);
    return static_cast<int64_t>((v ^ sign_bit) - sign_bit);
}

std::optional<uint64_t> eval_binop(ir::Opcode op, uint64_t a, uint64_t b,
                                   unsigned bits) {
    const uint64_t ma = mask_to(a, bits), mb = mask_to(b, bits);
    const int64_t sa = sign_extend_from(a, bits), sb = sign_extend_from(b, bits);
    switch (op) {
    case ir::Opcode::Add:  return mask_to(ma + mb, bits);
    case ir::Opcode::Sub:  return mask_to(ma - mb, bits);
    case ir::Opcode::Mul:  return mask_to(ma * mb, bits);
    case ir::Opcode::And:  return ma & mb;
    case ir::Opcode::Or:   return ma | mb;
    case ir::Opcode::Xor:  return ma ^ mb;
    case ir::Opcode::Shl:  return mb >= bits ? std::nullopt
                                             : std::optional<uint64_t>(mask_to(ma << mb, bits));
    case ir::Opcode::LShr: return mb >= bits ? std::nullopt
                                             : std::optional<uint64_t>(ma >> mb);
    case ir::Opcode::AShr: return mb >= bits ? std::nullopt
                                             : std::optional<uint64_t>(mask_to(
                                                   static_cast<uint64_t>(sa >> mb), bits));
    case ir::Opcode::UDiv: return mb == 0 ? std::nullopt
                                          : std::optional<uint64_t>(ma / mb);
    case ir::Opcode::URem: return mb == 0 ? std::nullopt
                                          : std::optional<uint64_t>(ma % mb);
    case ir::Opcode::SDiv:
        if (sb == 0 || (sa == INT64_MIN && sb == -1)) return std::nullopt;
        return mask_to(static_cast<uint64_t>(sa / sb), bits);
    case ir::Opcode::SRem:
        if (sb == 0 || (sa == INT64_MIN && sb == -1)) return std::nullopt;
        return mask_to(static_cast<uint64_t>(sa % sb), bits);
    default:
        return std::nullopt;
    }
}

std::optional<bool> eval_icmp(ir::CmpPredicate pred, uint64_t a, uint64_t b,
                              unsigned bits) {
    const uint64_t ma = mask_to(a, bits), mb = mask_to(b, bits);
    const int64_t sa = sign_extend_from(a, bits), sb = sign_extend_from(b, bits);
    switch (pred) {
    case ir::CmpPredicate::EQ:  return ma == mb;
    case ir::CmpPredicate::NE:  return ma != mb;
    case ir::CmpPredicate::UGT: return ma > mb;
    case ir::CmpPredicate::UGE: return ma >= mb;
    case ir::CmpPredicate::ULT: return ma < mb;
    case ir::CmpPredicate::ULE: return ma <= mb;
    case ir::CmpPredicate::SGT: return sa > sb;
    case ir::CmpPredicate::SGE: return sa >= sb;
    case ir::CmpPredicate::SLT: return sa < sb;
    case ir::CmpPredicate::SLE: return sa <= sb;
    default: return std::nullopt;  // FP predicates
    }
}

unsigned int_bits(const std::shared_ptr<ir::Type>& t) {
    return t && t->is_integer() ? static_cast<unsigned>(t->bit_width()) : 0;
}

// All value names defined by instructions inside the given loop blocks.
std::unordered_set<std::string> loop_defs(const ir::Function& fn,
                                          const ir::NaturalLoop& loop) {
    std::unordered_set<std::string> defs;
    for (auto& name : loop.blocks) {
        auto bb = fn.block(name);
        if (!bb) continue;
        for (auto& inst : bb->instructions()) {
            if (inst && inst->has_name()) defs.insert(inst->name());
        }
    }
    return defs;
}

std::unordered_set<std::string> all_value_names(const ir::Function& fn) {
    std::unordered_set<std::string> names;
    for (auto& arg : fn.arguments()) {
        if (!arg.name.empty()) names.insert(arg.name);
    }
    for (auto& bb : fn.blocks()) {
        for (auto& inst : bb->instructions()) {
            if (inst && inst->has_name()) names.insert(inst->name());
        }
    }
    return names;
}

// Split "a,b,c" (a phi's phi_blocks metadata) into labels.
std::vector<std::string> split_phi_blocks(const ir::Instruction& phi) {
    std::vector<std::string> out;
    auto it = phi.metadata().find("phi_blocks");
    if (it == phi.metadata().end()) return out;
    std::istringstream ss(it->second);
    std::string b;
    while (std::getline(ss, b, ',')) out.push_back(b);
    return out;
}

} // anonymous namespace

LoopOptimizer::LoopOptimizer(const LoopOptConfig& config) : config_(config) {}

// ── LICM ────────────────────────────────────────────────────────────────────

std::shared_ptr<ir::Function> LoopOptimizer::hoist_invariants(
    const ir::Function& fn) {
    auto work = ir::deep_copy_function(fn);
    auto loops = ir::find_natural_loops(*work);
    bool changed = false;

    for (auto& loop : loops) {
        if (loop.preheader.empty()) continue;
        auto pre = work->block(loop.preheader);
        if (!pre || !pre->terminator()) continue;

        auto defs = loop_defs(*work, loop);

        // Fixpoint: an instruction is invariant once none of its operands
        // are (still) defined inside the loop. Hoisting removes its name
        // from `defs`, which can unlock its users on the next sweep.
        bool progressed = true;
        while (progressed) {
            progressed = false;
            for (auto& bname : loop.blocks) {
                auto bb = work->block(bname);
                if (!bb) continue;
                auto& instrs = bb->instructions();
                for (size_t i = 0; i < instrs.size(); ++i) {
                    auto inst = instrs[i];
                    if (!inst || !inst->has_name()) continue;
                    if (!is_speculatable(*inst)) continue;
                    bool invariant = true;
                    for (auto& op : inst->operands()) {
                        if (op && op->has_name() && defs.count(op->name())) {
                            invariant = false;
                            break;
                        }
                    }
                    if (!invariant) continue;

                    // Hoist: move before the preheader's terminator, and
                    // strip poison flags (speculative execution must not
                    // introduce poison the original never computed).
                    instrs.erase(instrs.begin() + static_cast<std::ptrdiff_t>(i));
                    inst->binop_flags() = ir::BinOpFlags{};
                    pre->insert_instruction(pre->size() - 1, inst);
                    defs.erase(inst->name());
                    ++stats_.instructions_hoisted;
                    changed = true;
                    progressed = true;
                    --i;
                }
            }
        }
    }

    if (!changed || !ir::validate_function(*work)) return nullptr;
    return work;
}

// ── Constant-trip full unrolling ────────────────────────────────────────────

std::shared_ptr<ir::Function> LoopOptimizer::unroll_constant_loops(
    const ir::Function& fn) {
    auto work = ir::deep_copy_function(fn);
    bool changed = false;

    // Loops are re-discovered after each successful unroll (block contents
    // change under the analysis).
    for (bool retry = true; retry;) {
        retry = false;
        auto loops = ir::find_natural_loops(*work);

        for (auto& loop : loops) {
            if (!loop.is_single_block() || loop.preheader.empty()) continue;
            auto header = work->block(loop.header);
            if (!header) continue;

            // Shape: single-block loop, conditional branch to {self, exit}.
            auto term = header->terminator();
            if (!term || term->opcode() != ir::Opcode::Br ||
                term->num_operands() != 1) {
                continue;
            }
            auto md_t = term->metadata().find("true_bb");
            auto md_f = term->metadata().find("false_bb");
            if (md_t == term->metadata().end() || md_f == term->metadata().end())
                continue;
            const std::string t_bb = md_t->second, f_bb = md_f->second;
            if ((t_bb == loop.header) == (f_bb == loop.header)) continue;
            const std::string exit_bb = t_bb == loop.header ? f_bb : t_bb;
            const bool exit_on_false = t_bb == loop.header;

            // Collect phis (must have exactly [preheader, header] incoming)
            // and body instructions.
            struct PhiInfo {
                std::shared_ptr<ir::Instruction> phi;
                std::shared_ptr<ir::Value> init;     // from the preheader
                std::shared_ptr<ir::Value> carried;  // from the latch (self)
            };
            std::vector<PhiInfo> phis;
            std::vector<std::shared_ptr<ir::Instruction>> body;
            bool ok = true;
            for (size_t i = 0; i + 1 < header->size(); ++i) {  // excl. term
                auto inst = header->instruction(i);
                if (!inst) { ok = false; break; }
                if (inst->opcode() == ir::Opcode::Phi) {
                    if (!body.empty()) { ok = false; break; }  // phis lead
                    auto blocks = split_phi_blocks(*inst);
                    if (inst->num_operands() != 2 || blocks.size() != 2) {
                        ok = false;
                        break;
                    }
                    PhiInfo pi;
                    pi.phi = inst;
                    for (size_t k = 0; k < 2; ++k) {
                        if (blocks[k] == loop.preheader) pi.init = inst->operand(k);
                        else if (blocks[k] == loop.header) pi.carried = inst->operand(k);
                    }
                    if (!pi.init || !pi.carried || !inst->has_name()) {
                        ok = false;
                        break;
                    }
                    phis.push_back(std::move(pi));
                } else {
                    if (inst->opcode() == ir::Opcode::Invoke) { ok = false; break; }
                    body.push_back(inst);
                }
            }
            if (!ok) continue;

            // ── Abstract interpretation: fold the exit condition per
            // iteration. Values derived from arguments/memory stay
            // "unknown"; the condition itself must fold every round.
            using Env = std::unordered_map<std::string, uint64_t>;
            auto resolve_const = [](const Env& env,
                                    const std::shared_ptr<ir::Value>& v)
                -> std::optional<uint64_t> {
                if (auto* ci = dynamic_cast<const ir::ConstantInt*>(v.get()))
                    return static_cast<uint64_t>(ci->value());
                if (v && v->has_name()) {
                    auto it = env.find(v->name());
                    if (it != env.end()) return it->second;
                }
                return std::nullopt;
            };

            // Keep the abstract interpretation bounded WITHOUT a mid-body
            // bail (a partial body sweep would leave a stale environment
            // and could mis-fold the exit condition — a soundness bug).
            if (body.size() * (config_.max_trip + 1) >
                config_.max_simulation_steps) {
                continue;
            }

            size_t trip = 0;
            bool trip_known = false;
            {
                Env env;
                // Iteration 0 phi values.
                for (auto& pi : phis) {
                    auto c = resolve_const(env, pi.init);
                    if (c) env[pi.phi->name()] = *c;
                }
                for (size_t k = 0; k < config_.max_trip + 1; ++k) {
                    // Body sweep.
                    for (auto& inst : body) {
                        if (!inst->has_name()) continue;
                        std::optional<uint64_t> val;
                        if (inst->is_binary_op() && inst->num_operands() == 2) {
                            auto a = resolve_const(env, inst->operand(0));
                            auto b = resolve_const(env, inst->operand(1));
                            unsigned bits = int_bits(inst->type());
                            if (a && b && bits)
                                val = eval_binop(inst->opcode(), *a, *b, bits);
                        } else if (inst->opcode() == ir::Opcode::ICmp &&
                                   inst->num_operands() == 2 &&
                                   inst->operand(0)) {
                            auto a = resolve_const(env, inst->operand(0));
                            auto b = resolve_const(env, inst->operand(1));
                            unsigned bits = int_bits(inst->operand(0)->type());
                            auto pit = inst->metadata().find("pred");
                            if (a && b && bits && pit != inst->metadata().end()) {
                                auto r = eval_icmp(static_cast<ir::CmpPredicate>(
                                                       std::stoul(pit->second)),
                                                   *a, *b, bits);
                                if (r) val = *r ? 1 : 0;
                            }
                        } else if (inst->opcode() == ir::Opcode::Select &&
                                   inst->num_operands() == 3) {
                            auto c = resolve_const(env, inst->operand(0));
                            if (c) val = resolve_const(
                                env, inst->operand(*c ? 1 : 2));
                        }
                        if (val) env[inst->name()] = *val;
                        else env.erase(inst->name());
                    }
                    // Exit condition must fold.
                    auto cond = resolve_const(env, term->operand(0));
                    if (!cond) break;
                    const bool taken_true = (*cond & 1) != 0;
                    const bool exits = exit_on_false ? !taken_true : taken_true;
                    if (exits) {
                        trip = k + 1;
                        trip_known = true;
                        break;
                    }
                    // Advance phis for the next iteration (parallel).
                    Env next_phis;
                    for (auto& pi : phis) {
                        auto c = resolve_const(env, pi.carried);
                        if (c) next_phis[pi.phi->name()] = *c;
                    }
                    for (auto& pi : phis) {
                        auto it = next_phis.find(pi.phi->name());
                        if (it != next_phis.end()) env[pi.phi->name()] = it->second;
                        else env.erase(pi.phi->name());
                    }
                }
            }
            if (!trip_known || trip == 0 || trip > config_.max_trip) continue;
            if (trip * body.size() > config_.max_unrolled_instructions) continue;

            // ── Emit the straight-line expansion ────────────────────────
            auto used_names = all_value_names(*work);
            auto unique = [&used_names](const std::string& base) {
                std::string n = base;
                while (!used_names.insert(n).second) n += "_";
                return n;
            };

            // name -> current-iteration value (exit values after the loop).
            std::unordered_map<std::string, std::shared_ptr<ir::Value>> subst;
            auto resolve_val = [&subst](const std::shared_ptr<ir::Value>& v)
                -> std::shared_ptr<ir::Value> {
                if (v && v->has_name()) {
                    auto it = subst.find(v->name());
                    if (it != subst.end()) return it->second;
                }
                return v;
            };

            std::vector<std::shared_ptr<ir::Instruction>> expanded;
            for (size_t k = 0; k < trip; ++k) {
                // Parallel phi update: next values from the PREVIOUS map.
                std::unordered_map<std::string, std::shared_ptr<ir::Value>> next;
                for (auto& pi : phis) {
                    next[pi.phi->name()] =
                        k == 0 ? pi.init : resolve_val(pi.carried);
                }
                for (auto& [n, v] : next) subst[n] = v;

                for (auto& inst : body) {
                    auto clone = std::make_shared<ir::Instruction>(
                        inst->opcode(), inst->type(),
                        inst->has_name()
                            ? unique(inst->name() + ".it" + std::to_string(k))
                            : "");
                    for (auto& op : inst->operands())
                        clone->add_operand(resolve_val(op));
                    for (auto& [mk, mv] : inst->metadata())
                        clone->set_metadata(mk, mv);
                    clone->binop_flags() = inst->binop_flags();
                    if (inst->alignment())
                        clone->set_alignment(inst->alignment().value());
                    clone->set_volatile(inst->is_volatile());
                    expanded.push_back(clone);
                    if (inst->has_name()) subst[inst->name()] = clone;
                }
            }
            expanded.push_back(ir::inst::make_br_uncond(exit_bb));

            // Loop-defined names visible after the loop resolve to their
            // final-iteration values in every OTHER block.
            for (auto& bb : work->blocks()) {
                if (bb->name() == loop.header) continue;
                for (auto& inst : bb->instructions()) {
                    if (!inst) continue;
                    for (size_t oi = 0; oi < inst->num_operands(); ++oi) {
                        auto op = inst->operand(oi);
                        if (op && op->has_name() && subst.count(op->name()))
                            inst->set_operand(oi, subst[op->name()]);
                    }
                }
            }

            header->instructions() = std::move(expanded);
            ++stats_.loops_unrolled;
            stats_.iterations_expanded += trip;
            changed = true;
            retry = true;  // re-run loop discovery on the new CFG
            break;
        }
    }

    if (!changed || !ir::validate_function(*work)) return nullptr;
    return work;
}

} // namespace clunk::search
