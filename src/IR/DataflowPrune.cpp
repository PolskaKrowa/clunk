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
 * Clunk DataflowPrune — implementation.
 * See include/clunk/IR/DataflowPrune.h for the design/scope notes.
 */
#include "clunk/IR/DataflowPrune.h"

#include <deque>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

#include "clunk/Analysis/KnownBits.h"
#include "clunk/IR/BasicBlock.h"
#include "clunk/IR/Clone.h"
#include "clunk/IR/Instruction.h"
#include "clunk/IR/Value.h"

namespace clunk::ir {

namespace {

// True iff `inst` must never be removed regardless of use-count: it has an
// externally-visible side effect, or is a terminator. Mirrors
// StochasticSearch.cpp's is_dead_instruction() root set (kept independent —
// same convention already used across the codebase's several DCE-adjacent
// passes, e.g. MemOpt's dead-store elimination has its own rule too).
bool has_side_effect_or_is_root(const Instruction& inst) {
    if (inst.is_terminator()) return true;
    switch (inst.opcode()) {
        case Opcode::Store:
        case Opcode::Call:
        case Opcode::Fence:
            return true;
        case Opcode::Load:
            return inst.is_volatile();
        case Opcode::Alloca:
            // An unnamed alloca can have no uses by construction (nothing
            // can reference it), so treat it conservatively as always live
            // rather than infer anything from that.
            return !inst.has_name();
        default:
            return false;
    }
}

// Re-resolve every named-value operand across `fn` to the CURRENT
// instruction object with that name. Needed after replacing an
// instruction in place (e.g. rewriting a Phi's operand list): other
// instructions may still hold a shared_ptr to the OLD object, which is
// still name-correct but no longer the canonical def. Same idea as
// Clone.h's deep_copy_function remap pass.
void remap_operands_by_name(Function& fn) {
    std::unordered_map<std::string, std::shared_ptr<Value>> defs;
    for (auto& block : fn.blocks()) {
        if (!block) continue;
        for (auto& inst : block->instructions()) {
            if (inst && inst->has_name()) defs[inst->name()] = inst;
        }
    }
    for (auto& block : fn.blocks()) {
        if (!block) continue;
        for (auto& inst : block->instructions()) {
            if (!inst) continue;
            for (size_t i = 0; i < inst->num_operands(); ++i) {
                auto op = inst->operand(i);
                if (!op || !op->has_name()) continue;
                auto it = defs.find(op->name());
                if (it != defs.end() && it->second != op) {
                    inst->set_operand(i, it->second);
                }
            }
        }
    }
}

// Structural key for common-subexpression elimination: identical
// opcode/flags/predicate/type/operands means "same computation". Mirrors
// StochasticSearch.cpp's cse_key used by the opportunistic
// EliminateCommonSubexpr mutation (kept as an independent copy — same
// convention already used across this codebase's several DCE-adjacent
// passes, e.g. MemOpt's dead-store elimination has its own rule too).
// `binding` resolves operands through any fold already applied earlier
// in THIS pass, so a duplicate that only became visible after an earlier
// instruction was itself deduplicated/folded is still caught.
std::string cse_key(const Instruction& inst,
                     const std::unordered_map<std::string, std::shared_ptr<Value>>& binding) {
    switch (inst.opcode()) {
        case Opcode::Add: case Opcode::Sub: case Opcode::Mul:
        case Opcode::UDiv: case Opcode::SDiv:
        case Opcode::URem: case Opcode::SRem:
        case Opcode::And: case Opcode::Or: case Opcode::Xor:
        case Opcode::Shl: case Opcode::LShr: case Opcode::AShr:
        case Opcode::ICmp: case Opcode::Select:
        case Opcode::Trunc: case Opcode::ZExt: case Opcode::SExt:
        case Opcode::PtrToInt: case Opcode::IntToPtr:
        case Opcode::BitCast:
        case Opcode::GetElementPtr:
            break;  // eligible: pure, deterministic, no memory access
        default:
            return {};  // loads/stores/calls/phis/FP (NaN quirks)/etc.
    }
    if (!inst.has_name()) return {};  // result must be nameable to reuse

    std::string key;
    key.reserve(64);
    key += std::to_string(static_cast<unsigned>(inst.opcode()));
    key += '|';
    key += inst.binop_flags().to_string();
    key += '|';
    auto& md = inst.metadata();
    auto pred_it = md.find("pred");
    if (pred_it != md.end()) key += pred_it->second;
    key += '|';
    if (inst.type()) key += inst.type()->to_string();
    for (auto& raw_op : inst.operands()) {
        key += '|';
        auto op = raw_op;
        if (op && op->has_name()) {
            auto it = binding.find(op->name());
            if (it != binding.end()) op = it->second;
        }
        if (!op) { key += "<null>"; continue; }
        if (auto ci = std::dynamic_pointer_cast<ConstantInt>(op)) {
            key += '#';
            key += std::to_string(ci->value());
            key += ':';
            key += std::to_string(ci->bit_width());
        } else if (op->has_name()) {
            key += '%';
            key += op->name();
        } else {
            return {};  // unnamed non-constant operand — not comparable
        }
    }
    return key;
}

} // namespace

// ── eliminate_dead_code ─────────────────────────────────────────────────

std::shared_ptr<Function> eliminate_dead_code(const Function& fn, PruneStats* stats) {
    auto work = deep_copy_function(fn);

    // Map each named instruction's name to the instruction object, and
    // seed the liveness worklist with every side-effecting / terminator
    // instruction.
    std::unordered_map<std::string, Instruction*> by_name;
    std::vector<Instruction*> worklist;
    std::unordered_set<Instruction*> live;
    for (auto& block : work->blocks()) {
        if (!block) continue;
        for (auto& inst : block->instructions()) {
            if (!inst) continue;
            if (inst->has_name()) by_name[inst->name()] = inst.get();
            if (has_side_effect_or_is_root(*inst) && live.insert(inst.get()).second) {
                worklist.push_back(inst.get());
            }
        }
    }

    // Def-use closure: whatever a live instruction uses is live too. This
    // is a plain graph reachability walk over operand references, so it
    // needs no CFG traversal and is correct for arbitrary loop structure
    // (a Phi's back-edge operand is just another named-value reference).
    while (!worklist.empty()) {
        Instruction* cur = worklist.back();
        worklist.pop_back();
        for (auto& op : cur->operands()) {
            if (!op || !op->has_name()) continue;
            auto it = by_name.find(op->name());
            if (it == by_name.end()) continue;  // function argument, not an instruction
            if (live.insert(it->second).second) {
                worklist.push_back(it->second);
            }
        }
    }

    size_t removed = 0;
    for (auto& block : work->blocks()) {
        if (!block) continue;
        for (size_t i = block->size(); i-- > 0;) {
            auto inst = block->instruction(i);
            if (inst && !live.count(inst.get())) {
                block->remove_instruction(i);
                ++removed;
            }
        }
    }

    if (removed == 0) return nullptr;
    work->compute_predecessors();
    if (stats) stats->instructions_removed += removed;
    return work;
}

// ── remove_unreachable_blocks ───────────────────────────────────────────

std::shared_ptr<Function> remove_unreachable_blocks(const Function& fn, PruneStats* stats) {
    if (fn.blocks().size() < 2) return nullptr;  // nothing to prune

    auto work = deep_copy_function(fn);
    auto entry = work->entry_block();
    if (!entry) return nullptr;

    std::unordered_set<std::string> reachable;
    std::deque<std::string> q;
    reachable.insert(entry->name());
    q.push_back(entry->name());
    while (!q.empty()) {
        std::string cur = q.front();
        q.pop_front();
        auto bb = work->block(cur);
        if (!bb) continue;
        for (auto& succ : bb->successors()) {
            if (work->block(succ) && reachable.insert(succ).second) {
                q.push_back(succ);
            }
        }
    }

    if (reachable.size() == work->blocks().size()) return nullptr;  // all reachable

    std::vector<std::string> to_remove;
    for (auto& block : work->blocks()) {
        if (block && !reachable.count(block->name())) to_remove.push_back(block->name());
    }
    std::unordered_set<std::string> dead(to_remove.begin(), to_remove.end());

    // Fix up Phis in SURVIVING blocks that list a now-dead block as an
    // incoming predecessor: drop that operand + its phi_blocks entry.
    for (auto& block : work->blocks()) {
        if (!block || dead.count(block->name())) continue;
        for (size_t i = 0; i < block->size(); ++i) {
            auto inst = block->instruction(i);
            if (!inst || inst->opcode() != Opcode::Phi) continue;
            auto md_it = inst->metadata().find("phi_blocks");
            if (md_it == inst->metadata().end()) continue;
            std::vector<std::string> blocks;
            {
                std::istringstream bs(md_it->second);
                std::string b;
                while (std::getline(bs, b, ',')) blocks.push_back(b);
            }
            bool any_dead = false;
            for (auto& b : blocks) {
                if (dead.count(b)) { any_dead = true; break; }
            }
            if (!any_dead) continue;

            auto new_phi = std::make_shared<Instruction>(Opcode::Phi, inst->type(), inst->name());
            std::string new_blocks;
            for (size_t k = 0; k < inst->num_operands() && k < blocks.size(); ++k) {
                if (dead.count(blocks[k])) continue;
                new_phi->add_operand(inst->operand(k));
                if (!new_blocks.empty()) new_blocks += ',';
                new_blocks += blocks[k];
            }
            new_phi->set_metadata("phi_blocks", new_blocks);
            for (auto& [k, v] : inst->metadata()) {
                if (k != "phi_blocks") new_phi->set_metadata(k, v);
            }
            block->replace_instruction(i, new_phi);
        }
    }
    remap_operands_by_name(*work);

    for (auto& name : to_remove) work->remove_block(name);
    work->compute_predecessors();

    if (stats) stats->blocks_removed += to_remove.size();
    return work;
}

// ── simplify_known_bits ─────────────────────────────────────────────────

std::shared_ptr<Function> simplify_known_bits(const Function& fn, PruneStats* stats) {
    auto env = analysis::analyse_known_bits(fn);
    if (env.empty()) return nullptr;

    // Quick pre-check: is there anything to actually do? Avoids paying for
    // a deep copy + rebuild when the analysis found nothing foldable.
    bool any_fold = false;
    for (auto& block : fn.blocks()) {
        if (!block) continue;
        for (auto& inst : block->instructions()) {
            if (!inst || !inst->has_name()) continue;
            auto it = env.find(inst->name());
            if (it != env.end() && !it->second.has_conflict() && it->second.is_fully_known() &&
                inst->type() && inst->type()->is_integer()) {
                any_fold = true;
                break;
            }
        }
        if (any_fold) break;
    }
    // Also check for a foldable conditional branch even if no value folded
    // (the condition might be an argument-derived comparison already fully
    // known from an earlier round's constant, not just this round's fold).
    bool any_branch_fold = false;
    for (auto& block : fn.blocks()) {
        auto term = block ? block->terminator() : nullptr;
        if (!term || term->opcode() != Opcode::Br || term->num_operands() == 0) continue;
        auto kb = analysis::operand_known_bits(term->operand(0), env);
        if (kb.is_fully_known() && !kb.has_conflict()) { any_branch_fold = true; break; }
    }
    if (!any_fold && !any_branch_fold) return nullptr;

    auto work = std::make_shared<Function>(fn.name(), fn.function_type(), fn.linkage());
    for (auto& arg : fn.arguments()) work->add_argument(arg.type, arg.name, arg.attrs);
    for (auto& [k, v] : fn.attributes()) work->set_attribute(k, v);

    size_t folded = 0, masks_removed = 0, cmps_folded = 0, branches = 0;

    // binding: name -> replacement Value (a fold target, or the rebuilt
    // instruction itself for anything not folded).
    std::unordered_map<std::string, std::shared_ptr<Value>> binding;
    auto resolve = [&](const std::shared_ptr<Value>& v) -> std::shared_ptr<Value> {
        if (v && v->has_name()) {
            auto it = binding.find(v->name());
            if (it != binding.end()) return it->second;
        }
        return v;
    };

    for (auto& block : fn.blocks()) {
        if (!block) continue;
        auto& new_bb = work->add_block(block->name());
        new_bb.clear_predecessors();
        for (auto& inst : block->instructions()) {
            if (!inst) continue;

            // Try a known-bits constant fold first (applies to any
            // integer-typed, non-terminator instruction — including
            // ICmp, whose i1 result is just another integer-typed value
            // here, so it needs no special case).
            if (inst->has_name() && !inst->is_terminator() && inst->type() &&
                inst->type()->is_integer()) {
                auto it = env.find(inst->name());
                if (it != env.end()) {
                    auto cst = analysis::known_bits_constant(
                        it->second, inst->type().get());
                    if (cst) {
                        auto ity = std::dynamic_pointer_cast<IntegerType>(inst->type());
                        if (!ity) ity = std::make_shared<IntegerType>(
                            static_cast<unsigned>(inst->type()->bit_width()));
                        binding[inst->name()] = std::make_shared<ConstantInt>(ity, *cst);
                        if (inst->opcode() == Opcode::ICmp) ++cmps_folded;
                        else ++folded;
                        continue;  // don't emit the instruction at all
                    }
                }
            }

            // Redundant AND-mask: `and %x, C` where C's 1-bits already
            // cover every bit %x might set — the mask changes nothing.
            if (inst->has_name() && inst->opcode() == Opcode::And && inst->num_operands() == 2) {
                auto lhs_r = resolve(inst->operand(0));
                auto rhs_r = resolve(inst->operand(1));
                auto c = std::dynamic_pointer_cast<ConstantInt>(rhs_r);
                if (c && lhs_r && lhs_r->type() && lhs_r->type()->is_integer()) {
                    analysis::KnownBits lhs_kb = analysis::operand_known_bits(lhs_r, env);
                    unsigned w = static_cast<unsigned>(lhs_r->type()->bit_width());
                    uint64_t mask = analysis::KnownBits::mask_for_width(w);
                    uint64_t cbits = static_cast<uint64_t>(c->value()) & mask;
                    // Every bit of lhs is either known-zero, or the mask
                    // keeps it (cbits bit set) — so ANDing is a no-op.
                    if (((lhs_kb.zero | cbits) & mask) == mask) {
                        binding[inst->name()] = lhs_r;
                        ++masks_removed;
                        continue;
                    }
                }
            }

            // Rebuild the instruction verbatim with resolved operands.
            auto new_inst = std::make_shared<Instruction>(inst->opcode(), inst->type(), inst->name());
            for (auto& op : inst->operands()) {
                new_inst->add_operand(resolve(op));
            }
            for (auto& [k, v] : inst->metadata()) new_inst->set_metadata(k, v);
            new_inst->binop_flags() = inst->binop_flags();
            if (inst->alignment()) new_inst->set_alignment(inst->alignment().value());
            new_inst->set_volatile(inst->is_volatile());

            // A conditional Br whose (possibly just-resolved) condition is
            // a known constant becomes unconditional — this is what feeds
            // remove_unreachable_blocks its work next iteration.
            if (inst->opcode() == Opcode::Br && inst->num_operands() >= 1) {
                auto cond_r = new_inst->operand(0);
                auto c = std::dynamic_pointer_cast<ConstantInt>(cond_r);
                const auto& md = inst->metadata();
                auto true_it = md.find("true_bb");
                auto false_it = md.find("false_bb");
                if (c && true_it != md.end() && false_it != md.end()) {
                    auto uncond = std::make_shared<Instruction>(Opcode::Br, inst->type(), "");
                    uncond->set_metadata("dest_bb", (c->value() != 0) ? true_it->second
                                                                       : false_it->second);
                    new_bb.add_instruction(uncond);
                    ++branches;
                    continue;
                }
            }

            new_bb.add_instruction(new_inst);
        }
    }

    if (folded == 0 && masks_removed == 0 && cmps_folded == 0 && branches == 0) {
        return nullptr;
    }
    remap_operands_by_name(*work);
    work->compute_predecessors();
    if (stats) {
        stats->values_folded_constant += folded;
        stats->redundant_masks_removed += masks_removed;
        stats->comparisons_folded += cmps_folded;
        stats->branches_simplified += branches;
    }
    return work;
}

// ── simplify_cse ─────────────────────────────────────────────────────────

std::shared_ptr<Function> simplify_cse(const Function& fn, PruneStats* stats) {
    // Quick pre-check over the ORIGINAL function: is there even one
    // same-block duplicate pair? Cheap scan, avoids paying for a deep
    // copy + rebuild when there's nothing to do. This ignores duplicates
    // that would only appear after operand resolution mid-pass — fine,
    // it only needs to be a sound may-fire filter; the real pass below
    // (which does resolve operands) is exact.
    {
        bool any = false;
        const std::unordered_map<std::string, std::shared_ptr<Value>> empty_binding;
        for (auto& block : fn.blocks()) {
            if (!block) continue;
            std::unordered_set<std::string> seen;
            for (auto& inst : block->instructions()) {
                if (!inst) continue;
                std::string k = cse_key(*inst, empty_binding);
                if (!k.empty() && !seen.insert(k).second) { any = true; break; }
            }
            if (any) break;
        }
        if (!any) return nullptr;
    }

    auto work = std::make_shared<Function>(fn.name(), fn.function_type(), fn.linkage());
    for (auto& arg : fn.arguments()) work->add_argument(arg.type, arg.name, arg.attrs);
    for (auto& [k, v] : fn.attributes()) work->set_attribute(k, v);

    size_t eliminated = 0;

    // binding: name -> replacement Value, for instructions dropped as
    // duplicates (resolved through by later operand lookups, same idea
    // as simplify_known_bits' `binding`/`resolve`).
    std::unordered_map<std::string, std::shared_ptr<Value>> binding;
    auto resolve = [&](const std::shared_ptr<Value>& v) -> std::shared_ptr<Value> {
        if (v && v->has_name()) {
            auto it = binding.find(v->name());
            if (it != binding.end()) return it->second;
        }
        return v;
    };

    for (auto& block : fn.blocks()) {
        if (!block) continue;
        auto& new_bb = work->add_block(block->name());
        new_bb.clear_predecessors();

        // Per-block table: structural key -> earlier canonical value.
        // Reset for every block — same-block scope only (see header).
        std::unordered_map<std::string, std::shared_ptr<Value>> table;

        for (auto& inst : block->instructions()) {
            if (!inst) continue;

            std::string key = cse_key(*inst, binding);
            if (!key.empty()) {
                auto it = table.find(key);
                if (it != table.end()) {
                    // Exact duplicate of an earlier instruction in this
                    // block: reuse its result instead of recomputing.
                    binding[inst->name()] = it->second;
                    ++eliminated;
                    continue;  // don't emit the duplicate at all
                }
            }

            // Rebuild verbatim with resolved operands.
            auto new_inst = std::make_shared<Instruction>(inst->opcode(), inst->type(), inst->name());
            for (auto& op : inst->operands()) new_inst->add_operand(resolve(op));
            for (auto& [k, v] : inst->metadata()) new_inst->set_metadata(k, v);
            new_inst->binop_flags() = inst->binop_flags();
            if (inst->alignment()) new_inst->set_alignment(inst->alignment().value());
            new_inst->set_volatile(inst->is_volatile());
            new_bb.add_instruction(new_inst);

            if (!key.empty()) table[key] = new_inst;
        }
    }

    if (eliminated == 0) return nullptr;
    remap_operands_by_name(*work);
    work->compute_predecessors();
    if (stats) stats->subexpressions_eliminated += eliminated;
    return work;
}

// ── prune_dataflow ──────────────────────────────────────────────────────

std::shared_ptr<Function> prune_dataflow(const Function& fn, PruneStats* stats) {
    std::shared_ptr<Function> current;  // nullptr until something changes
    const Function* base = &fn;
    PruneStats local;

    constexpr int kMaxIterations = 4;
    for (int iter = 0; iter < kMaxIterations; ++iter) {
        bool changed_this_iter = false;

        if (auto simplified = simplify_known_bits(*base, &local)) {
            current = simplified;
            base = current.get();
            changed_this_iter = true;
        }
        if (auto deduped = simplify_cse(*base, &local)) {
            current = deduped;
            base = current.get();
            changed_this_iter = true;
        }
        if (auto pruned_blocks = remove_unreachable_blocks(*base, &local)) {
            current = pruned_blocks;
            base = current.get();
            changed_this_iter = true;
        }
        if (auto dced = eliminate_dead_code(*base, &local)) {
            current = dced;
            base = current.get();
            changed_this_iter = true;
        }

        if (!changed_this_iter) break;
    }

    if (!current) return nullptr;
    if (stats) *stats = local;
    return current;
}

} // namespace clunk::ir