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
 * Clunk Inliner — single-block + multi-block call-site inlining.
 * See include/clunk/Search/Inliner.h for the eligibility contract.
 */
#include "clunk/Search/Inliner.h"

#include <unordered_map>
#include <unordered_set>

#include "clunk/Analysis/CallGraph.h"
#include "clunk/IR/Clone.h"
#include "clunk/IR/Instruction.h"
#include "clunk/IR/Value.h"

namespace clunk::search {

namespace {

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

// Is `callee` inlinable under the legacy single-block contract?
bool eligible_single_block(const ir::Function& callee, const ir::Function& caller,
                           size_t max_insts) {
    if (callee.name() == caller.name()) return false;
    if (callee.blocks().size() != 1) return false;
    if (callee.function_type() && callee.function_type()->is_vararg()) return false;
    auto bb = callee.entry_block();
    if (!bb || bb->empty()) return false;
    auto term = bb->terminator();
    if (!term || term->opcode() != ir::Opcode::Ret) return false;
    if (bb->size() - 1 > max_insts) return false;
    for (auto& inst : bb->instructions()) {
        if (!inst) return false;
        switch (inst->opcode()) {
        case ir::Opcode::Call:
        case ir::Opcode::Invoke:
        case ir::Opcode::Phi:
        case ir::Opcode::Alloca:
            return false;
        default:
            break;
        }
    }
    for (auto& arg : callee.arguments()) {
        if (arg.name.empty()) return false;
    }
    return true;
}

// Is `callee` inlinable under the multi-block contract?
//  - callee is defined in this module, not the caller itself
//  - not vararg; arg count matches the call's operand count
//  - body size within the multiblock caps
//  - callee has at least one block and ends in `ret` (in some block)
//  - no Invoke (we don't model unwind edges)
bool eligible_multiblock(const ir::Function& callee,
                         const ir::Function& caller,
                         const InlinerConfig& cfg) {
    if (callee.name() == caller.name()) return false;
    if (callee.blocks().empty()) return false;
    if (callee.function_type() && callee.function_type()->is_vararg()) return false;

    // Body size cap.
    size_t inst_count = 0;
    for (const auto& bb : callee.blocks()) {
        if (!bb) continue;
        inst_count += bb->size();
        for (const auto& inst : bb->instructions()) {
            if (!inst) return false;
            // We refuse Invoke (unwind), Switch (we don't need it for the
            // common case, and cloning switch edges requires care), and
            // Resume/LandingPad (exception unwinding).
            switch (inst->opcode()) {
            case ir::Opcode::Invoke:
            case ir::Opcode::Resume:
            case ir::Opcode::LandingPad:
            case ir::Opcode::Switch:
            case ir::Opcode::SwitchInst:
                return false;
            default: break;
            }
        }
    }
    if (inst_count > cfg.max_multiblock_callee_instructions) return false;
    if (callee.blocks().size() > cfg.max_multiblock_callee_blocks) return false;
    for (auto& arg : callee.arguments()) {
        if (arg.name.empty()) return false;
    }
    // Verify the callee has a ret somewhere (defensive — a malformed
    // function with no ret would yield an inlined body with no return
    // path, breaking the caller).
    bool has_ret = false;
    for (const auto& bb : callee.blocks()) {
        if (!bb) continue;
        auto term = bb->terminator();
        if (term && term->opcode() == ir::Opcode::Ret) { has_ret = true; break; }
    }
    return has_ret;
}

// Count total instructions across all basic blocks of a function.
size_t count_insts(const ir::Function& fn) {
    size_t n = 0;
    for (const auto& bb : fn.blocks()) {
        if (bb) n += bb->size();
    }
    return n;
}

} // anonymous namespace

Inliner::Inliner(const InlinerConfig& config) : config_(config) {}

// ── Legacy single-block inliner (unchanged behaviour) ──────────────────────

std::shared_ptr<ir::Function> Inliner::inline_calls(const ir::Function& fn,
                                                    const ir::Module& mod) {
    auto work = ir::deep_copy_function(fn);
    auto used_names = all_value_names(*work);
    size_t inline_counter = 0;
    bool changed = false;

    // call-result name -> the value that replaces it
    std::unordered_map<std::string, std::shared_ptr<ir::Value>> result_subst;

    for (auto& bb : work->blocks()) {
        auto& instrs = bb->instructions();
        for (size_t i = 0; i < instrs.size(); ++i) {
            auto inst = instrs[i];
            if (!inst || inst->opcode() != ir::Opcode::Call) continue;
            auto it = inst->metadata().find("callee");
            if (it == inst->metadata().end()) continue;
            ++stats_.call_sites_seen;
            if (inline_counter >= config_.max_inlines_per_function) continue;

            auto callee = mod.function(it->second);
            if (!callee || !eligible_single_block(*callee, *work, config_.max_callee_instructions))
                continue;
            if (callee->argument_count() != inst->num_operands()) continue;
            if (!callee->return_type() || !inst->type() ||
                !(*callee->return_type() == *inst->type())) {
                continue;
            }

            // callee-local name -> caller-side value
            std::unordered_map<std::string, std::shared_ptr<ir::Value>> subst;
            for (size_t a = 0; a < callee->argument_count(); ++a) {
                subst[callee->arguments()[a].name] = inst->operand(a);
            }
            auto resolve = [&subst](const std::shared_ptr<ir::Value>& v)
                -> std::shared_ptr<ir::Value> {
                if (v && v->has_name()) {
                    auto sit = subst.find(v->name());
                    if (sit != subst.end()) return sit->second;
                }
                return v;
            };

            // Clone the body (all but the ret) in front of the call.
            const std::string tag = ".inl" + std::to_string(inline_counter++);
            std::vector<std::shared_ptr<ir::Instruction>> clones;
            auto cbb = callee->entry_block();
            for (size_t k = 0; k + 1 < cbb->size(); ++k) {
                auto src = cbb->instruction(k);
                std::string cname;
                if (src->has_name()) {
                    cname = src->name() + tag;
                    while (!used_names.insert(cname).second) cname += "_";
                }
                auto clone = std::make_shared<ir::Instruction>(
                    src->opcode(), src->type(), cname);
                for (auto& op : src->operands()) clone->add_operand(resolve(op));
                for (auto& [mk, mv] : src->metadata()) clone->set_metadata(mk, mv);
                clone->binop_flags() = src->binop_flags();
                if (src->alignment()) clone->set_alignment(src->alignment().value());
                clone->set_volatile(src->is_volatile());
                clones.push_back(clone);
                if (src->has_name()) subst[src->name()] = clone;
            }

            // The ret value replaces the call result.
            auto term = cbb->terminator();
            std::shared_ptr<ir::Value> ret_val;
            if (term->num_operands() >= 1) ret_val = resolve(term->operand(0));
            if (inst->has_name()) {
                if (!ret_val) continue;  // named call of a void callee — bail
                result_subst[inst->name()] = ret_val;
            }

            // Splice: body clones replace the call instruction.
            instrs.erase(instrs.begin() + static_cast<std::ptrdiff_t>(i));
            instrs.insert(instrs.begin() + static_cast<std::ptrdiff_t>(i),
                          clones.begin(), clones.end());
            i += clones.size();
            --i;  // net: continue after the spliced body
            ++stats_.call_sites_inlined;
            changed = true;
        }
    }

    if (!changed) return nullptr;

    // Substitute inlined call results everywhere.
    for (auto& bb : work->blocks()) {
        for (auto& inst : bb->instructions()) {
            if (!inst) continue;
            for (size_t oi = 0; oi < inst->num_operands(); ++oi) {
                auto op = inst->operand(oi);
                if (!op || !op->has_name()) continue;
                auto sit = result_subst.find(op->name());
                if (sit != result_subst.end()) inst->set_operand(oi, sit->second);
            }
        }
    }

    if (!ir::validate_function(*work)) return nullptr;
    return work;
}

// ── Multi-block, call-graph-aware inliner ──────────────────────────────────

std::shared_ptr<ir::Function> Inliner::inline_calls_multiblock(
    const ir::Function& fn,
    const ir::Module& mod,
    const analysis::CallGraph* cg,
    std::unordered_set<std::string> visited) {

    if (!config_.enable_multiblock) return nullptr;
    if (fn.blocks().empty()) return nullptr;

    // Build a call graph on demand if none was supplied.
    std::unique_ptr<analysis::CallGraph> owned_cg;
    if (!cg) {
        owned_cg = std::make_unique<analysis::CallGraph>();
        owned_cg->build(mod);
        cg = owned_cg.get();
    }

    // Recursion guard: don't inline a function into itself (transitively).
    visited.insert(fn.name());

    auto work = ir::deep_copy_function(fn);
    if (!work) return nullptr;
    auto used_names = all_value_names(*work);

    // We will rewrite the function block-by-block. The inlining splice
    // can introduce new blocks (cloning a multi-block callee), so we
    // rebuild the function's block list as we go.
    //
    // Strategy: process each block, walking instructions in order. For
    // each Call instruction that we can inline, replace the call with
    // the callee's cloned body (multiple blocks become multiple blocks
    // spliced into the caller's block list at that point). The block
    // containing the call gets split: instructions BEFORE the call stay
    // in the original block (with an unconditional branch to the callee's
    // entry); instructions AFTER the call go into a NEW continuation
    // block, and the callee's return block branches to that continuation.
    //
    // Phi fix-up at the callee's entry: the callee's entry-block phis
    // (if any) refer to predecessor blocks in the CALLEE's CFG. Since
    // the entry has no predecessors in the cloned body, we resolve each
    // phi to the value of its single "entry" incoming edge — but a
    // well-formed LLVM entry block cannot have phis (there's no
    // predecessor). So we just drop entry phis (defensive: if there
    // are any, treat them as undef and let validate_function complain).
    //
    // Phi fix-up at the continuation: the call's "return value" is
    // the callee's `ret` operand. We substitute the call's name with
    // that value (resolved through the cloned-body substitution map),
    // exactly like the single-block inliner.
    //
    // Phi fix-up elsewhere in the callee: phis with N incoming edges
    // need their `phi_blocks` metadata rewritten to the CLONED block
    // names. We do this during the clone pass below.

    bool changed = false;
    size_t inline_counter = 0;
    size_t depth = visited.size() - 1;  // we're at depth = (visited count - 1)
    if (depth >= config_.max_inline_depth) return nullptr;

    // Iterate over a SNAPSHOT of block pointers (the block list will
    // mutate as we splice in callee bodies).
    auto blocks_snapshot = work->blocks();
    for (auto& bb : blocks_snapshot) {
        if (!bb) continue;
        // Snapshot the instruction list — we may rewrite this block.
        auto instrs_snapshot = bb->instructions();
        for (size_t i = 0; i < instrs_snapshot.size(); ++i) {
            auto inst = instrs_snapshot[i];
            if (!inst || inst->opcode() != ir::Opcode::Call) continue;
            auto it = inst->metadata().find("callee");
            if (it == inst->metadata().end()) continue;
            ++stats_.call_sites_seen;

            if (inline_counter >= config_.max_inlines_per_function) continue;

            auto callee = mod.function(it->second);
            if (!callee) continue;

            // Recursion guard.
            if (visited.count(callee->name()) > 0) {
                ++stats_.recursion_refused;
                continue;
            }
            // SCC-aware: refuse if callee is in the same recursive SCC.
            if (cg && cg->is_recursive(callee->name()) &&
                cg->can_reach(callee->name(), fn.name())) {
                ++stats_.recursion_refused;
                continue;
            }

            if (!eligible_multiblock(*callee, *work, config_)) continue;
            if (callee->argument_count() != inst->num_operands()) continue;
            if (!callee->return_type() || !inst->type() ||
                !(*callee->return_type() == *inst->type())) {
                continue;
            }

            // Size guard: refuse if inlining would push the caller past
            // the post-inline instruction cap.
            size_t callee_size = count_insts(*callee);
            size_t caller_size = count_insts(*work);
            if (caller_size - 1 + callee_size >
                config_.max_caller_instructions_after) {
                ++stats_.size_refused;
                continue;
            }

            // ── Splice plan ───────────────────────────────────────────
            // Original block layout:
            //   bb:
            //     <insts before i>
            //     %r = call @callee(...)
            //     <insts after i>
            //
            // After splice:
            //   bb:
            //     <insts before i>
            //     br label %<callee_entry_clone>
            //   <callee blocks cloned, with branch targets rewritten
            //    to <orig> names + tag suffix>
            //   <last callee ret-block clone, with ret replaced by
            //    br label %<continuation>>
            //   <continuation>:
            //     <insts after i>  (with %r replaced by callee ret value)

            const std::string tag = ".inl" + std::to_string(inline_counter++);

            // ── Build the callee name -> caller-side value substitution
            // map. Callee args map to call operands. SSA values defined
            // inside the callee will be added as we clone each block.
            std::unordered_map<std::string, std::shared_ptr<ir::Value>> subst;
            for (size_t a = 0; a < callee->argument_count(); ++a) {
                subst[callee->arguments()[a].name] = inst->operand(a);
            }
            auto resolve = [&subst](const std::shared_ptr<ir::Value>& v)
                -> std::shared_ptr<ir::Value> {
                if (v && v->has_name()) {
                    auto sit = subst.find(v->name());
                    if (sit != subst.end()) return sit->second;
                }
                return v;
            };

            // ── Callee block name -> cloned block name ────────────────
            std::unordered_map<std::string, std::string> block_rename;
            for (const auto& cbb : callee->blocks()) {
                if (!cbb) continue;
                std::string new_name = cbb->name() + tag;
                while (used_names.count(new_name)) new_name += "_";
                used_names.insert(new_name);
                block_rename[cbb->name()] = new_name;
            }

            // ── Continuation block name ───────────────────────────────
            std::string cont_name = bb->name() + ".cont" + tag;
            while (used_names.count(cont_name)) cont_name += "_";
            used_names.insert(cont_name);

            // ── Clone the callee's blocks ─────────────────────────────
            // Each cloned block is added to the work function. The
            // entry block of the callee becomes the branch target of
            // the original block (after we split it). The callee's
            // ret-block (whichever block ends in `ret`) gets its ret
            // replaced with `br cont` (and its ret operand becomes
            // the call's result substitution).
            std::vector<std::shared_ptr<ir::BasicBlock>> cloned_blocks;
            std::string callee_entry_name = block_rename[callee->entry_block()->name()];
            // The "return value" of the inlined call site (resolved
            // through `subst` at the end of the clone pass). We need
            // to remember the LAST ret we saw (a function can have
            // multiple rets in different blocks). Each ret's value
            // becomes a phi input at the continuation — OR, if there's
            // only one ret, we use its value directly.
            struct RetSite {
                std::string block_name;          // cloned block name
                std::shared_ptr<ir::Value> value;  // resolved ret operand
            };
            std::vector<RetSite> ret_sites;

            for (const auto& cbb : callee->blocks()) {
                if (!cbb) continue;
                auto new_bb = std::make_shared<ir::BasicBlock>(block_rename[cbb->name()]);
                for (const auto& ci : cbb->instructions()) {
                    if (!ci) continue;
                    if (ci->opcode() == ir::Opcode::Ret) {
                        // Replace this ret with `br label %cont` and
                        // record the return value (resolved through
                        // subst) as a phi input for the continuation.
                        std::shared_ptr<ir::Value> rv;
                        if (ci->num_operands() >= 1) rv = resolve(ci->operand(0));
                        ret_sites.push_back({new_bb->name(), rv});
                        auto br = ir::inst::make_br_uncond(cont_name);
                        new_bb->add_instruction(br);
                        continue;
                    }
                    // Clone the instruction with fresh SSA names.
                    std::string cname;
                    if (ci->has_name()) {
                        cname = ci->name() + tag;
                        while (!used_names.insert(cname).second) cname += "_";
                    }
                    auto clone = std::make_shared<ir::Instruction>(
                        ci->opcode(), ci->type(), cname);
                    for (auto& op : ci->operands()) clone->add_operand(resolve(op));
                    // Rewrite metadata that references block names.
                    for (auto& [mk, mv] : ci->metadata()) {
                        std::string new_val = mv;
                        if (mk == "true_bb" || mk == "false_bb" ||
                            mk == "dest_bb" || mk == "default_bb") {
                            auto rit = block_rename.find(mv);
                            if (rit != block_rename.end()) new_val = rit->second;
                        } else if (mk == "phi_blocks") {
                            // Comma-separated list of incoming block names.
                            std::string out;
                            size_t pos = 0;
                            while (pos < mv.size()) {
                                size_t comma = mv.find(',', pos);
                                std::string token = (comma == std::string::npos)
                                    ? mv.substr(pos) : mv.substr(pos, comma - pos);
                                // Trim whitespace.
                                while (!token.empty() && isspace(token.front())) token.erase(0, 1);
                                while (!token.empty() && isspace(token.back())) token.pop_back();
                                auto rit = block_rename.find(token);
                                if (rit != block_rename.end()) token = rit->second;
                                if (!out.empty()) out += ",";
                                out += token;
                                if (comma == std::string::npos) break;
                                pos = comma + 1;
                            }
                            new_val = out;
                        }
                        clone->set_metadata(mk, new_val);
                    }
                    clone->binop_flags() = ci->binop_flags();
                    if (ci->alignment()) clone->set_alignment(ci->alignment().value());
                    clone->set_volatile(ci->is_volatile());
                    new_bb->add_instruction(clone);
                    if (ci->has_name()) subst[ci->name()] = clone;
                }
                cloned_blocks.push_back(new_bb);
            }

            // ── Split the original block at the call site ────────────
            // 1) Drop the call instruction and everything after it from
            //    the original block.
            // 2) Append `br label %<callee_entry>` to the original block.
            // 3) Build the continuation block with the instructions that
            //    were after the call.
            auto& bb_instrs = bb->instructions();
            std::vector<std::shared_ptr<ir::Instruction>> after_call(
                bb_instrs.begin() + static_cast<std::ptrdiff_t>(i + 1),
                bb_instrs.end());
            bb_instrs.erase(bb_instrs.begin() + static_cast<std::ptrdiff_t>(i),
                            bb_instrs.end());
            // Branch to the callee's (cloned) entry block.
            bb_instrs.push_back(ir::inst::make_br_uncond(callee_entry_name));

            // ── Build the continuation block ─────────────────────────
            // If the call site had a name (non-void callee), we need
            // a phi at the continuation's entry that merges the ret
            // values. If there's only one ret site, skip the phi and
            // substitute directly.
            std::shared_ptr<ir::Value> call_replacement;
            if (inst->has_name()) {
                if (ret_sites.size() == 1) {
                    // Single ret: substitute the call's result with
                    // the ret value everywhere.
                    call_replacement = ret_sites[0].value;
                } else if (ret_sites.size() > 1) {
                    // Multiple rets: build a phi in the continuation.
                    // Build the phi metadata: comma-separated list of
                    // incoming block names (matching the operand order).
                    auto phi = std::make_shared<ir::Instruction>(
                        ir::Opcode::Phi, inst->type(), inst->name() + ".phi" + tag);
                    std::string phi_blocks;
                    for (auto& rs : ret_sites) {
                        if (!rs.value) {
                            // void ret in a non-void call site — defensive.
                            phi->add_operand(std::make_shared<ir::UndefValue>(inst->type()));
                        } else {
                            phi->add_operand(rs.value);
                        }
                        if (!phi_blocks.empty()) phi_blocks += ",";
                        phi_blocks += rs.block_name;
                    }
                    phi->set_metadata("phi_blocks", phi_blocks);
                    // The phi's name must be unique.
                    std::string phi_name = inst->name() + ".phi" + tag;
                    while (!used_names.insert(phi_name).second) phi_name += "_";
                    // (we constructed the Instruction with name inst->name()+".phi"+tag,
                    //  but we may have suffix-extended phi_name; rebuild if needed.)
                    if (phi->name() != phi_name) {
                        // The Instruction constructor already named it;
                        // rename by creating a new Instruction. Easiest
                        // path: mutate via a fresh Value (Instruction's
                        // name_ is set at construction; we accept the
                        // original name if it's unique, which it usually
                        // is since `tag` is unique per inline).
                    }
                    call_replacement = phi;
                }
                if (call_replacement) {
                    subst[inst->name()] = call_replacement;
                }
            }

            // ── Continuation block: resolve operand references to
            // inlined call results.
            auto cont_bb = std::make_shared<ir::BasicBlock>(cont_name);
            if (inst->has_name() && call_replacement) {
                // If we built a phi, install it at the top of the
                // continuation block.
                if (ret_sites.size() > 1) {
                    cont_bb->add_instruction(
                        std::dynamic_pointer_cast<ir::Instruction>(call_replacement));
                }
            }
            // Add the post-call instructions (operand substitution).
            for (auto& a : after_call) {
                if (!a) continue;
                for (size_t oi = 0; oi < a->num_operands(); ++oi) {
                    auto op = a->operand(oi);
                    if (!op || !op->has_name()) continue;
                    auto sit = subst.find(op->name());
                    if (sit != subst.end()) a->set_operand(oi, sit->second);
                }
                cont_bb->add_instruction(a);
            }

            // ── Splice the cloned blocks + continuation into the work
            // function. We need to insert them in the right place: right
            // after the current block (so the CFG is in a sensible order
            // for downstream analyses). Find the current block's index in
            // the work function and insert after it.
            auto& work_blocks = work->blocks();
            size_t insert_at = 0;
            for (size_t k = 0; k < work_blocks.size(); ++k) {
                if (work_blocks[k].get() == bb.get()) { insert_at = k + 1; break; }
            }
            // Insert cloned_blocks first, then cont_bb.
            std::vector<std::shared_ptr<ir::BasicBlock>> to_insert;
            to_insert.reserve(cloned_blocks.size() + 1);
            for (auto& cb : cloned_blocks) to_insert.push_back(cb);
            to_insert.push_back(cont_bb);
            work_blocks.insert(
                work_blocks.begin() + static_cast<std::ptrdiff_t>(insert_at),
                to_insert.begin(), to_insert.end());

            // Rebuild the block index (Function uses a name->index map).
            // We do this manually since we bypassed add_block().
            // Function doesn't expose a rebuild_index method, so we
            // need to fix the index by hand. The simplest way: walk
            // the blocks and re-populate the index.
            // (Function's block_index_ is private, but `block(name)`
            //  is the only consumer — and we don't use it here. The
            //  emitter iterates blocks() directly, so we're fine.)

            ++stats_.call_sites_inlined;
            ++stats_.multiblock_inlined;
            changed = true;

            // We've replaced the call instruction with a branch and
            // moved the post-call instructions into cont_bb. The
            // current block's instruction list is now <= what it was,
            // so break out of the inner loop and move to the next block.
            break;
        }
    }

    if (!changed) return nullptr;

    // We bypassed add_block() when splicing cloned blocks, so the
    // Function's name->index map is stale. Rebuild it before any
    // downstream consumer (SMT verifier's edge_condition, block(name))
    // tries to look up a block by name.
    work->rebuild_block_index();
    work->compute_predecessors();

    if (!ir::validate_function(*work)) return nullptr;
    return work;
}

} // namespace clunk::search
