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
 * Clunk Inliner — single-block call-site inlining.
 * See include/clunk/Search/Inliner.h for the eligibility contract.
 */
#include "clunk/Search/Inliner.h"

#include <unordered_map>
#include <unordered_set>

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

// Is `callee` inlinable under the header's contract?
bool eligible(const ir::Function& callee, const ir::Function& caller,
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
    // Every callee argument must be named (substitution is by name).
    for (auto& arg : callee.arguments()) {
        if (arg.name.empty()) return false;
    }
    return true;
}

} // anonymous namespace

Inliner::Inliner(const InlinerConfig& config) : config_(config) {}

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
            if (!callee || !eligible(*callee, *work, config_.max_callee_instructions))
                continue;
            if (callee->argument_count() != inst->num_operands()) continue;
            // The call's result type must match the callee's return type —
            // a mismatch means the call site was lying about the signature.
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

} // namespace clunk::search
