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
 * Clunk Cross-Function Passes — DFE + IPCP.
 * See include/clunk/Search/CrossFunctionPasses.h for the public API.
 */
#include "clunk/Search/CrossFunctionPasses.h"

#include <algorithm>
#include <cstdint>
#include <unordered_map>
#include <unordered_set>

#include "clunk/Analysis/CallGraph.h"
#include "clunk/IR/Clone.h"
#include "clunk/IR/Function.h"
#include "clunk/IR/Instruction.h"
#include "clunk/IR/Value.h"

namespace clunk::search {

namespace {

// Walk every call instruction in every function and rewrite the callee
// name from `old_name` to `new_name`. Used by IPCP to redirect callers
// at the original function to the specialised clone.
size_t rewrite_callers(ir::Module& module,
                       const std::string& old_name,
                       const std::string& new_name) {
    size_t rewritten = 0;
    for (auto& fn : module.functions()) {
        if (!fn) continue;
        for (auto& bb : fn->blocks()) {
            if (!bb) continue;
            for (auto& inst : bb->instructions()) {
                if (!inst || inst->opcode() != ir::Opcode::Call) continue;
                auto& md = const_cast<std::unordered_map<std::string, std::string>&>(
                    inst->metadata());
                auto it = md.find("callee");
                if (it == md.end()) continue;
                if (it->second == old_name) {
                    it->second = new_name;
                    ++rewritten;
                }
            }
        }
    }
    return rewritten;
}

// Replace every use of `arg_name` (a function argument) inside `fn`
// with `replacement`. Used by IPCP after substituting a constant for
// the specialised argument.
size_t substitute_arg_uses(ir::Function& fn,
                           const std::string& arg_name,
                           std::shared_ptr<ir::Value> replacement) {
    size_t count = 0;
    for (auto& bb : fn.blocks()) {
        if (!bb) continue;
        for (auto& inst : bb->instructions()) {
            if (!inst) continue;
            for (size_t i = 0; i < inst->num_operands(); ++i) {
                auto op = inst->operand(i);
                if (!op || !op->has_name()) continue;
                if (op->name() == arg_name) {
                    inst->set_operand(i, replacement);
                    ++count;
                }
            }
        }
    }
    return count;
}

// Count total instructions in a function.
size_t count_insts(const ir::Function& fn) {
    size_t n = 0;
    for (const auto& bb : fn.blocks()) {
        if (bb) n += bb->size();
    }
    return n;
}

// Clone a function under a new name. The clone is a deep copy with a
// fresh name and Internal linkage (it's only callable from within the
// module — that's the whole point of IPCP specialisation).
std::shared_ptr<ir::Function> clone_function(const ir::Function& src,
                                             const std::string& new_name) {
    auto copy = ir::deep_copy_function(src);
    // Rename. We can't change the name through the public API, so we
    // rebuild the Function with the new name and steal the blocks.
    auto renamed = std::make_shared<ir::Function>(new_name, src.function_type(),
                                                   ir::Linkage::Internal);
    for (auto& arg : src.arguments()) {
        renamed->add_argument(arg.type, arg.name, arg.attrs);
    }
    for (auto& [k, v] : src.attributes()) {
        renamed->set_attribute(k, v);
    }
    // Move the cloned blocks into the renamed function.
    for (auto& bb : copy->blocks()) {
        if (!bb) continue;
        auto& nb = renamed->add_block(bb->name());
        for (auto& inst : bb->instructions()) {
            nb.add_instruction(inst);
        }
    }
    renamed->rebuild_block_index();
    renamed->compute_predecessors();
    return renamed;
}

// Pick the best argument position to specialise for a function. Heuristic:
// the position with the smallest per_arg set (size 1 = uniform constant)
// and the most uses inside the function body. We approximate "most uses"
// by counting argument references in the source function.
struct SpecialisationCandidate {
    std::string fn_name;
    size_t arg_pos;
    int64_t constant_value;
    unsigned arg_bit_width;
};
std::vector<SpecialisationCandidate> find_ipcp_candidates(
    const ir::Module& module,
    const analysis::CallGraph& cg,
    const CrossFnConfig& cfg) {
    std::vector<SpecialisationCandidate> out;
    for (const auto& fn : module.functions()) {
        if (!fn || fn->blocks().empty()) continue;
        size_t inst_count = count_insts(*fn);
        if (inst_count < cfg.ipcp_min_fn_size) continue;
        if (inst_count > cfg.ipcp_max_fn_size) continue;
        const auto& ac = cg.argument_constants();
        auto it = ac.find(fn->name());
        if (it == ac.end()) continue;
        if (it->second.has_unknown_caller) continue;
        const auto& per_arg = it->second.per_arg;
        for (size_t i = 0; i < per_arg.size(); ++i) {
            // Uniform constant: exactly one value, no sentinel.
            if (per_arg[i].size() != 1) continue;
            int64_t v = *per_arg[i].begin();
            if (v == INT64_MIN) continue;  // sentinel — non-constant
            // Argument must be an integer.
            if (i >= fn->argument_count()) continue;
            auto& arg = fn->arguments()[i];
            if (!arg.type || !arg.type->is_integer()) continue;
            unsigned bits = arg.type->bit_width();
            if (bits == 0 || bits > 64) continue;
            out.push_back({fn->name(), i, v, bits});
        }
    }
    return out;
}

} // namespace

// ── run ────────────────────────────────────────────────────────────────────

bool CrossFunctionPasses::run(ir::Module& module) {
    bool changed = false;
    if (config_.enable_dfe) {
        if (run_dead_function_elimination(module)) changed = true;
    }
    if (config_.enable_ipcp) {
        if (run_ipcp(module)) changed = true;
        // After IPCP creates specialised clones, more functions might be
        // dead (the original, if all callers were rewritten). Run DFE
        // again to clean up.
        if (config_.enable_dfe && run_dead_function_elimination(module)) {
            // already counted in changed
        }
    }
    return changed;
}

// ── DFE ────────────────────────────────────────────────────────────────────

bool CrossFunctionPasses::run_dead_function_elimination(ir::Module& module) {
    analysis::CallGraph cg;
    cg.build(module);

    // Externally visible functions are kept — they could be called from
    // outside the module. By default, ExternalLinkage functions are
    // externally visible. Internal/Private are not.
    std::unordered_set<std::string> externally_visible;
    for (const auto& fn : module.functions()) {
        if (!fn) continue;
        if (fn->linkage() == ir::Linkage::External ||
            fn->linkage() == ir::Linkage::LinkOnce ||
            fn->linkage() == ir::Linkage::Weak ||
            fn->linkage() == ir::Linkage::LinkOnceODR ||
            fn->linkage() == ir::Linkage::WeakODR ||
            fn->linkage() == ir::Linkage::AvailableExternally ||
            fn->linkage() == ir::Linkage::ExternalWeak ||
            fn->linkage() == ir::Linkage::Common ||
            fn->linkage() == ir::Linkage::Appending) {
            externally_visible.insert(fn->name());
        }
    }
    // Also never remove `main` or anything that looks like a kernel.
    externally_visible.insert("main");
    for (const auto& fn : module.functions()) {
        if (fn && fn->is_gpu_kernel()) externally_visible.insert(fn->name());
    }

    auto dead = cg.dead_functions(externally_visible);
    if (dead.empty()) return false;

    // Iterate to fixed point: removing a function might make its callees
    // dead too (transitive DFE).
    bool changed = false;
    std::unordered_set<std::string> removed;
    while (!dead.empty()) {
        for (const auto& name : dead) {
            if (removed.count(name)) continue;
            module.remove_function(name);
            removed.insert(name);
            ++stats_.dfe_removed;
            changed = true;
        }
        // Rebuild the call graph and check again.
        analysis::CallGraph cg2;
        cg2.build(module);
        dead = cg2.dead_functions(externally_visible);
        // Only consider newly-dead functions.
        std::vector<std::string> new_dead;
        for (const auto& name : dead) {
            if (removed.count(name) == 0) new_dead.push_back(name);
        }
        dead = std::move(new_dead);
    }
    return changed;
}

// ── IPCP ───────────────────────────────────────────────────────────────────

bool CrossFunctionPasses::run_ipcp(ir::Module& module) {
    analysis::CallGraph cg;
    cg.build(module);

    auto candidates = find_ipcp_candidates(module, cg, config_);
    if (candidates.empty()) return false;

    // Limit: at most ipcp_max_total_clones globally. We process candidates
    // in a stable order (by function name, then arg position) so the
    // behaviour is deterministic.
    std::sort(candidates.begin(), candidates.end(),
              [](const SpecialisationCandidate& a,
                 const SpecialisationCandidate& b) {
                  if (a.fn_name != b.fn_name) return a.fn_name < b.fn_name;
                  return a.arg_pos < b.arg_pos;
              });

    // Per-function clone counter (cap = ipcp_max_clones_per_fn).
    std::unordered_map<std::string, size_t> clones_per_fn;
    size_t total_clones = 0;
    bool changed = false;

    for (const auto& cand : candidates) {
        if (total_clones >= config_.ipcp_max_total_clones) break;
        if (clones_per_fn[cand.fn_name] >= config_.ipcp_max_clones_per_fn) {
            continue;
        }

        auto fn = module.function(cand.fn_name);
        if (!fn) continue;

        // Defensive: the call graph might be stale if we've already
        // rewritten callers. Re-check the candidate by re-reading the
        // call graph for this function's argument `cand.arg_pos`.
        analysis::CallGraph fresh_cg;
        fresh_cg.build(module);
        const auto& ac = fresh_cg.argument_constants();
        auto it = ac.find(cand.fn_name);
        if (it == ac.end() || it->second.has_unknown_caller) continue;
        if (cand.arg_pos >= it->second.per_arg.size()) continue;
        const auto& cset = it->second.per_arg[cand.arg_pos];
        if (cset.size() != 1) continue;
        int64_t v = *cset.begin();
        if (v == INT64_MIN) continue;
        if (v != cand.constant_value) continue;  // changed in the meantime

        // Build a clone with the constant substituted for the argument.
        std::string clone_name = cand.fn_name + ".ipcp_" +
            std::to_string(cand.arg_pos) + "_" + std::to_string(v);
        // Ensure the clone name is unique.
        size_t disamb = 0;
        std::string final_name = clone_name;
        while (module.function(final_name)) {
            final_name = clone_name + "_" + std::to_string(++disamb);
        }

        auto clone = clone_function(*fn, final_name);
        if (!clone) continue;

        // Replace every use of the argument inside the clone with a
        // ConstantInt of the specialised value.
        if (cand.arg_pos >= clone->argument_count()) continue;
        auto& arg = clone->arguments()[cand.arg_pos];
        if (!arg.type || !arg.type->is_integer()) continue;
        auto int_ty = std::dynamic_pointer_cast<ir::IntegerType>(arg.type);
        if (!int_ty) continue;
        auto const_val = std::make_shared<ir::ConstantInt>(int_ty,
            static_cast<int64_t>(v));
        substitute_arg_uses(*clone, arg.name, const_val);

        // Add the clone to the module.
        module.add_function(clone);
        ++clones_per_fn[cand.fn_name];
        ++total_clones;
        ++stats_.ipcp_cloned;

        // Rewrite all callers of `fn_name` to invoke `final_name` instead.
        size_t rew = rewrite_callers(module, cand.fn_name, final_name);
        stats_.ipcp_callers_rewritten += rew;
        if (rew > 0) changed = true;
    }

    return changed;
}

} // namespace clunk::search
