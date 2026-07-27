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
 * Clunk CallGraph — module-level call-graph analysis implementation.
 * See include/clunk/Analysis/CallGraph.h for the public API.
 */
#include "clunk/Analysis/CallGraph.h"

#include <algorithm>
#include <cstdint>
#include <stack>
#include <stdexcept>

#include "clunk/IR/Function.h"
#include "clunk/IR/Instruction.h"
#include "clunk/IR/Value.h"

namespace clunk::analysis {

namespace {

// Extract the callee name from a Call instruction's metadata, or empty
// if the call is indirect (callee is a function pointer, not a name).
std::string callee_of(const ir::Instruction& inst) {
    if (inst.opcode() != ir::Opcode::Call) return {};
    const auto& md = inst.metadata();
    auto it = md.find("callee");
    if (it == md.end()) return {};
    return it->second;
}

// Walk all instructions of a function and collect callee names plus
// whether any indirect call exists. Returns the callees in source order
// (with duplicates, which the caller dedups).
struct CallScan {
    std::vector<std::string> callees;
    bool has_indirect = false;
};

CallScan scan_calls(const ir::Function& fn) {
    CallScan out;
    for (const auto& bb : fn.blocks()) {
        if (!bb) continue;
        for (const auto& inst : bb->instructions()) {
            if (!inst || inst->opcode() != ir::Opcode::Call) continue;
            std::string name = callee_of(*inst);
            if (name.empty()) {
                out.has_indirect = true;
            } else {
                out.callees.push_back(std::move(name));
            }
        }
    }
    return out;
}

// Try to read a ConstantInt operand as an int64. Returns nullopt on
// anything other than a ConstantInt (or out-of-range). Used by the
// argument-constants collector.
bool try_read_constant_int(const std::shared_ptr<ir::Value>& v,
                           int64_t& out) {
    if (!v) return false;
    auto ci = std::dynamic_pointer_cast<ir::ConstantInt>(v);
    if (!ci) return false;
    out = ci->value();
    return true;
}

} // namespace

// ── build ──────────────────────────────────────────────────────────────────

void CallGraph::build(const ir::Module& module,
                      const std::vector<std::string>& entry_function_names) {
    nodes_.clear();
    node_order_.clear();
    sccs_.clear();
    arg_constants_.clear();

    // First pass: register every function (definition or declaration).
    std::unordered_set<std::string> entries(entry_function_names.begin(),
                                            entry_function_names.end());
    // Always treat `main` as an entry point if present.
    entries.insert("main");

    for (const auto& fn : module.functions()) {
        if (!fn) continue;
        Node n;
        n.function_name = fn->name();
        n.is_definition = !fn->blocks().empty();
        n.is_external = !n.is_definition;
        n.is_entry = entries.count(fn->name()) > 0 ||
                     fn->linkage() == ir::Linkage::External;
        nodes_[fn->name()] = std::move(n);
    }

    // Second pass: scan each definition for calls.
    for (const auto& fn : module.functions()) {
        if (!fn || fn->blocks().empty()) continue;
        const std::string caller = fn->name();
        auto scan = scan_calls(*fn);

        auto& caller_node = nodes_[caller];
        caller_node.has_indirect_call = scan.has_indirect;

        std::unordered_set<std::string> seen_callee;
        for (auto& callee : scan.callees) {
            // Ensure callee exists as a node even if undefined.
            if (nodes_.find(callee) == nodes_.end()) {
                Node n;
                n.function_name = callee;
                n.is_definition = false;
                n.is_external = true;
                n.is_entry = true;  // unknown -> conservatively treat as entry
                nodes_[callee] = std::move(n);
            }
            if (seen_callee.insert(callee).second) {
                caller_node.callees.push_back(callee);
                nodes_[callee].callers.push_back(caller);
            }
        }
    }

    // Materialise node_order_ in module order, then unknowns last.
    for (const auto& fn : module.functions()) {
        if (!fn) continue;
        auto it = nodes_.find(fn->name());
        if (it != nodes_.end()) node_order_.push_back(it->second);
    }
    for (auto& [name, n] : nodes_) {
        bool seen = false;
        for (const auto& no : node_order_) {
            if (no.function_name == name) { seen = true; break; }
        }
        if (!seen) node_order_.push_back(n);
    }

    // Compute SCCs (Tarjan, iterative to avoid stack overflow on big modules).
    compute_sccs();

    // Collect per-argument constant info for IPCP.
    collect_argument_constants(module);
}

// ── Tarjan's SCC (iterative) ───────────────────────────────────────────────

void CallGraph::compute_sccs() {
    const size_t N = node_order_.size();
    if (N == 0) return;

    // Index nodes by name for adjacency.
    std::unordered_map<std::string, size_t> idx;
    for (size_t i = 0; i < N; ++i) idx[node_order_[i].function_name] = i;

    std::vector<int> index(N, -1), lowlink(N, 0);
    std::vector<bool> on_stack(N, false);
    std::vector<size_t> visit_stack;
    std::vector<size_t> scc_of(N, SIZE_MAX);
    int counter = 0;

    // Iterative Tarjan: we manage our own stack of (v, child-iterator) pairs.
    struct Frame {
        size_t v;
        size_t child_idx;  // next callee index to visit
    };
    std::vector<Frame> call_stack;

    size_t next_scc_id = 0;
    sccs_.clear();

    for (size_t start = 0; start < N; ++start) {
        if (index[start] != -1) continue;

        call_stack.push_back({start, 0});
        while (!call_stack.empty()) {
            auto& top = call_stack.back();
            size_t v = top.v;
            if (top.child_idx == 0) {
                // First visit: assign index/lowlink, push onto stack.
                index[v] = counter;
                lowlink[v] = counter;
                ++counter;
                visit_stack.push_back(v);
                on_stack[v] = true;
            }

            const auto& callees = node_order_[v].callees;
            bool recursed = false;
            while (top.child_idx < callees.size()) {
                auto it = idx.find(callees[top.child_idx]);
                ++top.child_idx;
                if (it == idx.end()) continue;
                size_t w = it->second;
                if (index[w] == -1) {
                    call_stack.push_back({w, 0});
                    recursed = true;
                    break;
                } else if (on_stack[w]) {
                    if (index[w] < lowlink[v]) lowlink[v] = index[w];
                }
            }
            if (recursed) continue;

            // All children processed: check for SCC root.
            if (lowlink[v] == index[v]) {
                SCC scc;
                scc.id = next_scc_id++;
                size_t w;
                do {
                    w = visit_stack.back();
                    visit_stack.pop_back();
                    on_stack[w] = false;
                    scc_of[w] = scc.id;
                    scc.members.push_back(node_order_[w].function_name);
                } while (w != v);
                // Recursive iff size > 1 OR (size 1 AND has self-loop).
                scc.is_recursive = scc.members.size() > 1;
                if (!scc.is_recursive) {
                    const auto& callees_self = node_order_[v].callees;
                    for (const auto& c : callees_self) {
                        if (c == node_order_[v].function_name) {
                            scc.is_recursive = true;
                            break;
                        }
                    }
                }
                sccs_.push_back(std::move(scc));
            }

            call_stack.pop_back();
            if (!call_stack.empty()) {
                size_t parent = call_stack.back().v;
                if (lowlink[v] < lowlink[parent]) lowlink[parent] = lowlink[v];
            }
        }
    }

    // SCCs are produced in reverse-topological order already (Tarjan's
    // guarantee). Reverse to get topological order (callees first), then
    // reverse again to keep the convention "leaf SCCs first" — Tarjan
    // naturally produces roots LAST, so we want to keep that order.
    // Actually: Tarjan produces SCCs in reverse topological order, meaning
    // a SCC is emitted after all SCCs it depends on. So sccs_ as-is has
    // leaf SCCs FIRST. Good.

    // Annotate each node with scc_id and is_recursive.
    for (size_t i = 0; i < N; ++i) {
        node_order_[i].scc_id = scc_of[i];
        node_order_[i].is_recursive = sccs_[scc_of[i]].is_recursive;
        nodes_[node_order_[i].function_name].scc_id = scc_of[i];
        nodes_[node_order_[i].function_name].is_recursive =
            sccs_[scc_of[i]].is_recursive;
    }
}

// ── query helpers ──────────────────────────────────────────────────────────

bool CallGraph::calls(const std::string& caller,
                      const std::string& callee) const {
    auto it = nodes_.find(caller);
    if (it == nodes_.end()) return false;
    for (const auto& c : it->second.callees) {
        if (c == callee) return true;
    }
    return false;
}

bool CallGraph::can_reach(const std::string& caller,
                          const std::string& callee) const {
    if (caller == callee) return true;
    auto it = nodes_.find(caller);
    if (it == nodes_.end()) return false;
    if (it->second.has_indirect_call) return true;  // conservative

    // BFS over the call graph.
    std::unordered_set<std::string> visited;
    std::vector<std::string> stack = {caller};
    while (!stack.empty()) {
        std::string cur = std::move(stack.back());
        stack.pop_back();
        if (!visited.insert(cur).second) continue;
        if (cur == callee) return true;
        auto nit = nodes_.find(cur);
        if (nit == nodes_.end()) continue;
        if (nit->second.has_indirect_call) return true;  // conservative
        for (const auto& c : nit->second.callees) {
            if (visited.count(c) == 0) stack.push_back(c);
        }
    }
    return false;
}

bool CallGraph::is_recursive(const std::string& fn_name) const {
    auto it = nodes_.find(fn_name);
    return it != nodes_.end() && it->second.is_recursive;
}

size_t CallGraph::edge_count() const {
    size_t n = 0;
    for (const auto& [_, node] : nodes_) n += node.callees.size();
    return n;
}

// ── dead_functions ─────────────────────────────────────────────────────────

std::vector<std::string> CallGraph::dead_functions(
    const std::unordered_set<std::string>& externally_visible) const {
    std::vector<std::string> dead;
    for (const auto& [name, node] : nodes_) {
        if (!node.is_definition) continue;          // declarations never dead
        if (node.is_entry) continue;                // user-marked entry
        if (externally_visible.count(name)) continue; // linkage-external
        if (node.callers.empty()) {
            // No callers in this module AND not flagged as entry — dead.
            dead.push_back(name);
        }
    }
    return dead;
}

// ── argument-constants collection (for IPCP) ───────────────────────────────

void CallGraph::collect_argument_constants(const ir::Module& module) {
    // Initialise per-function info with the right arg-count.
    for (const auto& fn : module.functions()) {
        if (!fn) continue;
        ArgConstantInfo info;
        info.per_arg.resize(fn->argument_count());
        arg_constants_[fn->name()] = std::move(info);
    }

    // Walk every call site; record constant operands per callee-arg.
    for (const auto& caller : module.functions()) {
        if (!caller || caller->blocks().empty()) continue;
        for (const auto& bb : caller->blocks()) {
            if (!bb) continue;
            for (const auto& inst : bb->instructions()) {
                if (!inst || inst->opcode() != ir::Opcode::Call) continue;
                std::string callee = callee_of(*inst);
                if (callee.empty()) {
                    // Indirect call: anything could be the target. Mark
                    // every definition as having an unknown caller — but
                    // since we don't know the target, the only sound thing
                    // is to bail conservatively for ALL functions.
                    for (auto& [_, info] : arg_constants_) {
                        info.has_unknown_caller = true;
                    }
                    continue;
                }
                auto it = arg_constants_.find(callee);
                if (it == arg_constants_.end()) continue;
                auto& info = it->second;
                // External caller of this callee? If the callee is
                // externally visible, we don't know who else calls it.
                auto callee_node = nodes_.find(callee);
                if (callee_node != nodes_.end() && callee_node->second.is_entry) {
                    info.has_unknown_caller = true;
                }
                // Record constant operands per arg position.
                for (size_t i = 0; i < info.per_arg.size(); ++i) {
                    if (i >= inst->num_operands()) break;
                    int64_t v;
                    if (try_read_constant_int(inst->operand(i), v)) {
                        info.per_arg[i].insert(v);
                    } else {
                        // Non-constant argument: mark as "no uniform
                        // constant" by inserting a sentinel? No — we
                        // want to ALLOW specialisation only when ALL
                        // callers pass a constant, AND it's the SAME
                        // constant. So we need to track that this slot
                        // is "non-constant for at least one call" —
                        // represent this by inserting a sentinel value
                        // that we filter out at query time. We use
                        // INT64_MIN as the sentinel.
                        info.per_arg[i].insert(INT64_MIN);
                    }
                }
            }
        }
    }
}

} // namespace clunk::analysis
