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

#pragma once
/*
 * Clunk CallGraph — module-level call-graph analysis.
 *
 * The pipeline optimises functions independently, but real code calls real
 * functions. A call graph is the prerequisite for every cross-function
 * optimisation: inlining (depth-limited recursion), dead-function
 * elimination, interprocedural constant propagation, and SMT-level
 * reasoning about call sites via function summaries.
 *
 * The graph is built by a single walk over every instruction in every
 * function: every `call @callee(...)` instruction contributes an edge
 * caller -> callee. Indirect calls (call through a function pointer) and
 * calls to external/declaration-only callees contribute a "external call"
 * marker instead, which the consumer can treat as opaque.
 *
 * SCCs are computed with Tarjan's algorithm; an SCC of size > 1 (or a
 * self-loop on a single node) identifies recursion. This is what the
 * inliner's call-graph walk uses to refuse to inline into itself.
 */
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "clunk/IR/Module.h"

namespace clunk::analysis {

class CallGraph {
public:
    struct Node {
        std::string function_name;
        // Direct callees (deduplicated, in insertion order).
        std::vector<std::string> callees;
        // Direct callers (deduplicated).
        std::vector<std::string> callers;
        bool is_definition = false;     // has a body in this module
        bool is_external = false;       // declared but not defined
        bool has_indirect_call = false; // caller emits an indirect call
        bool is_entry = false;          // callable from outside the module
                                          // (no callers, or marked externally_visible)
        size_t scc_id = 0;              // Tarjan SCC id
        bool is_recursive = false;      // in an SCC of size > 1 OR has self-loop
    };

    struct SCC {
        size_t id;
        std::vector<std::string> members;  // functions in this SCC
        bool is_recursive;                  // size > 1 OR single self-loop
    };

    CallGraph() = default;

    // Build the call graph for `module`. Replaces any existing state.
    // `entry_function_names` (optional): functions explicitly marked as
    // entry points (e.g. `main`, kernels). Functions with no callers AND
    // not in this list are flagged as candidates for dead-function
    // elimination — but the consumer is responsible for honouring linkage
    // (ExternalLinkage functions are reachable from outside and must not
    // be eliminated).
    void build(const ir::Module& module,
               const std::vector<std::string>& entry_function_names = {});

    // Lookup a node by function name. Returns nullptr if unknown.
    const Node* node(const std::string& fn_name) const {
        auto it = nodes_.find(fn_name);
        return it == nodes_.end() ? nullptr : &it->second;
    }

    // All nodes, in insertion order (= module order from build()).
    const std::vector<Node>& nodes() const { return node_order_; }

    // All SCCs in reverse-topological order (leaf SCCs first).
    const std::vector<SCC>& sccs() const { return sccs_; }

    // True if `caller` directly calls `callee` (one-step edge).
    bool calls(const std::string& caller, const std::string& callee) const;

    // True if `caller` can reach `callee` via any number of calls.
    // Conservative: returns true for any indirect-call site.
    bool can_reach(const std::string& caller, const std::string& callee) const;

    // True if `fn_name` is in a recursive SCC (or has a self-loop).
    bool is_recursive(const std::string& fn_name) const;

    // Functions never called by any other function in the module AND not
    // flagged as entry points AND not externally visible (by linkage or
    // by the entry list). These are dead and can be eliminated.
    std::vector<std::string> dead_functions(
        const std::unordered_set<std::string>& externally_visible) const;

    // For each function argument position across the module, returns the
    // set of constant values passed by ALL direct callers (when known).
    // If a function is called indirectly or by an external caller, that
    // argument's constant-set is empty (no specialisation possible).
    //
    // Result:  function_name -> vector over arg positions of constant sets.
    // Each constant set is empty if the position is not uniformly constant.
    struct ArgConstantInfo {
        // arg position -> {constant values seen across all callers}
        // (size == argument_count of the function; empty set = not constant)
        std::vector<std::unordered_set<int64_t>> per_arg;
        // True if ANY call site is indirect or external — disables
        // specialisation for the whole function.
        bool has_unknown_caller = false;
    };
    const std::unordered_map<std::string, ArgConstantInfo>&
    argument_constants() const { return arg_constants_; }

    // Statistics
    size_t node_count() const { return nodes_.size(); }
    size_t edge_count() const;

private:
    std::unordered_map<std::string, Node> nodes_;
    std::vector<Node> node_order_;
    std::vector<SCC> sccs_;
    std::unordered_map<std::string, ArgConstantInfo> arg_constants_;

    void compute_sccs();
    void collect_argument_constants(const ir::Module& module);
};

} // namespace clunk::analysis
