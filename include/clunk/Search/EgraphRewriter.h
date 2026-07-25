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
 * Clunk EgraphRewriter — egg-style equality-saturation candidate generation.
 *
 * A e-graph data structure and equality-saturation driver
 * that takes an ir::Function + pattern::PatternLibrary, saturates the
 * e-graph with the library's source→replacement rewrites, and extracts
 * the cheapest representative per e-class as a fresh ir::Function.
 *
 * The extracted candidate is returned as a search::Candidate with
 * sound=false (the rewrites are pattern-library-sourced and need SMT
 * verification before adoption — see Pipeline.cpp's verify_and_select).
 *
 * This module solves the pattern-ordering problem (today
 * Pipeline::apply_patterns tries patterns sequentially and may miss wins
 * that require a different order). Equality saturation applies ALL
 * rewrites in ALL orders simultaneously and extracts the cheapest result.
 *
 * Self-contained: no external dependencies (no egg, no Boost). C++17,
 * -Wall -Wextra -Wpedantic clean.
 */
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "clunk/IR/Instruction.h"
#include "clunk/IR/Function.h"
#include "clunk/Pattern/PatternLibrary.h"
#include "clunk/Evaluator/EvaluationEngine.h"
#include "clunk/Search/StochasticSearch.h"  // for search::Candidate

namespace clunk::egraph {

// ── E-class ID type ──────────────────────────────────────────────────────
using EClassId = uint32_t;

// Sentinel for "no e-class".
constexpr EClassId kInvalidEClassId = static_cast<EClassId>(-1);

// ── E-node ───────────────────────────────────────────────────────────────
// An e-node is either:
//   • a constant leaf   (is_constant = true, constant_value set),
//   • an argument leaf  (is_argument = true, argument_name set), or
//   • an opcode node    (opcode + children + optional predicate/flags).
struct ENode {
    ir::Opcode opcode = ir::Opcode::Ret;
    std::vector<EClassId> children;
    int64_t constant_value = 0;
    bool is_constant = false;
    bool is_argument = false;
    std::string argument_name;       // valid iff is_argument
    ir::CmpPredicate predicate = ir::CmpPredicate::EQ;
    ir::BinOpFlags flags{};
    // The IR type this node's value actually had at lowering time.
    // Deliberately NOT part of operator==/ENodeHash: two structurally
    // identical nodes (same opcode+children) in a well-typed program
    // already have the same result type, so leaving it out of congruence
    // doesn't cause unsound merges — it's carried purely so extract()
    // can reconstruct the exact original type instead of guessing one
    // from operand types, which is wrong for any opcode whose result
    // type isn't simply "same as its first operand" (Alloca, Load,
    // GetElementPtr, Select, casts, ...). See build_instruction().
    std::shared_ptr<ir::Type> result_type;

    bool operator==(const ENode& other) const;
    bool operator!=(const ENode& other) const { return !(*this == other); }
};

// Hash functor for ENode (used by EGraph's hash-cons map).
struct ENodeHash {
    size_t operator()(const ENode& n) const noexcept;
};

// ── E-graph ──────────────────────────────────────────────────────────────
// A minimal e-graph: hash-consed e-nodes + union-find + congruence-closure
// rebuild. Each e-class is a set of equivalent e-nodes; `merge(a, b)`
// declares two e-classes equivalent and triggers a bounded rebuild that
// restores the congruence-closure invariant (parents that become
// structurally congruent are merged).
class EGraph {
public:
    EGraph();

    // Add an e-node. If a structurally identical e-node already exists,
    // returns the existing e-class id. Otherwise, creates a new e-class.
    // Children are canonicalised (find() is applied) before hash-consing.
    EClassId add(const ENode& node);

    // Merge two e-classes (declare them equivalent). Triggers a bounded
    // rebuild that restores the congruence-closure invariant.
    void merge(EClassId a, EClassId b);

    // Find the canonical e-class id for a given e-class id.
    EClassId find(EClassId id) const;

    // An e-class: an id, a list of equivalent e-nodes, and the e-classes
    // that reference this one (parents).
    struct EClass {
        EClassId id;
        std::vector<ENode> nodes;
        std::vector<EClassId> parents;
    };

    const std::vector<EClass>& classes() const { return classes_; }
    size_t num_classes() const { return classes_.size(); }

    // Bounded rebuild limit (per rebuild() invocation) — prevents
    // runaway rebuild loops. The brief mandates 1000.
    static constexpr size_t kMaxRebuildMerges = 1000;

private:
    std::vector<EClass> classes_;
    // Union-find parent for each e-class id. Mutable because find() is
    // const but performs path compression.
    mutable std::vector<EClassId> parent_;
    // Hash-cons index: canonical ENode -> EClassId, for deduplication.
    std::unordered_map<ENode, EClassId, ENodeHash> hash_cons_;
    // Re-entrancy guard: merge() called inside rebuild() must not
    // recursively trigger rebuild().
    bool in_rebuild_ = false;

    // Restore congruence closure: canonicalise children, re-hash-cons,
    // merge congruent parents. Bounded by kMaxRebuildMerges per call.
    void rebuild();

    // Union-find link (no rebuild trigger). Used by rebuild() itself.
    void link(EClassId a, EClassId b);
};

// ── IR-to-e-graph lowering ───────────────────────────────────────────────
struct LoweringResult {
    std::unique_ptr<EGraph> egraph;
    // SSA value name (instruction result or function arg) -> e-class id.
    std::unordered_map<std::string, EClassId> value_to_class;
    // E-class id of the function's single ret value, or nullopt if the
    // function has no single ret value (void return, multiple rets, etc.).
    std::optional<EClassId> return_class;
    // Non-owning pointer to the source function (used by extract() to
    // re-materialise IR with the same signature / block structure).
    const ir::Function* source_function = nullptr;
};

// Lower an ir::Function into an e-graph. Each non-terminator instruction
// becomes an e-node (terminators — Ret/Br/etc. — are kept as IR; their
// value operands are resolved via the value_to_class map at extract time).
// Each ConstantInt operand becomes a constant leaf e-node; each function
// argument becomes an argument leaf e-node.
LoweringResult lower_to_egraph(const ir::Function& fn);

// ── E-graph rewriter ─────────────────────────────────────────────────────
class EgraphRewriter {
public:
    EgraphRewriter(const pattern::PatternLibrary* lib,
                   const pattern::ArchDescriptor& arch);

    // Take ownership of a pre-built e-graph (typically the egraph field
    // of a LoweringResult returned by lower_to_egraph). The other fields
    // of the LoweringResult (value_to_class, return_class, source_function)
    // remain valid for the subsequent extract() call.
    void take_egraph(std::unique_ptr<EGraph> eg) { egraph_ = std::move(eg); }

    // Add rewrite rules from the pattern library. Each pattern's
    // source_function -> replacement_function becomes a rewrite rule:
    // for every e-class that matches source_function's structure, the
    // replacement_function's structure is added as an alternative and
    // merged. Returns the number of rules added.
    size_t add_pattern_rules();

    // Run equality saturation: apply all rules until fixpoint or
    // `max_iterations` reached. Returns the number of rewrites applied.
    size_t saturate(size_t max_iterations = 30);

    // Extract the cheapest function from the saturated e-graph, using
    // the evaluator's cost model. Walks the source function's block /
    // instruction structure and emits fresh IR, picking the cheapest
    // e-node per e-class (recursively). Returns a new ir::Function.
    std::shared_ptr<ir::Function> extract(
        const LoweringResult& lowering,
        const evaluator::EvaluationEngine& eval) const;

    // Accessors for testing / inspection.
    const EGraph& egraph() const { return *egraph_; }
    size_t rewrite_count() const { return rewrites_applied_; }

private:
    const pattern::PatternLibrary* lib_;
    const pattern::ArchDescriptor arch_;
    std::unique_ptr<EGraph> egraph_;
    size_t rewrites_applied_ = 0;
};

} // namespace clunk::egraph

// ── Public driver API ────────────────────────────────────────────────────
namespace clunk::search {

// Run equality saturation on `fn` with `lib`'s rewrites, extract the
// cheapest candidate, and return it as a search::Candidate (sound=false:
// the e-graph rewrites are pattern-library-sourced and need SMT
// verification). Returns nullopt if no improvement was found.
std::optional<Candidate> egraph_rewrite(
    const ir::Function& fn,
    const pattern::PatternLibrary& lib,
    const pattern::ArchDescriptor& arch,
    const evaluator::EvaluationEngine& eval);

} // namespace clunk::search
