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
 * Clunk DataflowPrune — dataflow-based search-space pruning.
 *
 * The search/rewrite phases (stochastic, evolutionary, e-graph, pattern
 * library, ...) routinely leave behind dead instructions and, once branch
 * conditions get folded to constants, unreachable blocks: an algebraic
 * identity turns `x*0` into `0` and leaves the multiply's now-unused
 * operand computation behind; a mutation replaces an operand and orphans
 * whatever used to feed it; equality saturation extracts a cheaper
 * expression but the discarded alternative's instructions are still
 * sitting in the block.
 *
 * None of this is WRONG — every phase already only emits semantics-
 * preserving rewrites — but left unpruned it costs real search quality:
 * every later round re-lowers it into the e-graph, re-mutates around it,
 * and re-measures it (MCA/interpreter cost scales with instruction
 * count), all for code that provably cannot affect the function's
 * output. This header prunes it away, each round, before the next round's
 * phases ever see it — smaller baselines are cheaper to search AND
 * cheaper to evaluate.
 *
 * Every transform here is sound by construction (no SMT proof required,
 * same trust tier as LoopOpt/MemOpt — see Candidate::sound):
 *   - eliminate_dead_code: removes instructions whose result is neither
 *     used (transitively) nor has a side effect. Classic SSA mark-sweep
 *     over the def-use graph — no CFG traversal needed, so it's correct
 *     regardless of loop structure.
 *   - remove_unreachable_blocks: drops blocks unreachable from entry
 *     (CFG reachability via BasicBlock::successors()), fixing up any
 *     surviving Phi that listed a removed block as a predecessor.
 *   - simplify_known_bits: uses analysis::KnownBits (see
 *     clunk/Analysis/KnownBits.h) to fold instructions that are provably
 *     constant, drop AND-masks that provably change nothing, resolve
 *     ICmp results that are provably true/false, and turn a conditional
 *     Br with a provably-constant condition into an unconditional one —
 *     which is what feeds remove_unreachable_blocks its work.
 *   - simplify_cse: same-block common-subexpression elimination — two
 *     structurally identical pure instructions in one block always
 *     compute the same value, so every duplicate after the first is
 *     rewired to reuse the earlier result and dropped. Block-local only
 *     (same scope restriction as MemOpt's forwarding passes — this
 *     codebase has no dominator tree, and same-block scope needs none:
 *     everything earlier in a block always executes before everything
 *     later in the SAME block). This is the deterministic, exhaustive
 *     counterpart to StochasticSearch's opportunistic
 *     EliminateCommonSubexpr mutation, which only fires when the random
 *     search happens to propose that exact duplicate pair.
 *   - prune_dataflow: runs all four to a (small, bounded) fixed point —
 *     folding/deduplication can expose more dead code / more unreachable
 *     blocks, so a single pass often leaves easy wins on the table.
 */
#include <memory>

#include "clunk/IR/Function.h"

namespace clunk::ir {

struct PruneStats {
    size_t instructions_removed = 0;
    size_t blocks_removed = 0;
    size_t values_folded_constant = 0;
    size_t redundant_masks_removed = 0;
    size_t comparisons_folded = 0;
    size_t branches_simplified = 0;
    size_t subexpressions_eliminated = 0;

    bool changed() const {
        return instructions_removed || blocks_removed || values_folded_constant ||
               redundant_masks_removed || comparisons_folded || branches_simplified ||
               subexpressions_eliminated;
    }
};

// Remove instructions with no side effect whose result is not used,
// directly or transitively, by anything with a side effect or by the
// function's terminators. Returns nullptr if nothing was removable.
std::shared_ptr<Function> eliminate_dead_code(const Function& fn, PruneStats* stats = nullptr);

// Remove blocks unreachable from the entry block, fixing up surviving
// Phi nodes that named a removed block as a predecessor. Returns nullptr
// if every block is already reachable.
std::shared_ptr<Function> remove_unreachable_blocks(const Function& fn,
                                                     PruneStats* stats = nullptr);

// Use known-bits inference (analysis::KnownBits) to fold provably-constant
// values, drop no-op AND masks, resolve provably-true/false comparisons,
// and simplify a conditional Br whose condition is now a known constant to
// an unconditional one. Returns nullptr if nothing was simplified.
std::shared_ptr<Function> simplify_known_bits(const Function& fn, PruneStats* stats = nullptr);

// Same-block common-subexpression elimination: every instruction after
// the first in a block that is a structural duplicate (same opcode,
// flags, predicate, type, and operands — by resolved identity for named
// values, by value for constants) of an earlier instruction in the SAME
// block is replaced by that earlier result. Restricted to pure,
// deterministic, non-memory opcodes (no loads — those need alias
// analysis, see MemOpt). Returns nullptr if nothing was eliminated.
std::shared_ptr<Function> simplify_cse(const Function& fn, PruneStats* stats = nullptr);

// Runs simplify_known_bits → simplify_cse → remove_unreachable_blocks →
// eliminate_dead_code repeatedly (bounded iterations) until nothing
// changes. This is the entry point Pipeline uses. Returns nullptr if
// nothing changed at all.
std::shared_ptr<Function> prune_dataflow(const Function& fn, PruneStats* stats = nullptr);

} // namespace clunk::ir