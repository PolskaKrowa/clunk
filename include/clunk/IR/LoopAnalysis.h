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
 * Clunk LoopAnalysis — natural-loop detection over the block CFG.
 *
 * Finds natural loops the classic way: compute dominators (iterative
 * dataflow — CFGs here are small), find back-edges (u -> h where h
 * dominates u), and grow each loop body backwards from the latch until
 * the header. Loops sharing a header are merged.
 *
 * This is the substrate for LICM and constant-trip unrolling (see
 * Search/LoopOpt.h) and is intentionally minimal: no loop nesting tree,
 * no LCSSA — callers get block sets, the header, latches, and the
 * preheader (the unique out-of-loop predecessor of the header, "" when
 * there isn't exactly one).
 */
#include <string>
#include <unordered_set>
#include <vector>

#include "clunk/IR/Function.h"

namespace clunk::ir {

struct NaturalLoop {
    std::string header;
    std::unordered_set<std::string> blocks;  // every block in the loop (incl. header)
    std::vector<std::string> latches;        // in-loop predecessors of the header
    std::string preheader;                   // unique out-of-loop pred of header ("" if not unique)

    bool is_single_block() const { return blocks.size() == 1; }
    bool contains(const std::string& bb) const { return blocks.count(bb) != 0; }
};

// All natural loops of `fn` (headers unique — loops with a shared header
// are merged). Unreachable blocks are ignored. Returns an empty vector
// for irreducible or loop-free CFGs (irreducible cycles have no
// dominating header, so no back-edge is found for them — sound: callers
// simply see "no loop" and leave the function alone).
std::vector<NaturalLoop> find_natural_loops(const Function& fn);

// True iff the CFG has any cycle (cheap DFS; used as a bail-out by
// passes that must not run on loops at all).
bool has_back_edge(const Function& fn);

} // namespace clunk::ir
