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
 * Clunk MemOpt — alias-aware memory optimisations.
 *
 * The SMT verifier (soundly) refuses functions with memory operations, so
 * — like LoopOpt — every rewrite here is exact by construction, justified
 * by a deliberately conservative alias oracle:
 *
 *   AliasOracle::alias(p, q) returns NoAlias / MustAlias / MayAlias with
 *   MayAlias as the default answer. NoAlias is only claimed for:
 *     - two distinct allocas (fresh stack memory never overlaps);
 *     - an alloca vs. a function argument or global (a caller-supplied
 *       pointer cannot address stack memory that does not exist yet);
 *     - an alloca that never escapes vs. anything not derived from it;
 *     - two distinct `noalias` arguments (the attribute's contract);
 *     - same-base constant-index GEPs with the same source element type
 *       and index count that differ in some index (distinct elements).
 *   MustAlias is only claimed for the same SSA pointer value, or two
 *   same-base GEPs with identical constant index tuples.
 *
 * On that oracle, MemOptimizer runs three BLOCK-LOCAL rewrites (analysis
 * never crosses a block boundary; substitution of a forwarded value is
 * function-wide, which is sound because SSA dominance already guaranteed
 * every use is reachable only through the definition):
 *
 *   1. Store-to-load forwarding:  store v, p ; ... ; %x = load p  →  v
 *      (no intervening may-alias store / call / fence)
 *   2. Redundant load elimination: %a = load p ; ... ; %b = load p → %a
 *   3. Dead store elimination:    store v0, p ; ... ; store v1, p
 *      (first store removed when nothing may read between and the value
 *       types match)
 *
 * Volatile accesses are never touched and clobber everything. Calls and
 * fences clobber everything (clunk has no mod/ref summaries).
 */
#include <memory>
#include <string>
#include <vector>

#include "clunk/IR/Function.h"

namespace clunk::search {

// ── Alias oracle ────────────────────────────────────────────────────────────
enum class AliasResult { NoAlias, MayAlias, MustAlias };

class AliasOracle {
public:
    explicit AliasOracle(const ir::Function& fn);

    AliasResult alias(const std::shared_ptr<ir::Value>& p,
                      const std::shared_ptr<ir::Value>& q) const;

private:
    struct PointerInfo {
        enum class RootKind { Alloca, Argument, Global, Unknown } kind =
            RootKind::Unknown;
        std::string root_name;            // "" when the root is unnamed
        bool root_noalias = false;        // noalias argument
        bool root_escapes = true;         // allocas: address escapes?
        bool const_gep = true;            // every index a ConstantInt
        std::string gep_source_type;      // operand-0 pointee type text
        std::vector<int64_t> indices;     // flattened const GEP indices
    };
    PointerInfo classify(const std::shared_ptr<ir::Value>& p) const;

    const ir::Function& fn_;
    // Names of allocas whose address never escapes the function.
    std::vector<std::string> nonescaping_allocas_;
};

// ── Memory optimiser ────────────────────────────────────────────────────────
class MemOptimizer {
public:
    // Returns the rewritten function, or nullptr if nothing changed.
    std::shared_ptr<ir::Function> optimize(const ir::Function& fn);

    struct Stats {
        size_t loads_forwarded = 0;   // store-to-load forwarding
        size_t loads_eliminated = 0;  // redundant load elimination
        size_t stores_eliminated = 0; // dead store elimination
    };
    const Stats& stats() const { return stats_; }

private:
    Stats stats_{};
};

} // namespace clunk::search
