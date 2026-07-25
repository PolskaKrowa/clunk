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
 * Clunk Interpreter — a small AST-walking interpreter for the clunk IR.
 *
 * Scope: a sanity oracle for tests, NOT a production execution engine.
 *   - Single function (no calls).
 *   - Integer types only (i1, i8, i16, i32, i64).
 *   - Operands resolve to int64_t internally; results are masked to
 *     the instruction's result-type bit width.
 *   - Supports: ConstantInt, function arguments, alloca/load/store
 *     (via a simple handle → value map), Add/Sub/Mul/UDiv/SDiv/URem/SRem/
 *     And/Or/Xor/Shl/LShr/AShr, ICmp, Select, Br (cond + uncond),
 *     Ret.
 *   - Phi nodes pick the incoming value from the *previously executed*
 *     block (best-effort; works for the typical test shapes).
 *   - Floats, calls, GEP into struct/array, vector ops, etc. are
 *     unsupported and return std::nullopt.
 *
 * The interpreter is intentionally side-effect-free beyond its own
 * internal state: it does not modify the IR, does not allocate OS
 * resources, and is safe to call concurrently on independent
 * ir::Function instances.
 */
#include <cstdint>
#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

#include "clunk/IR/Function.h"

namespace clunk::evaluator {

class Interpreter {
public:
    // Evaluate `fn` on the supplied integer argument list. Returns
    // std::nullopt if the function is unsupported (non-integer ops,
    // calls, infinite loop guard tripped, …) or the argument count
    // doesn't match `fn.argument_count()`.
    //
    // `args.size()` must equal `fn.argument_count()`.
    static std::optional<int64_t> interpret(
        const ir::Function& fn,
        const std::vector<int64_t>& args);

private:
    // Cap on dynamic block transitions to prevent runaway loops in
    // the oracle. Generous enough for all existing examples/simple_add.ll
    // and the test shapes; tight enough to fail fast on a broken input.
    static constexpr size_t kMaxBlockHops = 1024;
};

} // namespace clunk::evaluator
