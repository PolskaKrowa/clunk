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
 * Clunk Scalarizer — lane-blasting lowering of vector functions.
 *
 * Lowers a function containing fixed-width integer vector operations to an
 * equivalent integer-only function the existing Interpreter and SMTVerifier
 * can handle unchanged:
 *
 *   - every `<N x iM>` argument becomes N iM arguments (lane order,
 *     names `<arg>.l0 .. <arg>.lN-1`);
 *   - vector binops become N scalar binops (per-lane, flags preserved —
 *     nsw/nuw poison is a lane-wise property in LLVM, so this is exact);
 *   - extractelement/insertelement with CONSTANT indices, and
 *     shufflevector with a constant non-undef mask, become pure lane
 *     re-wiring (no emitted instructions);
 *   - `call @clunk.vector.reduce.<op>.*(%v)` intrinsics (op ∈ add, mul,
 *     and, or, xor — the associative-commutative reductions the
 *     VectorSynth pass emits) become a balanced binop tree over the lanes;
 *   - select with a scalar i1 condition and vector operands becomes N
 *     scalar selects.
 *
 * Anything else involving vectors (vector phi, vector memory ops, vector
 * icmp, non-constant lane indices, undef lanes / undef shuffle masks,
 * unnamed vector intermediates) is REFUSED — the functions return nullptr
 * and the caller must treat the input as unverifiable (sound: refusing
 * only ever widens "Unknown", never blesses a wrong rewrite).
 *
 * Equivalence transport: scalarization is deterministic and depends only
 * on the function SIGNATURE for the argument expansion, so two functions
 * with the same signature scalarize against the same expanded argument
 * space. Proving scalarize(f) ≡ scalarize(g) over all expanded-argument
 * values therefore proves f ≡ g over all vector argument values, and vice
 * versa — each lane of each vector argument is universally quantified
 * either way.
 */
#include <memory>
#include <string>

#include "clunk/IR/Function.h"

namespace clunk::ir {

// True iff `fn` mentions a vector type anywhere (argument, return type,
// instruction result, operand) or calls a `clunk.vector.*` intrinsic.
bool function_has_vector_ops(const Function& fn);

// True iff `callee` is a reduction intrinsic this scalarizer understands
// (`clunk.vector.reduce.<add|mul|and|or|xor>[.suffix]`). `out_op` (if
// non-null) receives the lane-combining opcode.
bool parse_reduce_intrinsic(const std::string& callee, Opcode* out_op);

// Lower a SCALAR-RETURNING function. Returns nullptr if `fn` returns a
// vector (use scalarize_lane) or contains unsupported vector constructs.
std::shared_ptr<Function> scalarize_function(const Function& fn);

// Lower a VECTOR-RETURNING function to the scalar function computing lane
// `lane` of its result. Returns nullptr on unsupported constructs or if
// `lane` is out of range.
std::shared_ptr<Function> scalarize_lane(const Function& fn, size_t lane);

} // namespace clunk::ir
