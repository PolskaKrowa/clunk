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
 * Clunk EvaluationCache — implementation of the structural-hash function.
 *
 * The LRU cache itself is fully templated and lives in the header; this
 * translation unit only contains the structural_hash() free function.
 */
#include "clunk/Evaluator/EvaluationCache.h"

#include <cstdint>
#include <sstream>

#include "clunk/IR/Instruction.h"
#include "clunk/IR/Type.h"

namespace clunk::evaluator {

namespace {

// 64-bit FNV-1a constants.
constexpr uint64_t kFNVOffset = 0xcbf29ce484222325ULL;
constexpr uint64_t kFNVPrime  = 0x100000001b3ULL;

class Hasher {
public:
    Hasher() : h_(kFNVOffset) {}

    void mix(uint64_t v) {
        h_ ^= v;
        h_ *= kFNVPrime;
    }

    void mix_bytes(const std::string& s) {
        for (unsigned char c : s) {
            h_ ^= static_cast<uint64_t>(c);
            h_ *= kFNVPrime;
        }
        // Terminator to distinguish "ab" from "a" + "b".
        h_ ^= 0xFF;
        h_ *= kFNVPrime;
    }

    uint64_t value() const { return h_; }

private:
    uint64_t h_;
};

uint64_t type_fingerprint(const ir::Type& t) {
    // Encode the type as its to_string() — types are interned via
    // TypeContext so identical types share pointer identity, but
    // to_string() is the canonical structural description and works
    // even across TypeContexts.
    Hasher h;
    h.mix_bytes(t.to_string());
    return h.value();
}

} // namespace

uint64_t structural_hash(const ir::Function& fn) {
    Hasher h;

    // Function-level fingerprint.
    h.mix_bytes(fn.name());
    if (auto ft = fn.function_type()) {
        h.mix_bytes(ft->to_string());
    }
    h.mix(fn.argument_count());
    h.mix(fn.blocks().size());

    // Walk every block / instruction / operand-type in order.
    for (auto& block : fn.blocks()) {
        h.mix_bytes(block->name());
        h.mix(block->instructions().size());

        for (auto& inst : block->instructions()) {
            h.mix(static_cast<uint64_t>(inst->opcode()));
            h.mix(inst->num_operands());
            if (inst->type()) {
                h.mix(type_fingerprint(*inst->type()));
            }
            // Operand *types* (not names) — semantically equivalent
            // renames hit the same cache entry.
            for (auto& op : inst->operands()) {
                if (op && op->type()) {
                    h.mix(type_fingerprint(*op->type()));
                } else {
                    h.mix(0);
                }
            }
            // Mix in the metadata keys (not values) — branches carry
            // successor-block names there, which affects control flow.
            for (auto& [key, val] : inst->metadata()) {
                h.mix_bytes(key);
                h.mix_bytes(val);
            }
        }
    }
    return h.value();
}

} // namespace clunk::evaluator
