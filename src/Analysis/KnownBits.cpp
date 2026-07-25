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
 * Clunk KnownBits — implementation.
 * See include/clunk/Analysis/KnownBits.h for the design/scope notes.
 */
#include "clunk/Analysis/KnownBits.h"

#include <algorithm>

namespace clunk::analysis {

namespace {

// Number of low bits provably zero (a cheap, sound proxy for "how
// divisible is this value" — used by Add/Sub/Mul instead of full carry
// propagation).
unsigned trailing_known_zeros(const KnownBits& kb) {
    if (kb.width == 0) return 0;
    unsigned n = 0;
    while (n < kb.width && (kb.zero & (uint64_t(1) << n))) ++n;
    return n;
}

// The comparison predicate stored in metadata "pred" (see
// Instruction.cpp's to_string / EgraphRewriter.cpp's predicate_from_inst —
// same convention, kept as an independent local copy like that one).
ir::CmpPredicate cmp_predicate(const ir::Instruction& inst) {
    auto it = inst.metadata().find("pred");
    if (it == inst.metadata().end()) return ir::CmpPredicate::EQ;
    try {
        return static_cast<ir::CmpPredicate>(
            static_cast<unsigned>(std::stoul(it->second)));
    } catch (...) {
        return ir::CmpPredicate::EQ;
    }
}

// Evaluate an integer comparison exactly, given two fully-known operands.
bool eval_icmp_exact(ir::CmpPredicate pred, const KnownBits& lhs, const KnownBits& rhs) {
    const uint64_t mask = KnownBits::mask_for_width(lhs.width);
    const uint64_t ul = lhs.one & mask, ur = rhs.one & mask;
    const int64_t sl = lhs.signed_value(), sr = rhs.signed_value();
    switch (pred) {
        case ir::CmpPredicate::EQ:  return ul == ur;
        case ir::CmpPredicate::NE:  return ul != ur;
        case ir::CmpPredicate::UGT: return ul > ur;
        case ir::CmpPredicate::UGE: return ul >= ur;
        case ir::CmpPredicate::ULT: return ul < ur;
        case ir::CmpPredicate::ULE: return ul <= ur;
        case ir::CmpPredicate::SGT: return sl > sr;
        case ir::CmpPredicate::SGE: return sl >= sr;
        case ir::CmpPredicate::SLT: return sl < sr;
        case ir::CmpPredicate::SLE: return sl <= sr;
        default: break;  // float predicates never reach here (ICmp only)
    }
    return false;
}

KnownBits known_bits_add_sub(const KnownBits& lhs, const KnownBits& rhs, bool is_sub) {
    if (lhs.is_fully_known() && rhs.is_fully_known()) {
        int64_t v = is_sub ? (lhs.signed_value() - rhs.signed_value())
                            : (lhs.signed_value() + rhs.signed_value());
        return KnownBits::exact(lhs.width, v);
    }
    // Sound-but-cheap: if the low k bits of both operands are known zero,
    // the low k bits of the sum/difference are known zero too (no carry
    // can reach them). No claim about any other bit.
    unsigned k = std::min(trailing_known_zeros(lhs), trailing_known_zeros(rhs));
    KnownBits kb = KnownBits::unknown(lhs.width);
    if (k > 0) kb.zero = KnownBits::mask_for_width(k);
    return kb;
}

KnownBits known_bits_mul(const KnownBits& lhs, const KnownBits& rhs) {
    if (lhs.is_fully_known() && rhs.is_fully_known()) {
        int64_t v = lhs.signed_value() * rhs.signed_value();
        return KnownBits::exact(lhs.width, v);
    }
    // Sound: a multiple of 2^a times a multiple of 2^b is a multiple of
    // 2^(a+b) — trailing zero counts add.
    unsigned k = std::min(lhs.width,
                          trailing_known_zeros(lhs) + trailing_known_zeros(rhs));
    KnownBits kb = KnownBits::unknown(lhs.width);
    if (k > 0) kb.zero = KnownBits::mask_for_width(k);
    return kb;
}

KnownBits known_bits_shl(const KnownBits& lhs, const KnownBits& rhs) {
    KnownBits kb = KnownBits::unknown(lhs.width);
    if (!rhs.is_fully_known()) return kb;
    int64_t sc = rhs.signed_value();
    if (sc < 0 || static_cast<uint64_t>(sc) >= lhs.width) return kb;  // poison/UB shift
    unsigned c = static_cast<unsigned>(sc);
    const uint64_t mask = KnownBits::mask_for_width(lhs.width);
    kb.width = lhs.width;
    kb.zero = ((lhs.zero << c) | KnownBits::mask_for_width(c)) & mask;
    kb.one = (lhs.one << c) & mask;
    return kb;
}

KnownBits known_bits_lshr(const KnownBits& lhs, const KnownBits& rhs) {
    KnownBits kb = KnownBits::unknown(lhs.width);
    if (!rhs.is_fully_known()) return kb;
    int64_t sc = rhs.signed_value();
    if (sc < 0 || static_cast<uint64_t>(sc) >= lhs.width) return kb;
    unsigned c = static_cast<unsigned>(sc);
    kb.width = lhs.width;
    uint64_t high_zero = KnownBits::mask_for_width(lhs.width) &
                          ~KnownBits::mask_for_width(lhs.width - c);
    kb.zero = (lhs.zero >> c) | high_zero;
    kb.one = lhs.one >> c;
    return kb;
}

KnownBits known_bits_ashr(const KnownBits& lhs, const KnownBits& rhs) {
    KnownBits kb = KnownBits::unknown(lhs.width);
    if (!rhs.is_fully_known()) return kb;
    int64_t sc = rhs.signed_value();
    if (sc < 0 || static_cast<uint64_t>(sc) >= lhs.width || lhs.width == 0) return kb;
    unsigned c = static_cast<unsigned>(sc);
    kb.width = lhs.width;
    const uint64_t sign_bit = uint64_t(1) << (lhs.width - 1);
    const uint64_t high_mask =
        KnownBits::mask_for_width(lhs.width) & ~KnownBits::mask_for_width(lhs.width - c);
    kb.zero = lhs.zero >> c;
    kb.one = lhs.one >> c;
    if (lhs.zero & sign_bit) {
        kb.zero |= high_mask;  // sign known 0: shifted-in bits are 0
    } else if (lhs.one & sign_bit) {
        kb.one |= high_mask;   // sign known 1: shifted-in bits are 1
    }
    // else: sign bit unknown, so the shifted-in high bits stay unknown too.
    return kb;
}

KnownBits known_bits_and(const KnownBits& lhs, const KnownBits& rhs) {
    KnownBits kb;
    kb.width = lhs.width;
    kb.zero = lhs.zero | rhs.zero;
    kb.one = lhs.one & rhs.one;
    return kb;
}

KnownBits known_bits_or(const KnownBits& lhs, const KnownBits& rhs) {
    KnownBits kb;
    kb.width = lhs.width;
    kb.zero = lhs.zero & rhs.zero;
    kb.one = lhs.one | rhs.one;
    return kb;
}

KnownBits known_bits_xor(const KnownBits& lhs, const KnownBits& rhs) {
    KnownBits kb;
    kb.width = lhs.width;
    kb.zero = (lhs.zero & rhs.zero) | (lhs.one & rhs.one);
    kb.one = (lhs.zero & rhs.one) | (lhs.one & rhs.zero);
    return kb;
}

KnownBits known_bits_trunc(const KnownBits& src, unsigned new_width) {
    KnownBits kb;
    kb.width = new_width;
    const uint64_t mask = KnownBits::mask_for_width(new_width);
    kb.zero = src.zero & mask;
    kb.one = src.one & mask;
    return kb;
}

KnownBits known_bits_zext(const KnownBits& src, unsigned new_width) {
    KnownBits kb;
    kb.width = new_width;
    kb.one = src.one & KnownBits::mask_for_width(src.width);
    const uint64_t new_mask = KnownBits::mask_for_width(new_width);
    const uint64_t old_mask = KnownBits::mask_for_width(src.width);
    kb.zero = (src.zero & old_mask) | (new_mask & ~old_mask);  // extended bits are 0
    return kb;
}

KnownBits known_bits_sext(const KnownBits& src, unsigned new_width) {
    KnownBits kb;
    kb.width = new_width;
    const uint64_t old_mask = KnownBits::mask_for_width(src.width);
    const uint64_t new_mask = KnownBits::mask_for_width(new_width);
    const uint64_t ext_bits = new_mask & ~old_mask;
    kb.zero = src.zero & old_mask;
    kb.one = src.one & old_mask;
    if (src.width == 0) return kb;
    const uint64_t sign_bit = uint64_t(1) << (src.width - 1);
    if (src.zero & sign_bit) {
        kb.zero |= ext_bits;
    } else if (src.one & sign_bit) {
        kb.one |= ext_bits;
    }
    return kb;
}

} // namespace

KnownBits operand_known_bits(const std::shared_ptr<ir::Value>& v,
                              const std::unordered_map<std::string, KnownBits>& env) {
    if (!v || !v->type() || !v->type()->is_integer()) {
        return KnownBits::unknown(v && v->type() ? v->type()->bit_width() : 0);
    }
    const unsigned w = static_cast<unsigned>(v->type()->bit_width());
    if (auto ci = std::dynamic_pointer_cast<ir::ConstantInt>(v)) {
        return KnownBits::exact(w, ci->value());
    }
    if (v->has_name()) {
        auto it = env.find(v->name());
        if (it != env.end()) return it->second;
    }
    return KnownBits::unknown(w);
}

KnownBits compute_known_bits(const ir::Instruction& inst,
                              const std::unordered_map<std::string, KnownBits>& env) {
    const ir::Type* ty = inst.type().get();
    const unsigned width = (ty && ty->is_integer()) ? static_cast<unsigned>(ty->bit_width()) : 0;
    if (width == 0 && inst.opcode() != ir::Opcode::ICmp) {
        return KnownBits::unknown(0);
    }

    auto opKB = [&](size_t i) -> KnownBits {
        if (i >= inst.num_operands()) return KnownBits::unknown(width);
        return operand_known_bits(inst.operand(i), env);
    };

    switch (inst.opcode()) {
        case ir::Opcode::Add:
            return known_bits_add_sub(opKB(0), opKB(1), /*is_sub=*/false);
        case ir::Opcode::Sub:
            return known_bits_add_sub(opKB(0), opKB(1), /*is_sub=*/true);
        case ir::Opcode::Mul:
            return known_bits_mul(opKB(0), opKB(1));
        case ir::Opcode::And:
            return known_bits_and(opKB(0), opKB(1));
        case ir::Opcode::Or:
            return known_bits_or(opKB(0), opKB(1));
        case ir::Opcode::Xor:
            return known_bits_xor(opKB(0), opKB(1));
        case ir::Opcode::Shl:
            return known_bits_shl(opKB(0), opKB(1));
        case ir::Opcode::LShr:
            return known_bits_lshr(opKB(0), opKB(1));
        case ir::Opcode::AShr:
            return known_bits_ashr(opKB(0), opKB(1));
        case ir::Opcode::Trunc:
            return known_bits_trunc(opKB(0), width);
        case ir::Opcode::ZExt:
            return known_bits_zext(opKB(0), width);
        case ir::Opcode::SExt:
            return known_bits_sext(opKB(0), width);
        case ir::Opcode::Select: {
            KnownBits cond = opKB(0);
            KnownBits a = opKB(1), b = opKB(2);
            if (cond.width >= 1 && (cond.one & 1)) return a;   // condition provably true
            if (cond.width >= 1 && (cond.zero & 1)) return b;  // condition provably false
            return KnownBits::meet(a, b);
        }
        case ir::Opcode::Phi: {
            // Meet of every incoming operand ALREADY in env (see header:
            // back-edge operands not yet computed contribute "unknown",
            // which collapses the meet to unknown for those bits — sound).
            if (inst.num_operands() == 0) return KnownBits::unknown(width);
            KnownBits acc = opKB(0);
            for (size_t i = 1; i < inst.num_operands(); ++i) {
                acc = KnownBits::meet(acc, opKB(i));
            }
            acc.width = width;
            return acc;
        }
        case ir::Opcode::ICmp: {
            if (inst.num_operands() < 2) return KnownBits::unknown(1);
            KnownBits lhs = opKB(0), rhs = opKB(1);
            ir::CmpPredicate pred = cmp_predicate(inst);
            if (lhs.is_fully_known() && rhs.is_fully_known()) {
                return KnownBits::exact(1, eval_icmp_exact(pred, lhs, rhs) ? 1 : 0);
            }
            // "Definitely differs" is decidable even without full
            // knowledge: any bit position known-1 in one operand and
            // known-0 in the other means the values cannot be equal.
            bool definitely_differs =
                ((lhs.one & rhs.zero) | (lhs.zero & rhs.one)) != 0;
            if (definitely_differs) {
                if (pred == ir::CmpPredicate::EQ) return KnownBits::exact(1, 0);
                if (pred == ir::CmpPredicate::NE) return KnownBits::exact(1, 1);
            }
            return KnownBits::unknown(1);
        }
        default:
            return KnownBits::unknown(width);
    }
}

std::unordered_map<std::string, KnownBits> analyse_known_bits(const ir::Function& fn) {
    std::unordered_map<std::string, KnownBits> env;
    for (auto& block : fn.blocks()) {
        if (!block) continue;
        for (auto& inst : block->instructions()) {
            if (!inst || !inst->has_name()) continue;
            if (inst->is_terminator()) continue;
            env[inst->name()] = compute_known_bits(*inst, env);
        }
    }
    return env;
}

std::optional<int64_t> known_bits_constant(const KnownBits& kb, const ir::Type* ty) {
    if (!ty || !ty->is_integer()) return std::nullopt;
    if (kb.has_conflict() || !kb.is_fully_known()) return std::nullopt;
    return kb.signed_value();
}

} // namespace clunk::analysis
