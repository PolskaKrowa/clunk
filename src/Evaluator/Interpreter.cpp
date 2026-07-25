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
 * Clunk Interpreter — implementation.
 *
 * See Interpreter.h for scope. This is a deliberately small AST-walking
 * interpreter; correctness of edge cases (overflow semantics of signed
 * division, etc.) follows C++ signed/unsigned integer rules applied to
 * the IR opcodes.
 */
#include "clunk/Evaluator/Interpreter.h"

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "clunk/IR/BasicBlock.h"
#include "clunk/IR/Instruction.h"
#include "clunk/IR/Type.h"
#include "clunk/IR/Value.h"

namespace clunk::evaluator {

namespace {

using ir::BasicBlock;
using ir::Function;
using ir::Instruction;
using ir::Opcode;
using ir::Type;
using ir::Value;

// Mask off the low `bits` of v, returning a value in [0, 2^bits).
// Used to model unsigned arithmetic on a sub-word type.
uint64_t mask_unsigned(uint64_t v, unsigned bits) {
    if (bits == 0 || bits >= 64) return v;
    return v & ((uint64_t{1} << bits) - 1);
}

// Sign-extend the low `bits` of v to a 64-bit signed integer.
int64_t sign_extend(uint64_t v, unsigned bits) {
    if (bits == 0 || bits >= 64) return static_cast<int64_t>(v);
    uint64_t m = uint64_t{1} << (bits - 1);
    if (v & m) {
        // Fill high bits with 1.
        v |= ~((uint64_t{1} << bits) - 1);
    }
    return static_cast<int64_t>(v);
}

// Resolve the bit-width of a type if it is an integer type.
std::optional<unsigned> integer_bit_width(const Type& t) {
    if (t.is_integer()) {
        return static_cast<unsigned>(t.bit_width());
    }
    return std::nullopt;
}

// Runtime context for a single interpret() call.
class Context {
public:
    Context(const Function& fn, const std::vector<int64_t>& args)
        : fn_(fn), args_(args), next_handle_(1) {}

    // Bind a Value to a runtime integer.
    void bind(const Value& v, int64_t val) {
        values_[key(v)] = val;
    }

    // Lookup a Value's runtime integer. Returns nullopt if the value
    // has not been bound (e.g. unsupported type).
    std::optional<int64_t> lookup(const Value& v) const {
        auto it = values_.find(key(v));
        if (it == values_.end()) return std::nullopt;
        return it->second;
    }

    // Allocate a fresh memory handle. The handle is a positive int64_t;
    // load from an uninitialised handle returns 0 (matching undef-ish
    // semantics, which is fine for a sanity oracle).
    int64_t alloc_handle() { return next_handle_++; }

    void store(int64_t handle, int64_t val) { memory_[handle] = val; }

    int64_t load(int64_t handle) const {
        auto it = memory_.find(handle);
        return it == memory_.end() ? 0 : it->second;
    }

    const Function& fn() const { return fn_; }
    const std::vector<int64_t>& args() const { return args_; }

private:
    // Key by raw pointer identity: SSA values are shared_ptr-managed
    // and unique within a function.
    uintptr_t key(const Value& v) const {
        return reinterpret_cast<uintptr_t>(&v);
    }

    const Function& fn_;
    const std::vector<int64_t>& args_;
    int64_t next_handle_;
    std::unordered_map<uintptr_t, int64_t> values_;
    std::unordered_map<int64_t, int64_t> memory_;
};

// Resolve a Value to an int64_t, handling ConstantInt and bound values.
// Returns nullopt on unsupported value kinds.
std::optional<int64_t> resolve(Context& ctx, const Value& v) {
    if (auto ci = dynamic_cast<const ir::ConstantInt*>(&v)) {
        return ci->value();
    }
    return ctx.lookup(v);
}

// Apply a binary opcode to two resolved int64_t operands, masking the
// result to the instruction's result-type width. Returns nullopt if
// the opcode is unsupported.
std::optional<int64_t> apply_binop(Opcode op, int64_t lhs, int64_t rhs,
                                    const Type& result_type) {
    auto bits_opt = integer_bit_width(result_type);
    unsigned bits = bits_opt.value_or(64);

    uint64_t ul = static_cast<uint64_t>(lhs);
    uint64_t ur = static_cast<uint64_t>(rhs);
    int64_t  sl = lhs;
    int64_t  sr = rhs;

    uint64_t ures = 0;
    int64_t  sres = 0;
    bool ok = true;
    bool use_signed = false;

    switch (op) {
        case Opcode::Add:  ures = ul + ur;                       break;
        case Opcode::Sub:  ures = ul - ur;                       break;
        case Opcode::Mul:  ures = ul * ur;                       break;
        case Opcode::And:  ures = ul & ur;                       break;
        case Opcode::Or:   ures = ul | ur;                       break;
        case Opcode::Xor:  ures = ul ^ ur;                       break;
        case Opcode::Shl:  ures = ul << (ur & 63);               break;
        case Opcode::LShr: ures = ul >> (ur & 63);               break;
        case Opcode::AShr:
            sres = static_cast<int64_t>(ul) >> (ur & 63);
            use_signed = true;
            break;
        case Opcode::UDiv:
            if (ur == 0) return std::nullopt; // trap-like: refuse
            ures = ul / ur;
            break;
        case Opcode::URem:
            if (ur == 0) return std::nullopt;
            ures = ul % ur;
            break;
        case Opcode::SDiv:
            if (sr == 0 || (sl == INT64_MIN && sr == -1)) return std::nullopt;
            sres = sl / sr;
            use_signed = true;
            break;
        case Opcode::SRem:
            if (sr == 0 || (sl == INT64_MIN && sr == -1)) return std::nullopt;
            sres = sl % sr;
            use_signed = true;
            break;
        default:
            ok = false;
            break;
    }
    if (!ok) return std::nullopt;

    // Store every result in a single canonical representation: the bwN value
    // sign-extended to int64. Sign-extending keeps the two's-complement bit
    // pattern while making the int64 sign match the bwN sign, so signed and
    // unsigned consumers agree.
    if (use_signed) {
        return sign_extend(static_cast<uint64_t>(sres), bits);
    } else {
        return sign_extend(ures, bits);
    }
}

// Apply an ICMP predicate. The result is 0 or 1 (i1).
std::optional<int64_t> apply_icmp(const Instruction& inst,
                                   int64_t lhs, int64_t rhs) {
    auto it = inst.metadata().find("pred");
    if (it == inst.metadata().end()) return std::nullopt;
    // Parse predicate number (see CmpPredicate in Instruction.h).
    int pred = 0;
    try { pred = std::stoi(it->second); } catch (...) { return std::nullopt; }

    // Values are stored sign-extended to int64 (see apply_binop). Signed
    // predicates compare the int64s directly; unsigned predicates must first
    // mask each operand to its true bit-width, otherwise a sign-extended
    // negative (0xFFFF...) would compare as a huge 64-bit unsigned instead of
    // the intended bwN unsigned value.
    unsigned bits = 64;
    if (inst.num_operands() >= 1 && inst.operand(0) && inst.operand(0)->type() &&
        inst.operand(0)->type()->is_integer()) {
        bits = static_cast<unsigned>(inst.operand(0)->type()->bit_width());
    }
    uint64_t ul = mask_unsigned(static_cast<uint64_t>(lhs), bits);
    uint64_t ur = mask_unsigned(static_cast<uint64_t>(rhs), bits);
    int64_t  sl = lhs;
    int64_t  sr = rhs;
    bool result = false;

    using P = ir::CmpPredicate;
    switch (static_cast<P>(pred)) {
        case P::EQ:  result = (sl == sr); break;
        case P::NE:  result = (sl != sr); break;
        case P::UGT: result = (ul >  ur); break;
        case P::UGE: result = (ul >= ur); break;
        case P::ULT: result = (ul <  ur); break;
        case P::ULE: result = (ul <= ur); break;
        case P::SGT: result = (sl >  sr); break;
        case P::SGE: result = (sl >= sr); break;
        case P::SLT: result = (sl <  sr); break;
        case P::SLE: result = (sl <= sr); break;
        default: return std::nullopt; // FP predicates unsupported
    }
    return result ? 1 : 0;
}

// Execute a single instruction. Returns:
//   - nullopt + no control-flow change: instruction had no observable
//     effect (e.g. a non-terminator we couldn't model) — caller skips it.
//   - nullopt + control-flow set: a terminator was executed; caller
//     should jump.
//   - a value: the result of the instruction was bound in ctx.
struct ExecResult {
    bool ok = true;           // false = unsupported / fatal
    bool is_ret = false;
    int64_t ret_value = 0;
    bool is_br = false;
    std::string next_block;   // empty if not a br
};

ExecResult execute(Context& ctx, const Instruction& inst,
                   const std::string& prev_block) {
    ExecResult r;
    Opcode op = inst.opcode();

    switch (op) {
        // ── Terminators ───────────────────────────────────────────────
        case Opcode::Ret: {
            if (inst.num_operands() == 0) {
                r.is_ret = true;
                r.ret_value = 0;
                return r;
            }
            auto v = resolve(ctx, *inst.operand(0));
            if (!v) { r.ok = false; return r; }
            r.is_ret = true;
            r.ret_value = *v;
            return r;
        }
        case Opcode::Br: {
            // Conditional: operand(0) is the condition; metadata has
            // true_bb / false_bb.
            if (inst.num_operands() > 0) {
                auto cond_v = resolve(ctx, *inst.operand(0));
                if (!cond_v) { r.ok = false; return r; }
                bool taken = (*cond_v != 0);
                auto it_true  = inst.metadata().find("true_bb");
                auto it_false = inst.metadata().find("false_bb");
                if (it_true == inst.metadata().end() ||
                    it_false == inst.metadata().end()) {
                    r.ok = false; return r;
                }
                r.is_br = true;
                r.next_block = taken ? it_true->second : it_false->second;
                return r;
            }
            // Unconditional: metadata dest_bb.
            auto it = inst.metadata().find("dest_bb");
            if (it == inst.metadata().end()) { r.ok = false; return r; }
            r.is_br = true;
            r.next_block = it->second;
            return r;
        }
        case Opcode::Unreachable:
            r.ok = false; // trap
            return r;

        // ── Memory ────────────────────────────────────────────────────
        case Opcode::Alloca: {
            // Returns a fresh memory handle. The type is a pointer.
            int64_t handle = ctx.alloc_handle();
            ctx.bind(inst, handle);
            return r;
        }
        case Opcode::Load: {
            if (inst.num_operands() < 1) { r.ok = false; return r; }
            auto ptr_v = resolve(ctx, *inst.operand(0));
            if (!ptr_v) { r.ok = false; return r; }
            int64_t v = ctx.load(*ptr_v);
            // Mask to loaded type width.
            auto bits = integer_bit_width(*inst.type());
            if (bits && *bits < 64) {
                v = sign_extend(static_cast<uint64_t>(v), *bits);
            }
            ctx.bind(inst, v);
            return r;
        }
        case Opcode::Store: {
            if (inst.num_operands() < 2) { r.ok = false; return r; }
            auto val_v  = resolve(ctx, *inst.operand(0));
            auto ptr_v  = resolve(ctx, *inst.operand(1));
            if (!val_v || !ptr_v) { r.ok = false; return r; }
            ctx.store(*ptr_v, *val_v);
            return r;
        }

        // ── Integer binary ops ────────────────────────────────────────
        case Opcode::Add: case Opcode::Sub: case Opcode::Mul:
        case Opcode::UDiv: case Opcode::SDiv:
        case Opcode::URem: case Opcode::SRem:
        case Opcode::And: case Opcode::Or:  case Opcode::Xor:
        case Opcode::Shl: case Opcode::LShr: case Opcode::AShr: {
            if (inst.num_operands() < 2) { r.ok = false; return r; }
            auto lhs = resolve(ctx, *inst.operand(0));
            auto rhs = resolve(ctx, *inst.operand(1));
            if (!lhs || !rhs) { r.ok = false; return r; }
            if (!inst.type()) { r.ok = false; return r; }
            auto v = apply_binop(op, *lhs, *rhs, *inst.type());
            if (!v) { r.ok = false; return r; }
            ctx.bind(inst, *v);
            return r;
        }

        // ── Compare ───────────────────────────────────────────────────
        case Opcode::ICmp: {
            if (inst.num_operands() < 2) { r.ok = false; return r; }
            auto lhs = resolve(ctx, *inst.operand(0));
            auto rhs = resolve(ctx, *inst.operand(1));
            if (!lhs || !rhs) { r.ok = false; return r; }
            auto v = apply_icmp(inst, *lhs, *rhs);
            if (!v) { r.ok = false; return r; }
            ctx.bind(inst, *v);
            return r;
        }

        // ── Select ────────────────────────────────────────────────────
        case Opcode::Select: {
            if (inst.num_operands() < 3) { r.ok = false; return r; }
            auto cond = resolve(ctx, *inst.operand(0));
            auto tv   = resolve(ctx, *inst.operand(1));
            auto fv   = resolve(ctx, *inst.operand(2));
            if (!cond || !tv || !fv) { r.ok = false; return r; }
            ctx.bind(inst, *cond != 0 ? *tv : *fv);
            return r;
        }

        // ── Phi ───────────────────────────────────────────────────────
        case Opcode::Phi: {
            // Phi operands alternate (value, block-name-as-ConstantInt?).
            // In this IR, phi incoming blocks are stored in metadata
            // "phi_blocks" as a comma-separated list, with operands
            // in the same order. We pick the operand whose block matches
            // the previously-executed block.
            auto it = inst.metadata().find("phi_blocks");
            if (it == inst.metadata().end()) { r.ok = false; return r; }
            // Simple split on comma.
            std::vector<std::string> blocks;
            {
                std::string cur;
                for (char c : it->second) {
                    if (c == ',') { blocks.push_back(cur); cur.clear(); }
                    else cur.push_back(c);
                }
                if (!cur.empty()) blocks.push_back(cur);
            }
            for (size_t i = 0; i < blocks.size() && i < inst.num_operands(); ++i) {
                if (blocks[i] == prev_block) {
                    auto v = resolve(ctx, *inst.operand(i));
                    if (!v) { r.ok = false; return r; }
                    ctx.bind(inst, *v);
                    return r;
                }
            }
            // No matching incoming block — pick operand 0 as a fallback.
            if (inst.num_operands() >= 1) {
                auto v = resolve(ctx, *inst.operand(0));
                if (!v) { r.ok = false; return r; }
                ctx.bind(inst, *v);
                return r;
            }
            r.ok = false;
            return r;
        }

        // ── Casts (integer-only) ──────────────────────────────────────
        case Opcode::Trunc: case Opcode::ZExt: case Opcode::SExt:
        case Opcode::BitCast: case Opcode::PtrToInt: case Opcode::IntToPtr: {
            if (inst.num_operands() < 1) { r.ok = false; return r; }
            auto v = resolve(ctx, *inst.operand(0));
            if (!v) { r.ok = false; return r; }
            auto bits = integer_bit_width(*inst.type());
            if (!bits) { r.ok = false; return r; }
            if (op == Opcode::Trunc) {
                // Keep the low dest-width bits, then SIGN-extend so the
                // stored value follows the interpreter's canonical
                // (sign-extended) representation — a zero-extended store
                // would make a later signed icmp on the result compare the
                // wrong value (e.g. trunc i32 255 to i8 must behave as -1).
                ctx.bind(inst, sign_extend(
                    mask_unsigned(static_cast<uint64_t>(*v), *bits), *bits));
            } else if (op == Opcode::ZExt) {
                // ZExt: mask the canonical (sign-extended) operand to its
                // source width. The result is < 2^src_bits, so it is
                // already canonical at the wider destination.
                auto src_ty = inst.operand(0)->type();
                auto src_bits = src_ty ? integer_bit_width(*src_ty)
                                       : std::nullopt;
                if (!src_bits || *src_bits >= *bits) { r.ok = false; return r; }
                ctx.bind(inst, static_cast<int64_t>(
                    mask_unsigned(static_cast<uint64_t>(*v), *src_bits)));
            } else { // SExt, BitCast, PtrToInt, IntToPtr
                // The canonical representation is already sign-extended, so
                // sign-extension to a wider type is the identity on it.
                ctx.bind(inst, *v);
            }
            return r;
        }

        default:
            // Unsupported opcode (floats, calls, vector, etc.).
            r.ok = false;
            return r;
    }
}

} // namespace

std::optional<int64_t> Interpreter::interpret(
    const ir::Function& fn,
    const std::vector<int64_t>& args) {

    if (args.size() != fn.argument_count()) return std::nullopt;
    if (fn.blocks().empty()) return std::nullopt;

    // Refuse to interpret non-integer return types.
    auto ret_ty = fn.return_type();
    if (ret_ty && !ret_ty->is_integer() && !ret_ty->is_void()) {
        return std::nullopt;
    }

    Context ctx(fn, args);

    // Bind arguments.
    const auto& fn_args = fn.arguments();
    for (size_t i = 0; i < fn_args.size() && i < args.size(); ++i) {
        // Each Argument is a struct with a name; we cannot easily build
        // a Value& from it, but operands that reference arguments will
        // be Value instances with the same name. We resolve arguments
        // lazily via name lookup below.
        (void)fn_args[i];
    }
    // Cache argument values by name so that operand-resolution can find them.
    // We do this by giving each argument a synthetic "binding" the first
    // time we encounter it as an operand. Since Argument is not a Value,
    // we instead intercept named operands below.

    // Helper: try to resolve `v` as an argument by name.
    auto resolve_argument = [&](const ir::Value& v) -> std::optional<int64_t> {
        if (!v.has_name()) return std::nullopt;
        // Strip leading "%" if any (this IR stores names without %).
        const std::string& nm = v.name();
        // Match against fn.arguments().
        const auto& fa = fn.arguments();
        for (size_t i = 0; i < fa.size(); ++i) {
            if (fa[i].name == nm) {
                return args[i];
            }
        }
        return std::nullopt;
    };

    // Override resolve to also try argument-by-name. We do this by
    // pre-binding any named value we encounter: before resolving an
    // operand, if the operand is not a ConstantInt and not yet bound,
    // check if it's an argument and bind it. (No separate
    // `resolve_full` lambda is needed — the loop below pre-binds
    // argument-named operands before executing the instruction.)

    // Block-walk: maintain a (current_block, prev_block) pair so that
    // phi nodes can resolve their incoming value for the actual
    // control-flow path the interpreter took. Hops are bounded by
    // kMaxBlockHops to prevent runaway loops.
    std::string current_block_name = fn.blocks().front()->name();
    std::string prev_block_name;
    size_t hops = 0;

    while (hops < kMaxBlockHops) {
        ++hops;
        auto bb = fn.block(current_block_name);
        if (!bb) return std::nullopt;

        bool branched = false;
        for (auto& inst : bb->instructions()) {
            // Pre-bind any operands that are arguments.
            for (auto& op : inst->operands()) {
                if (op && op->has_name() && !dynamic_cast<const ir::ConstantInt*>(op.get())) {
                    if (!ctx.lookup(*op)) {
                        auto av = resolve_argument(*op);
                        if (av) ctx.bind(*op, *av);
                    }
                }
            }

            ExecResult er = execute(ctx, *inst, prev_block_name);
            if (!er.ok) return std::nullopt;
            if (er.is_ret) return er.ret_value;
            if (er.is_br) {
                prev_block_name = current_block_name;
                current_block_name = er.next_block;
                branched = true;
                break; // continue to next block
            }
            // Non-terminator executed — fall through to next instruction.
        }
        // A block that ended without a taken branch or a ret is malformed.
        // Use an explicit flag — a single-block loop's back-edge legitimately
        // makes current/prev names equal, and a name-comparison check would
        // reject every such loop.
        if (!branched) {
            return std::nullopt;
        }
    }
    // Exceeded hop budget.
    return std::nullopt;
}

} // namespace clunk::evaluator
