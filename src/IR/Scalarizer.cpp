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
 * Clunk Scalarizer — lane-blasting lowering of vector functions.
 * See include/clunk/IR/Scalarizer.h for the contract.
 */
#include "clunk/IR/Scalarizer.h"

#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "clunk/IR/Instruction.h"
#include "clunk/IR/Value.h"

namespace clunk::ir {

namespace {

bool is_vector_type(const std::shared_ptr<Type>& t) {
    return t && t->is_vector();
}

// Integer binops that lane-blast directly.
bool is_lane_blastable_binop(Opcode op) {
    switch (op) {
    case Opcode::Add: case Opcode::Sub: case Opcode::Mul:
    case Opcode::UDiv: case Opcode::SDiv: case Opcode::URem: case Opcode::SRem:
    case Opcode::And: case Opcode::Or: case Opcode::Xor:
    case Opcode::Shl: case Opcode::LShr: case Opcode::AShr:
        return true;
    default:
        return false;
    }
}

// Per-function lowering state.
struct LowerCtx {
    // vector value name -> its lane values in the lowered function
    std::unordered_map<std::string, std::vector<std::shared_ptr<Value>>> lanes;
    // scalar value name -> replacement value (extractelement aliasing)
    std::unordered_map<std::string, std::shared_ptr<Value>> replacement;
    // every name already used in the lowered function (uniquing)
    std::unordered_set<std::string> used_names;

    std::string unique_name(const std::string& base) {
        std::string n = base;
        while (!used_names.insert(n).second) n += "_";
        return n;
    }
};

std::shared_ptr<Value> remap_scalar(LowerCtx& ctx, const std::shared_ptr<Value>& v) {
    if (v && v->has_name()) {
        auto it = ctx.replacement.find(v->name());
        if (it != ctx.replacement.end()) return it->second;
    }
    return v;
}

// Resolve a vector-typed operand to its lane values. Returns false on
// anything unsupported (undef lanes, unknown names, non-int constants).
bool lanes_of(LowerCtx& ctx, const std::shared_ptr<Value>& v,
              std::vector<std::shared_ptr<Value>>& out) {
    if (!v || !is_vector_type(v->type())) return false;
    if (auto* cv = dynamic_cast<const ConstantVector*>(v.get())) {
        out.clear();
        for (auto& e : cv->elements()) {
            if (!dynamic_cast<const ConstantInt*>(e.get())) return false; // undef/FP lane
            out.push_back(e);
        }
        return !out.empty();
    }
    if (v->has_name()) {
        auto it = ctx.lanes.find(v->name());
        if (it == ctx.lanes.end()) return false;
        out = it->second;
        return true;
    }
    return false;
}

// Constant lane index of an extract/insert/shuffle-mask operand.
bool const_index(const std::shared_ptr<Value>& v, int64_t& out) {
    auto* ci = dynamic_cast<const ConstantInt*>(v.get());
    if (!ci) return false;
    out = ci->value();
    return true;
}

// Shared lowering body. `want_lane`: SIZE_MAX for a scalar-returning
// function, otherwise the result lane to return.
std::shared_ptr<Function> scalarize_impl(const Function& fn, size_t want_lane) {
    const bool vector_ret = is_vector_type(fn.return_type());
    if (vector_ret == (want_lane == SIZE_MAX)) return nullptr; // wrong entry point

    LowerCtx ctx;

    // Seed used_names with every name in the source so fresh lane names
    // can't collide with an existing scalar value.
    for (auto& arg : fn.arguments()) {
        if (!arg.name.empty()) ctx.used_names.insert(arg.name);
    }
    for (auto& block : fn.blocks()) {
        for (auto& inst : block->instructions()) {
            if (inst && inst->has_name()) ctx.used_names.insert(inst->name());
        }
    }

    // ── Expanded signature ───────────────────────────────────────────────
    std::vector<std::shared_ptr<Type>> param_types;
    struct NewArg { std::shared_ptr<Type> type; std::string name; };
    std::vector<NewArg> new_args;
    for (auto& arg : fn.arguments()) {
        if (is_vector_type(arg.type)) {
            auto& vt = static_cast<const VectorType&>(*arg.type);
            auto elem = vt.element_type();
            if (!elem->is_integer() || arg.name.empty()) return nullptr;
            std::vector<std::shared_ptr<Value>> lane_vals;
            for (uint64_t k = 0; k < vt.count(); ++k) {
                std::string ln = ctx.unique_name(arg.name + ".l" + std::to_string(k));
                lane_vals.push_back(std::make_shared<Value>(elem, ln));
                param_types.push_back(elem);
                new_args.push_back({elem, ln});
            }
            ctx.lanes[arg.name] = std::move(lane_vals);
        } else {
            param_types.push_back(arg.type);
            new_args.push_back({arg.type, arg.name});
        }
    }

    std::shared_ptr<Type> ret_ty = fn.return_type();
    if (vector_ret) {
        auto& vt = static_cast<const VectorType&>(*ret_ty);
        if (want_lane >= vt.count() || !vt.element_type()->is_integer()) return nullptr;
        ret_ty = vt.element_type();
    }

    auto fn_ty = std::make_shared<FunctionType>(ret_ty, param_types);
    std::string suffix = vector_ret ? ".lane" + std::to_string(want_lane) : "";
    auto out = std::make_shared<Function>(fn.name() + ".scalarized" + suffix,
                                          fn_ty, fn.linkage());
    for (auto& a : new_args) out->add_argument(a.type, a.name);
    for (auto& [k, v] : fn.attributes()) out->set_attribute(k, v);

    // ── Lower each block ─────────────────────────────────────────────────
    for (auto& block : fn.blocks()) {
        auto& nb = out->add_block(block->name());

        for (auto& inst : block->instructions()) {
            if (!inst) continue;
            const auto op = inst->opcode();
            const bool result_is_vector = is_vector_type(inst->type());

            // Any vector intermediate must be nameable for by-name wiring.
            // (Terminators carry their operand's type but define no value,
            // so a vector-typed `ret` is fine — handled below.)
            if (result_is_vector && !inst->is_terminator() && !inst->has_name()) {
                return nullptr;
            }

            // -- Vector binop: lane-blast --------------------------------
            if (inst->is_binary_op() && result_is_vector) {
                if (!is_lane_blastable_binop(op)) return nullptr; // FP vector
                std::vector<std::shared_ptr<Value>> a, b;
                if (!lanes_of(ctx, inst->operand(0), a) ||
                    !lanes_of(ctx, inst->operand(1), b) || a.size() != b.size()) {
                    return nullptr;
                }
                std::vector<std::shared_ptr<Value>> res;
                for (size_t k = 0; k < a.size(); ++k) {
                    auto li = inst::make_binop(
                        op, a[k], b[k],
                        ctx.unique_name(inst->name() + ".l" + std::to_string(k)),
                        inst->binop_flags());
                    nb.add_instruction(li);
                    res.push_back(li);
                }
                ctx.lanes[inst->name()] = std::move(res);
                continue;
            }

            // -- ExtractElement (constant index): pure aliasing ----------
            if (op == Opcode::ExtractElement) {
                std::vector<std::shared_ptr<Value>> v;
                int64_t idx = 0;
                if (!lanes_of(ctx, inst->operand(0), v) ||
                    !const_index(inst->operand(1), idx) ||
                    idx < 0 || static_cast<size_t>(idx) >= v.size() ||
                    !inst->has_name()) {
                    return nullptr;
                }
                ctx.replacement[inst->name()] = v[static_cast<size_t>(idx)];
                continue;
            }

            // -- InsertElement (constant index): lane re-wiring ----------
            if (op == Opcode::InsertElement) {
                std::vector<std::shared_ptr<Value>> v;
                int64_t idx = 0;
                if (!lanes_of(ctx, inst->operand(0), v) ||
                    !const_index(inst->operand(2), idx) ||
                    idx < 0 || static_cast<size_t>(idx) >= v.size()) {
                    return nullptr;
                }
                auto elem = remap_scalar(ctx, inst->operand(1));
                if (is_vector_type(elem->type())) return nullptr;
                v[static_cast<size_t>(idx)] = elem;
                ctx.lanes[inst->name()] = std::move(v);
                continue;
            }

            // -- ShuffleVector (constant, non-undef mask) ----------------
            if (op == Opcode::ShuffleVector) {
                std::vector<std::shared_ptr<Value>> a, b;
                if (!lanes_of(ctx, inst->operand(0), a) ||
                    !lanes_of(ctx, inst->operand(1), b) || a.size() != b.size()) {
                    return nullptr;
                }
                auto* mask = dynamic_cast<const ConstantVector*>(inst->operand(2).get());
                if (!mask) return nullptr;
                std::vector<std::shared_ptr<Value>> res;
                for (auto& m : mask->elements()) {
                    int64_t idx = 0;
                    if (!const_index(m, idx) || idx < 0 ||
                        static_cast<size_t>(idx) >= a.size() + b.size()) {
                        return nullptr; // undef (-1) or out-of-range lane
                    }
                    res.push_back(static_cast<size_t>(idx) < a.size()
                                      ? a[static_cast<size_t>(idx)]
                                      : b[static_cast<size_t>(idx) - a.size()]);
                }
                ctx.lanes[inst->name()] = std::move(res);
                continue;
            }

            // -- Reduce intrinsic: balanced binop tree -------------------
            if (op == Opcode::Call) {
                auto it = inst->metadata().find("callee");
                Opcode lane_op;
                if (it != inst->metadata().end() &&
                    parse_reduce_intrinsic(it->second, &lane_op)) {
                    std::vector<std::shared_ptr<Value>> v;
                    if (inst->num_operands() != 1 ||
                        !lanes_of(ctx, inst->operand(0), v) || !inst->has_name()) {
                        return nullptr;
                    }
                    // Balanced tree: repeatedly combine adjacent pairs. The
                    // last combine takes the call's own result name so later
                    // by-name uses resolve to the tree root.
                    std::vector<std::shared_ptr<Value>> level = std::move(v);
                    size_t tmp = 0;
                    while (level.size() > 1) {
                        std::vector<std::shared_ptr<Value>> next;
                        for (size_t k = 0; k + 1 < level.size(); k += 2) {
                            const bool is_root =
                                level.size() == 2 && k == 0;
                            std::string nm = is_root
                                ? inst->name()
                                : ctx.unique_name(inst->name() + ".r" +
                                                  std::to_string(tmp++));
                            auto bi = inst::make_binop(lane_op, level[k],
                                                       level[k + 1], nm);
                            nb.add_instruction(bi);
                            next.push_back(bi);
                        }
                        if (level.size() % 2 == 1) next.push_back(level.back());
                        level = std::move(next);
                    }
                    // Register the tree root under the call's name so later
                    // uses resolve to the NEW object (the interpreter keys
                    // values by pointer identity, not name). Also covers the
                    // single-lane degenerate case, where the "root" is the
                    // lane value itself.
                    if (level.size() == 1 && level[0]) {
                        ctx.replacement[inst->name()] = level[0];
                    }
                    continue;
                }
                // Non-intrinsic call with vector involvement: unsupported.
                if (result_is_vector) return nullptr;
                for (auto& o : inst->operands()) {
                    if (o && is_vector_type(o->type())) return nullptr;
                }
                // fall through to the scalar copy below
            }

            // -- Select with scalar cond over vectors: lane-wise ---------
            if (op == Opcode::Select && result_is_vector) {
                auto cond = remap_scalar(ctx, inst->operand(0));
                if (is_vector_type(cond->type())) return nullptr; // vector cond
                std::vector<std::shared_ptr<Value>> t, f;
                if (!lanes_of(ctx, inst->operand(1), t) ||
                    !lanes_of(ctx, inst->operand(2), f) || t.size() != f.size()) {
                    return nullptr;
                }
                std::vector<std::shared_ptr<Value>> res;
                for (size_t k = 0; k < t.size(); ++k) {
                    auto si = inst::make_select(
                        cond, t[k], f[k],
                        ctx.unique_name(inst->name() + ".l" + std::to_string(k)));
                    nb.add_instruction(si);
                    res.push_back(si);
                }
                ctx.lanes[inst->name()] = std::move(res);
                continue;
            }

            // -- Ret -----------------------------------------------------
            if (op == Opcode::Ret) {
                if (inst->num_operands() == 0) {
                    if (vector_ret) return nullptr;
                    nb.add_instruction(inst::make_ret_void());
                    continue;
                }
                auto rv = inst->operand(0);
                if (is_vector_type(rv->type())) {
                    if (!vector_ret) return nullptr;
                    std::vector<std::shared_ptr<Value>> v;
                    if (!lanes_of(ctx, rv, v) || want_lane >= v.size()) return nullptr;
                    nb.add_instruction(inst::make_ret(v[want_lane]));
                } else {
                    if (vector_ret) return nullptr;
                    nb.add_instruction(inst::make_ret(remap_scalar(ctx, rv)));
                }
                continue;
            }

            // -- Everything else: must be vector-free; copy verbatim -----
            if (result_is_vector) return nullptr; // vector phi/load/icmp/...
            for (auto& o : inst->operands()) {
                if (o && is_vector_type(o->type())) return nullptr;
            }
            auto ni = std::make_shared<Instruction>(op, inst->type(), inst->name());
            for (auto& o : inst->operands()) ni->add_operand(remap_scalar(ctx, o));
            for (auto& [k, v] : inst->metadata()) ni->set_metadata(k, v);
            ni->binop_flags() = inst->binop_flags();
            if (inst->alignment()) ni->set_alignment(inst->alignment().value());
            ni->set_volatile(inst->is_volatile());
            nb.add_instruction(ni);
            // Register the copy under its own name: later operand uses must
            // point at the NEW object, not the source function's instruction
            // (the interpreter keys bindings by pointer identity, not name).
            if (ni->has_name()) ctx.replacement[ni->name()] = ni;
        }
    }

    return out;
}

} // anonymous namespace

bool function_has_vector_ops(const Function& fn) {
    if (is_vector_type(fn.return_type())) return true;
    for (auto& arg : fn.arguments()) {
        if (is_vector_type(arg.type)) return true;
    }
    for (auto& block : fn.blocks()) {
        for (auto& inst : block->instructions()) {
            if (!inst) continue;
            if (is_vector_type(inst->type())) return true;
            switch (inst->opcode()) {
            case Opcode::ExtractElement:
            case Opcode::InsertElement:
            case Opcode::ShuffleVector:
                return true;
            default:
                break;
            }
            for (auto& o : inst->operands()) {
                if (o && is_vector_type(o->type())) return true;
            }
            if (inst->opcode() == Opcode::Call) {
                auto it = inst->metadata().find("callee");
                if (it != inst->metadata().end() &&
                    it->second.rfind("clunk.vector.", 0) == 0) {
                    return true;
                }
            }
        }
    }
    return false;
}

bool parse_reduce_intrinsic(const std::string& callee, Opcode* out_op) {
    static const char kPrefix[] = "clunk.vector.reduce.";
    if (callee.rfind(kPrefix, 0) != 0) return false;
    std::string rest = callee.substr(sizeof(kPrefix) - 1);
    auto dot = rest.find('.');
    std::string opname = dot == std::string::npos ? rest : rest.substr(0, dot);
    Opcode op;
    if (opname == "add")      op = Opcode::Add;
    else if (opname == "mul") op = Opcode::Mul;
    else if (opname == "and") op = Opcode::And;
    else if (opname == "or")  op = Opcode::Or;
    else if (opname == "xor") op = Opcode::Xor;
    else return false;
    if (out_op) *out_op = op;
    return true;
}

std::shared_ptr<Function> scalarize_function(const Function& fn) {
    return scalarize_impl(fn, SIZE_MAX);
}

std::shared_ptr<Function> scalarize_lane(const Function& fn, size_t lane) {
    return scalarize_impl(fn, lane);
}

} // namespace clunk::ir
