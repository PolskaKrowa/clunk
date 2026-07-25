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
 * Clunk MemOpt — alias-aware memory optimisations.
 * See include/clunk/Search/MemOpt.h for the soundness contract.
 */
#include "clunk/Search/MemOpt.h"

#include <algorithm>
#include <unordered_map>
#include <unordered_set>

#include "clunk/IR/Clone.h"
#include "clunk/IR/Instruction.h"
#include "clunk/IR/Value.h"

namespace clunk::search {

namespace {

// Find the defining instruction for a named value (nullptr for args etc.).
const ir::Instruction* find_def(const ir::Function& fn, const std::string& name) {
    for (auto& bb : fn.blocks()) {
        for (auto& inst : bb->instructions()) {
            if (inst && inst->has_name() && inst->name() == name) return inst.get();
        }
    }
    return nullptr;
}

} // anonymous namespace

// ── AliasOracle ─────────────────────────────────────────────────────────────

AliasOracle::AliasOracle(const ir::Function& fn) : fn_(fn) {
    // Escape analysis for allocas: collect each alloca's derived-name
    // closure (GEP/bitcast chains), then look for a use that lets the
    // address leave the function: stored AS A VALUE, passed to a call,
    // ptrtoint'd, returned, or merged through a phi/select (conservative).
    std::vector<std::string> allocas;
    for (auto& bb : fn.blocks()) {
        for (auto& inst : bb->instructions()) {
            if (inst && inst->opcode() == ir::Opcode::Alloca && inst->has_name())
                allocas.push_back(inst->name());
        }
    }
    for (auto& a : allocas) {
        // Derived-name closure.
        std::unordered_set<std::string> derived = {a};
        bool grew = true;
        while (grew) {
            grew = false;
            for (auto& bb : fn.blocks()) {
                for (auto& inst : bb->instructions()) {
                    if (!inst || !inst->has_name() || derived.count(inst->name()))
                        continue;
                    if (inst->opcode() != ir::Opcode::GetElementPtr &&
                        inst->opcode() != ir::Opcode::BitCast)
                        continue;
                    auto base = inst->num_operands() ? inst->operand(0) : nullptr;
                    if (base && base->has_name() && derived.count(base->name())) {
                        derived.insert(inst->name());
                        grew = true;
                    }
                }
            }
        }
        // Escape scan.
        bool escapes = false;
        for (auto& bb : fn.blocks()) {
            for (auto& inst : bb->instructions()) {
                if (!inst || escapes) break;
                const auto op = inst->opcode();
                for (size_t i = 0; i < inst->num_operands(); ++i) {
                    auto v = inst->operand(i);
                    if (!v || !v->has_name() || !derived.count(v->name())) continue;
                    const bool benign =
                        (op == ir::Opcode::Load && i == 0) ||
                        (op == ir::Opcode::Store && i == 1) ||  // as ADDRESS
                        ((op == ir::Opcode::GetElementPtr ||
                          op == ir::Opcode::BitCast) && i == 0);
                    if (!benign) { escapes = true; break; }
                }
            }
            if (escapes) break;
        }
        if (!escapes) nonescaping_allocas_.push_back(a);
    }
}

AliasOracle::PointerInfo AliasOracle::classify(
    const std::shared_ptr<ir::Value>& p) const {
    PointerInfo info;
    std::shared_ptr<ir::Value> cur = p;
    // Walk GEP/bitcast chains down to the root. Only a SINGLE GEP level's
    // indices are kept (nested GEPs make the tuple comparison ambiguous —
    // fall back to non-const which answers MayAlias).
    size_t gep_levels = 0;
    for (int fuel = 32; cur && fuel > 0; --fuel) {
        if (!cur->has_name()) break;
        const auto* def = find_def(fn_, cur->name());
        if (!def) break;  // argument or global
        if (def->opcode() == ir::Opcode::BitCast && def->num_operands() >= 1) {
            cur = def->operand(0);
            continue;
        }
        if (def->opcode() == ir::Opcode::GetElementPtr && def->num_operands() >= 2) {
            if (++gep_levels > 1) info.const_gep = false;
            if (gep_levels == 1) {
                for (size_t i = 1; i < def->num_operands(); ++i) {
                    auto* ci = dynamic_cast<const ir::ConstantInt*>(
                        def->operand(i).get());
                    if (!ci) { info.const_gep = false; break; }
                    info.indices.push_back(ci->value());
                }
                // Element geometry: the GEP's OWN result pointee (robust
                // under opaque `ptr` bases, where the base operand's type
                // says nothing about the element size).
                if (def->type() && def->type()->is_pointer()) {
                    info.gep_source_type = static_cast<const ir::PointerType&>(
                                               *def->type())
                                               .pointee()
                                               ->to_string();
                }
                if (info.gep_source_type.empty() ||
                    info.gep_source_type == "void") {
                    info.const_gep = false;  // unknown geometry → MayAlias
                }
            }
            cur = def->operand(0);
            continue;
        }
        if (def->opcode() == ir::Opcode::Alloca) {
            info.kind = PointerInfo::RootKind::Alloca;
            info.root_name = def->name();
            info.root_escapes = true;
            for (auto& a : nonescaping_allocas_) {
                if (a == info.root_name) { info.root_escapes = false; break; }
            }
            return info;
        }
        break;  // load result, call result, phi, select, ... → Unknown root
    }
    if (cur && cur->has_name() && !find_def(fn_, cur->name())) {
        // Argument or global reference.
        for (auto& arg : fn_.arguments()) {
            if (arg.name == cur->name()) {
                info.kind = PointerInfo::RootKind::Argument;
                info.root_name = arg.name;
                info.root_noalias = arg.attrs.count("noalias") != 0;
                return info;
            }
        }
        info.kind = PointerInfo::RootKind::Global;
        info.root_name = cur->name();
        return info;
    }
    return info;  // Unknown
}

AliasResult AliasOracle::alias(const std::shared_ptr<ir::Value>& p,
                               const std::shared_ptr<ir::Value>& q) const {
    if (!p || !q) return AliasResult::MayAlias;
    const auto a = classify(p), b = classify(q);
    using RK = PointerInfo::RootKind;

    const bool same_root = a.kind != RK::Unknown && a.kind == b.kind &&
                           !a.root_name.empty() && a.root_name == b.root_name;
    if (same_root) {
        // Identical constant access paths → the same address.
        if (a.const_gep && b.const_gep && a.indices == b.indices &&
            a.gep_source_type == b.gep_source_type) {
            return AliasResult::MustAlias;
        }
        // Distinct constant elements of the same object → disjoint.
        // Restricted to SINGLE-index GEPs: base + k*sizeof(elem) with
        // k1 != k2 can never coincide. Multi-index tuples are NOT safe to
        // compare element-wise — an out-of-range inner index (e.g. (0,4)
        // vs (1,0) into [4 x i32]) can address the same byte with a
        // different tuple.
        if (a.const_gep && b.const_gep &&
            a.gep_source_type == b.gep_source_type &&
            a.indices.size() == 1 && b.indices.size() == 1 &&
            a.indices != b.indices) {
            return AliasResult::NoAlias;
        }
        return AliasResult::MayAlias;
    }

    // Different roots.
    const bool a_alloca = a.kind == RK::Alloca, b_alloca = b.kind == RK::Alloca;
    if (a_alloca && b_alloca) return AliasResult::NoAlias;  // distinct stack slots
    // Stack memory vs. pointers that existed before the frame.
    if ((a_alloca && (b.kind == RK::Argument || b.kind == RK::Global)) ||
        (b_alloca && (a.kind == RK::Argument || a.kind == RK::Global))) {
        return AliasResult::NoAlias;
    }
    // A never-escaping alloca vs. anything not derived from it.
    if ((a_alloca && !a.root_escapes) || (b_alloca && !b.root_escapes)) {
        return AliasResult::NoAlias;
    }
    // Two distinct noalias arguments.
    if (a.kind == RK::Argument && b.kind == RK::Argument &&
        a.root_noalias && b.root_noalias) {
        return AliasResult::NoAlias;
    }
    return AliasResult::MayAlias;
}

// ── MemOptimizer ────────────────────────────────────────────────────────────

std::shared_ptr<ir::Function> MemOptimizer::optimize(const ir::Function& fn) {
    auto work = ir::deep_copy_function(fn);
    AliasOracle oracle(*work);
    bool changed = false;

    // name of a forwarded/eliminated load → the value replacing it
    std::unordered_map<std::string, std::shared_ptr<ir::Value>> subst;

    for (auto& bb : work->blocks()) {
        auto& instrs = bb->instructions();

        struct Avail {
            std::shared_ptr<ir::Value> ptr;
            std::shared_ptr<ir::Value> value;   // stored value or load result
            size_t store_index = SIZE_MAX;      // defining store (DSE target)
            bool maybe_read = false;            // a later access may read it
        };
        std::vector<Avail> avail;
        std::vector<size_t> to_delete;

        auto clobber_all = [&avail] {
            avail.clear();
        };
        auto mark_all_read = [&avail] {
            for (auto& r : avail) r.maybe_read = true;
        };

        for (size_t i = 0; i < instrs.size(); ++i) {
            auto inst = instrs[i];
            if (!inst) continue;
            const auto op = inst->opcode();

            if (op == ir::Opcode::Load && inst->num_operands() >= 1) {
                auto p = inst->operand(0);
                if (inst->is_volatile()) {
                    mark_all_read();
                    clobber_all();
                    continue;
                }
                // Forward from a must-aliasing available value of the same
                // type.
                bool forwarded = false;
                for (auto& r : avail) {
                    if (oracle.alias(p, r.ptr) != AliasResult::MustAlias) continue;
                    if (!r.value || !r.value->type() || !inst->type() ||
                        !(*r.value->type() == *inst->type())) {
                        continue;
                    }
                    if (inst->has_name()) subst[inst->name()] = r.value;
                    to_delete.push_back(i);
                    if (r.store_index != SIZE_MAX) ++stats_.loads_forwarded;
                    else ++stats_.loads_eliminated;
                    forwarded = true;
                    changed = true;
                    break;
                }
                if (forwarded) continue;
                // Reading may-aliased pending stores blocks their DSE.
                for (auto& r : avail) {
                    if (r.store_index != SIZE_MAX &&
                        oracle.alias(p, r.ptr) != AliasResult::NoAlias) {
                        r.maybe_read = true;
                    }
                }
                if (inst->has_name()) {
                    avail.push_back({p, inst, SIZE_MAX, false});
                }
                continue;
            }

            if (op == ir::Opcode::Store && inst->num_operands() >= 2) {
                auto v = inst->operand(0);
                auto p = inst->operand(1);
                if (inst->is_volatile()) {
                    mark_all_read();
                    clobber_all();
                    continue;
                }
                // DSE: a pending store to the SAME address that nothing may
                // have read, overwritten by a same-width value, is dead.
                for (auto& r : avail) {
                    if (r.store_index == SIZE_MAX || r.maybe_read) continue;
                    if (oracle.alias(p, r.ptr) != AliasResult::MustAlias) continue;
                    auto prev = instrs[r.store_index];
                    if (!prev || prev->num_operands() < 1 || !v ||
                        !prev->operand(0) || !prev->operand(0)->type() ||
                        !v->type() ||
                        !(*prev->operand(0)->type() == *v->type())) {
                        continue;
                    }
                    to_delete.push_back(r.store_index);
                    ++stats_.stores_eliminated;
                    changed = true;
                    break;
                }
                // Invalidate everything this store may overwrite, then
                // record it.
                std::vector<Avail> kept;
                for (auto& r : avail) {
                    if (oracle.alias(p, r.ptr) == AliasResult::NoAlias) {
                        kept.push_back(r);
                    }
                }
                avail = std::move(kept);
                avail.push_back({p, v, i, false});
                continue;
            }

            if (op == ir::Opcode::Call || op == ir::Opcode::Invoke ||
                op == ir::Opcode::Fence) {
                mark_all_read();
                clobber_all();
                continue;
            }
        }

        // Apply deletions (descending index keeps positions valid).
        std::sort(to_delete.begin(), to_delete.end());
        for (size_t k = to_delete.size(); k-- > 0;) {
            instrs.erase(instrs.begin() +
                         static_cast<std::ptrdiff_t>(to_delete[k]));
        }
    }

    if (!changed) return nullptr;

    // Function-wide substitution of forwarded load results.
    if (!subst.empty()) {
        for (auto& bb : work->blocks()) {
            for (auto& inst : bb->instructions()) {
                if (!inst) continue;
                for (size_t oi = 0; oi < inst->num_operands(); ++oi) {
                    auto op = inst->operand(oi);
                    if (!op || !op->has_name()) continue;
                    auto it = subst.find(op->name());
                    if (it != subst.end()) inst->set_operand(oi, it->second);
                }
            }
        }
    }

    if (!ir::validate_function(*work)) return nullptr;
    return work;
}

} // namespace clunk::search
