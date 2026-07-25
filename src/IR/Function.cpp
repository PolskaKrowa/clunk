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
 * Clunk IR Function — implementation of non-inline methods.
 */
#include "clunk/IR/Function.h"

namespace clunk::ir {

// ── Helper: linkage keyword ──────────────────────────────────────────────
static const char* linkage_name(Linkage l) {
    switch (l) {
        case Linkage::External:           return "";
        case Linkage::Internal:           return "internal ";
        case Linkage::Private:            return "private ";
        case Linkage::LinkOnce:           return "linkonce ";
        case Linkage::Weak:               return "weak ";
        case Linkage::Common:             return "common ";
        case Linkage::Appending:          return "appending ";
        case Linkage::ExternalWeak:       return "extern_weak ";
        case Linkage::LinkOnceODR:        return "linkonce_odr ";
        case Linkage::WeakODR:            return "weak_odr ";
        case Linkage::AvailableExternally:return "available_externally ";
    }
    return "";
}

// ── compute_predecessors() ───────────────────────────────────────────────
// Rebuild predecessor lists for every basic block in this function.
//
// Algorithm:
//   1. Clear all predecessor lists.
//   2. For each block B, inspect its terminator to get successor labels.
//   3. For each successor S, look up the block and add B's name as a
//      predecessor of S.
//
void Function::compute_predecessors() {
    // Step 1: Clear all predecessor lists
    for (auto& bb : blocks_) {
        bb->clear_predecessors();
    }

    // Step 2–3: Walk every block and wire predecessor edges
    for (const auto& bb : blocks_) {
        auto succs = bb->successors();
        for (const auto& succ_name : succs) {
            auto succ_bb = block(succ_name);
            if (succ_bb) {
                succ_bb->add_predecessor(bb->name());
            }
        }
    }
}

// ── instruction_count() ─────────────────────────────────────────────────
// Sum of all instructions across every basic block.
size_t Function::instruction_count() const {
    size_t count = 0;
    for (const auto& bb : blocks_) {
        count += bb->size();
    }
    return count;
}

// ── to_string() ─────────────────────────────────────────────────────────
// Render the function in LLVM IR textual form:
//
//   define i32 @func_name(i32 %arg0, i32 %arg1) {
//   entry:
//     ...
//   bb1:
//     ...
//   }
//
std::string Function::to_string() const {
    std::string s;
    s.reserve(256);

    // Linkage keyword (empty for external, which is the default)
    s.append(linkage_name(linkage_));

    // "define <return_type> @<name>(<args>) {"
    s.append("define ");
    s.append(fn_type_->return_type()->to_string());
    s.append(" @");
    s.append(name_);
    s.append("(");

    // Arguments
    for (size_t i = 0; i < args_.size(); ++i) {
        if (i > 0) s.append(", ");
        s.append(args_[i].type->to_string());
        if (!args_[i].name.empty()) {
            s.append(" %");
            s.append(args_[i].name);
        }
    }

    // Vararg
    if (fn_type_->is_vararg()) {
        if (!args_.empty()) s.append(", ");
        s.append("...");
    }

    s.append(") {\n");

    // Basic blocks
    for (const auto& bb : blocks_) {
        s.append(bb->to_string());
    }

    s.append("}\n");
    return s;
}

} // namespace clunk::ir
