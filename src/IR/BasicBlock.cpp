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
 * Clunk IR BasicBlock — implementation of non-inline methods.
 */
#include "clunk/IR/BasicBlock.h"

namespace clunk::ir {

// ── successors() ─────────────────────────────────────────────────────────
// Inspect the terminator instruction to determine which basic-block labels
// control flow may transfer to.
//
//   Conditional Br  →  metadata "true_label" / "false_label"
//   Unconditional Br → metadata "dest_label"
//   Ret / other      → empty
//
std::vector<std::string> BasicBlock::successors() const {
    auto term = terminator();
    if (!term) return {};

    if (term->opcode() == Opcode::Br) {
        const auto& md = term->metadata();

        // Conditional branch: has true/false labels
        auto true_it = md.find("true_bb");
        auto false_it = md.find("false_bb");
        if (true_it != md.end() && false_it != md.end()) {
            return {true_it->second, false_it->second};
        }

        // Unconditional branch: single destination
        auto dest_it = md.find("dest_bb");
        if (dest_it != md.end()) {
            return {dest_it->second};
        }

        // Fallback: try to extract from operands (as string-named values)
        // This handles cases where branch targets were stored as operands.
        std::vector<std::string> succs;
        for (const auto& op : term->operands()) {
            if (op && op->has_name()) {
                succs.push_back(op->name());
            }
        }
        return succs;
    }

    if (term->opcode() == Opcode::Switch) {
        // Switch: metadata "default_bb" + "case_N_bb" entries
        std::vector<std::string> succs;
        const auto& md = term->metadata();
        auto def_it = md.find("default_bb");
        if (def_it != md.end()) {
            succs.push_back(def_it->second);
        }
        // Collect case labels in order
        for (size_t i = 0; ; ++i) {
            auto it = md.find("case_" + std::to_string(i) + "_bb");
            if (it == md.end()) break;
            succs.push_back(it->second);
        }
        return succs;
    }

    // Ret, Invoke, Resume, Unreachable, etc. → no successors
    return {};
}

// ── to_string() ─────────────────────────────────────────────────────────
// Render the basic block in LLVM IR textual form:
//
//   bb_name:
//     %x = add i32 %a, %b
//     br label %exit
//
std::string BasicBlock::to_string() const {
    std::string s;
    s.reserve(256);
    s.append(name_);
    s.append(":\n");
    for (const auto& inst : instrs_) {
        s.append("  ");
        s.append(inst->to_string());
        s.append("\n");
    }
    return s;
}

} // namespace clunk::ir
