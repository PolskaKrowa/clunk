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
 * Clunk IR Module — implementation of non-inline methods.
 */
#include "clunk/IR/Module.h"

namespace clunk::ir {

// ── Helper: linkage keyword for globals/declarations ────────────────────
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

// ── instruction_count() ─────────────────────────────────────────────────
// Sum of all instructions across every function.
size_t Module::instruction_count() const {
    size_t count = 0;
    for (const auto& fn : functions_) {
        count += fn->instruction_count();
    }
    return count;
}

// ── to_string() ─────────────────────────────────────────────────────────
// Render the complete module as LLVM IR text:
//
//   ; ModuleID = 'module_name'
//   source_filename = "..."
//   target triple = "x86_64-unknown-linux-gnu"
//   target datalayout = "e-m:e-..."
//
//   !llvm.module.flags = !{ ... }
//
//   declare i32 @external_func(i32)
//
//   define i32 @my_func(i32 %x) {
//   ...
//   }
//
std::string Module::to_string() const {
    std::string s;
    s.reserve(1024);

    // Module header
    s.append("; ModuleID = '");
    s.append(name_);
    s.append("'\n");

    // Source filename
    if (!source_filename_.empty()) {
        s.append("source_filename = \"");
        s.append(source_filename_);
        s.append("\"\n");
    }

    // Target information
    if (has_target()) {
        if (!target_.triple.empty()) {
            s.append("target triple = \"");
            s.append(target_.triple);
            s.append("\"\n");
        }
        if (!target_.datalayout.empty()) {
            s.append("target datalayout = \"");
            s.append(target_.datalayout);
            s.append("\"\n");
        }
    }

    // Module flags — required for clang to determine PIC mode, code model, etc.
    // LLVM IR syntax:
    //   !0 = !{ i32 1, !"PIC Level", i32 2 }
    //   !llvm.module.flags = !{ !0, !1 }
    if (!module_flags_.empty()) {
        s.append("\n");
        for (size_t i = 0; i < module_flags_.size(); ++i) {
            const auto& mf = module_flags_[i];
            s.append("!");
            s.append(std::to_string(i));
            s.append(" = !{ i32 ");
            s.append(std::to_string(mf.behavior));
            s.append(", !\"");
            s.append(mf.key);
            s.append("\", ");
            s.append(mf.value);
            s.append(" }\n");
        }
        s.append("!llvm.module.flags = !{");
        for (size_t i = 0; i < module_flags_.size(); ++i) {
            if (i > 0) s.append(", ");
            s.append(" !");
            s.append(std::to_string(i));
        }
        s.append(" }\n");
    }

    // Named types
    for (const auto& [tname, ty] : named_types_) {
        s.append("%");
        s.append(tname);
        s.append(" = type ");
        s.append(ty->to_string());
        s.append("\n");
    }

    // Global variables — emit in canonical LLVM IR form:
    //   @name = [linkage] [constant|global] [type] [init_value] [, align N]
    for (const auto& gv : globals_) {
        s.append(gv.name);
        s.append(" = ");
        const char* linkage_str = linkage_name(gv.linkage);
        s.append(linkage_str);
        if (gv.is_constant) {
            s.append("constant ");
        } else {
            s.append("global ");
        }
        if (gv.type) {
            s.append(gv.type->to_string());
        } else {
            s.append("ptr");  // fallback if type was lost during parsing
        }
        if (!gv.init_value.empty()) {
            s += ' ';
            s.append(gv.init_value);
        }
        if (gv.alignment > 0) {
            s.append(", align ");
            s.append(std::to_string(gv.alignment));
        }
        s.append("\n");
    }

    // Separate globals from functions with a blank line if both exist
    if (!globals_.empty() && !functions_.empty()) {
        s.append("\n");
    }

    // Functions — declarations (no basic blocks) vs definitions
    for (const auto& fn : functions_) {
        if (fn->blocks().empty()) {
            // Declaration (external, no body)
            s.append("declare ");
            s.append(fn->return_type()->to_string());
            s.append(" @");
            s.append(fn->name());
            s.append("(");
            const auto& args = fn->arguments();
            for (size_t i = 0; i < args.size(); ++i) {
                if (i > 0) s.append(", ");
                s.append(args[i].type->to_string());
            }
            if (fn->function_type()->is_vararg()) {
                if (!args.empty()) s.append(", ");
                s.append("...");
            }
            s.append(")\n\n");
        } else {
            // Definition with body
            s.append(fn->to_string());
            s.append("\n");
        }
    }

    return s;
}

} // namespace clunk::ir
