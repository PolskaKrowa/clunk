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
 * Clunk IR Module — top-level container for functions and global values.
 */
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include "clunk/IR/Type.h"
#include "clunk/IR/Function.h"

namespace clunk::ir {

struct GlobalValue {
    std::string name;
    std::shared_ptr<Type> type;
    std::string init_value; // Initialiser string (simplified)
    Linkage linkage = Linkage::External;
    bool is_constant = false;
    unsigned alignment = 0;
};

struct ModuleFlag {
    unsigned behavior;  // 1=max, 2=min, 3=warning, 4=require, 5=override
    std::string key;
    std::string value;   // could be i32 value or string
};

struct TargetInfo {
    std::string triple;       // e.g. "x86_64-unknown-linux-gnu"
    std::string datalayout;   // e.g. "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
};

class Module {
public:
    explicit Module(const std::string& name = "") : name_(name) {}

    const std::string& name() const { return name_; }

    // Functions
    Function& add_function(const std::string& name,
                           std::shared_ptr<FunctionType> fn_type,
                           Linkage linkage = Linkage::External) {
        auto fn = std::make_shared<Function>(name, fn_type, linkage);
        functions_.push_back(fn);
        fn_index_[name] = functions_.size() - 1;
        return *fn;
    }

    void add_function(std::shared_ptr<Function> fn) {
        fn_index_[fn->name()] = functions_.size();
        functions_.push_back(fn);
    }

    std::shared_ptr<Function> function(const std::string& name) const {
        auto it = fn_index_.find(name);
        if (it == fn_index_.end()) return nullptr;
        return functions_.at(it->second);
    }

    const std::vector<std::shared_ptr<Function>>& functions() const { return functions_; }

    // Globals
    void add_global(GlobalValue gv) {
        globals_.push_back(std::move(gv));
    }
    const std::vector<GlobalValue>& globals() const { return globals_; }

    // Target info
    void set_target(const TargetInfo& info) { target_ = info; }
    const TargetInfo& target() const { return target_; }
    bool has_target() const { return !target_.triple.empty(); }

    // Source filename (from `source_filename = "..."`)
    void set_source_filename(const std::string& sf) { source_filename_ = sf; }
    const std::string& source_filename() const { return source_filename_; }

    // Module flags (from `!llvm.module.flags`)
    void add_module_flag(const ModuleFlag& flag) { module_flags_.push_back(flag); }
    const std::vector<ModuleFlag>& module_flags() const { return module_flags_; }

    // Type context
    TypeContext& type_context() { return type_ctx_; }

    // Named types
    void add_named_type(const std::string& name, std::shared_ptr<Type> ty) {
        named_types_[name] = ty;
    }
    std::shared_ptr<Type> named_type(const std::string& name) const {
        auto it = named_types_.find(name);
        return it != named_types_.end() ? it->second : nullptr;
    }

    // Statistics
    size_t function_count() const { return functions_.size(); }
    size_t instruction_count() const;

    // Serialization
    std::string to_string() const;

private:
    std::string name_;
    std::vector<std::shared_ptr<Function>> functions_;
    std::unordered_map<std::string, size_t> fn_index_;
    std::vector<GlobalValue> globals_;
    TargetInfo target_;
    std::string source_filename_;
    std::vector<ModuleFlag> module_flags_;
    TypeContext type_ctx_;
    std::unordered_map<std::string, std::shared_ptr<Type>> named_types_;
};

} // namespace clunk::ir
