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
 * Clunk IR Function — a function with basic blocks, arguments, and attributes.
 */
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include "clunk/IR/Type.h"
#include "clunk/IR/BasicBlock.h"

namespace clunk::ir {

enum class Linkage {
    External,
    Internal,
    Private,
    LinkOnce,
    Weak,
    Common,
    Appending,
    ExternalWeak,
    LinkOnceODR,
    WeakODR,
    AvailableExternally
};

enum class Visibility {
    Default,
    Hidden,
    Protected
};

struct Argument {
    std::shared_ptr<Type> type;
    std::string name;
    std::unordered_map<std::string, std::string> attrs; // noalias, nocapture, etc.
};

class Function {
public:
    Function(const std::string& name,
             std::shared_ptr<FunctionType> fn_type,
             Linkage linkage = Linkage::External)
        : name_(name), fn_type_(fn_type), linkage_(linkage) {}

    const std::string& name() const { return name_; }
    // Rename in place. Does NOT fix up call sites elsewhere that reference
    // the old name (mirrors Module::remove_function's contract) — callers
    // are responsible for that. Primarily used to align a candidate's name
    // with the original's before an external tool that matches functions
    // by name across two modules (see AliveVerifier).
    void set_name(const std::string& name) { name_ = name; }
    std::shared_ptr<FunctionType> function_type() const { return fn_type_; }
    std::shared_ptr<Type> return_type() const { return fn_type_->return_type(); }
    Linkage linkage() const { return linkage_; }
    void set_linkage(Linkage l) { linkage_ = l; }

    // Arguments
    void add_argument(std::shared_ptr<Type> type, const std::string& name = "",
                      std::unordered_map<std::string, std::string> attrs = {}) {
        args_.push_back({type, name, std::move(attrs)});
    }
    const std::vector<Argument>& arguments() const { return args_; }
    size_t argument_count() const { return args_.size(); }

    // Basic blocks
    BasicBlock& add_block(const std::string& name) {
        blocks_.push_back(std::make_shared<BasicBlock>(name));
        block_index_[name] = blocks_.size() - 1;
        return *blocks_.back();
    }

    std::shared_ptr<BasicBlock> block(const std::string& name) const {
        auto it = block_index_.find(name);
        if (it == block_index_.end()) return nullptr;
        return blocks_.at(it->second);
    }

    std::shared_ptr<BasicBlock> entry_block() const {
        return blocks_.empty() ? nullptr : blocks_.front();
    }

    const std::vector<std::shared_ptr<BasicBlock>>& blocks() const { return blocks_; }
    std::vector<std::shared_ptr<BasicBlock>>& blocks() { return blocks_; }

    // Attributes (key-value, e.g. kernel)
    void set_attribute(const std::string& key, const std::string& val) {
        attrs_[key] = val;
    }
    const std::unordered_map<std::string, std::string>& attributes() const { return attrs_; }

    // Function attribute strings (nounwind, noinline, #0, etc.)
    void add_function_attribute(const std::string& attr) { function_attributes_.push_back(attr); }
    const std::vector<std::string>& function_attributes() const { return function_attributes_; }

    // Build predecessor info for all blocks
    void compute_predecessors();

    // ── Rebuild block index ───────────────────────────────────────────
    // Reconstructs the name->index map from the current blocks_ vector.
    // Call this after directly mutating blocks_ (e.g. inserting blocks
    // out-of-order). add_block() maintains the index automatically, so
    // this is only needed when callers bypass add_block().
    void rebuild_block_index() {
        block_index_.clear();
        for (size_t i = 0; i < blocks_.size(); ++i) {
            if (blocks_[i]) block_index_[blocks_[i]->name()] = i;
        }
    }

    // Check if this is a GPU kernel
    bool is_gpu_kernel() const {
        return attrs_.count("kernel") || attrs_.count("nvptx-kernel");
    }

    // Remove a block by name (no-op if it doesn't exist). Does NOT fix up
    // references to it (branch targets / phi incoming-edges elsewhere) —
    // callers that remove blocks are responsible for leaving the function
    // well-formed. See ir::remove_unreachable_blocks() for a caller that
    // does this correctly.
    void remove_block(const std::string& name) {
        auto it = block_index_.find(name);
        if (it == block_index_.end()) return;
        size_t idx = it->second;
        blocks_.erase(blocks_.begin() + static_cast<ptrdiff_t>(idx));
        block_index_.erase(it);
        for (auto& [k, v] : block_index_) {
            if (v > idx) --v;
        }
    }

    // Instruction count
    size_t instruction_count() const;

    std::string to_string() const;

private:
    std::string name_;
    std::shared_ptr<FunctionType> fn_type_;
    Linkage linkage_;
    Visibility visibility_ = Visibility::Default;
    std::vector<Argument> args_;
    std::vector<std::shared_ptr<BasicBlock>> blocks_;
    std::unordered_map<std::string, size_t> block_index_;
    std::unordered_map<std::string, std::string> attrs_;
    std::vector<std::string> function_attributes_;
};

} // namespace clunk::ir
