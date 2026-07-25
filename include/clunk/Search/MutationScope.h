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
 * Clunk Search MutationScope — RAII transactional guard for in-place
 * IR mutations.
 *
 * Problem: StochasticSearch::apply_mutation previously deep-copied the
 * entire Function on every mutation attempt (~10 000 clones per search).
 * Most mutations are rejected by the SA acceptance criterion, so the
 * clone-and-discard dominates runtime.
 *
 * Solution: apply the mutation IN PLACE on the live Function, with a
 * MutationScope guarding the affected instructions. If the caller
 * rejects the candidate, the scope's destructor restores the original
 * instruction shared_ptrs (and any structural changes — block name
 * swaps, insertions / removals) exactly. If the caller accepts, it
 * calls scope.commit() before the scope destructs, and the changes
 * stay.
 *
 * Supported undo operations (recorded by the mutation helpers):
 *   - replace_instruction(pos): save the old shared_ptr<Instruction>.
 *   - remove_instruction(pos): save the old shared_ptr<Instruction> +
 *     its position. Undo re-inserts at the original pos.
 *   - insert_instruction(pos, inst): save the pos. Undo erases the
 *     instruction currently at that pos (which will be the one we
 *     inserted, assuming no intervening edits).
 *   - clear_block_instructions(): save the old vector. Undo restores it.
 *
 * The scope is tied to a single BasicBlock. Mutations that span blocks
 * or that fundamentally restructure the Function (e.g. crossover) should
 * NOT use this scope — they should use the deep_copy_function fallback
 * in clunk/IR/Clone.h.
 *
 * Thread-safety: a MutationScope is NOT thread-safe. Each thread must
 * have its own scope (and its own Function to mutate).
 */
#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "clunk/IR/BasicBlock.h"
#include "clunk/IR/Function.h"
#include "clunk/IR/Instruction.h"

namespace clunk::search {

class MutationScope {
public:
    explicit MutationScope(std::shared_ptr<ir::Function> fn)
        : fn_(std::move(fn)), committed_(false) {}

    ~MutationScope() {
        if (!committed_ && fn_) {
            undo_all();
        }
    }

    MutationScope(const MutationScope&) = delete;
    MutationScope& operator=(const MutationScope&) = delete;
    MutationScope(MutationScope&&) = delete;
    MutationScope& operator=(MutationScope&&) = delete;

    // Commit the mutation — the destructor will NOT undo.
    void commit() noexcept { committed_ = true; }

    // ── Recording helpers ──────────────────────────────────────────────

    // Record a replace_instruction at `pos` in `block_name`. The caller
    // must have ALREADY performed the replace (we save the old pointer
    // here). Returns false if the block was not found.
    bool record_replace(const std::string& block_name,
                         std::size_t pos,
                         std::shared_ptr<ir::Instruction> old_inst) {
        auto bb = fn_->block(block_name);
        if (!bb) return false;
        actions_.push_back(Action{ActionType::Replace, block_name, pos, std::move(old_inst), {}});
        return true;
    }

    // Record a remove_instruction at `pos` in `block_name`. The caller
    // must have ALREADY performed the remove and pass the removed
    // instruction pointer.
    bool record_remove(const std::string& block_name,
                        std::size_t pos,
                        std::shared_ptr<ir::Instruction> removed_inst) {
        auto bb = fn_->block(block_name);
        if (!bb) return false;
        actions_.push_back(Action{ActionType::Remove, block_name, pos, std::move(removed_inst), {}});
        return true;
    }

    // Record an insert_instruction at `pos` in `block_name`. The caller
    // must have ALREADY performed the insert. Undo will erase the
    // instruction at `pos`.
    bool record_insert(const std::string& block_name, std::size_t pos) {
        auto bb = fn_->block(block_name);
        if (!bb) return false;
        actions_.push_back(Action{ActionType::Insert, block_name, pos, nullptr, {}});
        return true;
    }

    // Record a full clear of a block's instructions. The caller must
    // have ALREADY cleared the block. Undo restores the saved vector.
    bool record_clear(const std::string& block_name,
                       std::vector<std::shared_ptr<ir::Instruction>> old_insts) {
        auto bb = fn_->block(block_name);
        if (!bb) return false;
        actions_.push_back(Action{ActionType::Clear, block_name, 0, nullptr, std::move(old_insts)});
        return true;
    }

    // Access the underlying function (for the mutation helpers to edit).
    ir::Function& function() { return *fn_; }
    const ir::Function& function() const { return *fn_; }

    std::size_t action_count() const noexcept { return actions_.size(); }

private:
    enum class ActionType : uint8_t {
        Replace,
        Remove,
        Insert,
        Clear,
    };

    struct Action {
        ActionType type;
        std::string block_name;
        std::size_t pos;
        std::shared_ptr<ir::Instruction> saved_inst;
        std::vector<std::shared_ptr<ir::Instruction>> saved_vec;  // for Clear only
    };

    void undo_all() {
        // Undo in reverse order so positions remain valid.
        for (auto it = actions_.rbegin(); it != actions_.rend(); ++it) {
            auto bb = fn_->block(it->block_name);
            if (!bb) continue;
            switch (it->type) {
                case ActionType::Replace: {
                    if (it->pos < bb->size()) {
                        bb->replace_instruction(it->pos, it->saved_inst);
                    }
                    break;
                }
                case ActionType::Remove: {
                    // Re-insert at the original pos.
                    bb->insert_instruction(it->pos, it->saved_inst);
                    break;
                }
                case ActionType::Insert: {
                    // Erase whatever is now at pos (which should be the
                    // instruction we inserted, assuming no intervening
                    // edits at this same pos).
                    if (it->pos < bb->size()) {
                        bb->remove_instruction(it->pos);
                    }
                    break;
                }
                case ActionType::Clear: {
                    bb->instructions() = std::move(it->saved_vec);
                    break;
                }
            }
        }
        actions_.clear();
    }

    std::shared_ptr<ir::Function> fn_;
    std::vector<Action> actions_;
    bool committed_;
};

} // namespace clunk::search
