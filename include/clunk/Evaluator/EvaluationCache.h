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
 * Clunk EvaluationCache — thread-safe LRU memoisation for the evaluator.
 *
 * The cache is keyed by a 64-bit structural hash of an ir::Function
 * (see structural_hash()). Two functions with identical opcode streams,
 * operand types, and basic-block structure produce the same hash, so
 * candidates that the search has already seen are O(1) to re-score.
 *
 * The cache is templated on the value type (typically
 * std::shared_ptr<const FunctionAnalysis>). Thread-safety is provided
 * by a std::shared_mutex: multiple concurrent readers, exclusive writer.
 *
 * The cache is bounded by a configurable capacity (default 1024 entries)
 * with LRU eviction. Hits/misses/evictions are tracked atomically for
 * observability.
 */
#include <atomic>
#include <cstdint>
#include <list>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>

#include "clunk/IR/Function.h"

namespace clunk::evaluator {

// ── Structural hash of an IR function ─────────────────────────────────────
//
// The hash captures the function's opcode stream, operand *types*
// (not operand names — semantically equivalent renames should hit the
// cache), and basic-block structure (count + order). It is *not* a
// canonicalisation: two functions differing only in operand names will
// hash the same; two functions with the same opcode stream but different
// successor structure will hash differently (because the block count
// differs).
//
// The algorithm is a 64-bit FNV-1a over a canonicalised textual
// representation. FNV-1a is not cryptographically secure but it is
// fast, dependency-free, and good enough for hash-table keying of IR.
uint64_t structural_hash(const ir::Function& fn);

// ── Cache statistics ──────────────────────────────────────────────────────
struct CacheStats {
    size_t hits       = 0;
    size_t misses     = 0;
    size_t evictions  = 0;
    size_t size       = 0;
    size_t capacity   = 0;
};

// ── Thread-safe bounded LRU cache ─────────────────────────────────────────
template <typename V>
class EvaluationCache {
public:
    explicit EvaluationCache(size_t capacity = 1024)
        : capacity_(capacity == 0 ? 1 : capacity) {}

    ~EvaluationCache() = default;

    EvaluationCache(const EvaluationCache&) = delete;
    EvaluationCache& operator=(const EvaluationCache&) = delete;

    // Lookup by structural-hash key. Returns nullptr on miss.
    // On hit, the entry is promoted to most-recently-used.
    std::shared_ptr<const V> get(uint64_t key) {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        auto it = index_.find(key);
        if (it == index_.end()) {
            misses_.fetch_add(1, std::memory_order_relaxed);
            return nullptr;
        }
        // Promote to front (most-recently-used).
        lru_.splice(lru_.begin(), lru_, it->second);
        hits_.fetch_add(1, std::memory_order_relaxed);
        return it->second->second;
    }

    // Insert (or overwrite) a value. Evicts the LRU entry if at capacity.
    void put(uint64_t key, std::shared_ptr<const V> value) {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        auto it = index_.find(key);
        if (it != index_.end()) {
            // Overwrite: update value and promote.
            it->second->second = std::move(value);
            lru_.splice(lru_.begin(), lru_, it->second);
            return;
        }
        // New entry.
        lru_.emplace_front(key, std::move(value));
        index_[key] = lru_.begin();
        if (lru_.size() > capacity_) {
            auto victim = std::prev(lru_.end());
            index_.erase(victim->first);
            lru_.pop_back();
            evictions_.fetch_add(1, std::memory_order_relaxed);
        }
    }

    void clear() {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        lru_.clear();
        index_.clear();
    }

    CacheStats stats() const {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        CacheStats s;
        s.hits      = hits_.load(std::memory_order_relaxed);
        s.misses    = misses_.load(std::memory_order_relaxed);
        s.evictions = evictions_.load(std::memory_order_relaxed);
        s.size      = lru_.size();
        s.capacity  = capacity_;
        return s;
    }

    size_t capacity() const { return capacity_; }

    void set_capacity(size_t new_capacity) {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        capacity_ = new_capacity == 0 ? 1 : new_capacity;
        while (lru_.size() > capacity_) {
            auto victim = std::prev(lru_.end());
            index_.erase(victim->first);
            lru_.pop_back();
            evictions_.fetch_add(1, std::memory_order_relaxed);
        }
    }

private:
    using ListIter = typename std::list<std::pair<uint64_t, std::shared_ptr<const V>>>::iterator;

    size_t capacity_;
    mutable std::shared_mutex mutex_;
    std::list<std::pair<uint64_t, std::shared_ptr<const V>>> lru_;  // front = MRU
    std::unordered_map<uint64_t, ListIter> index_;
    mutable std::atomic<size_t> hits_{0};
    mutable std::atomic<size_t> misses_{0};
    mutable std::atomic<size_t> evictions_{0};
};

} // namespace clunk::evaluator
