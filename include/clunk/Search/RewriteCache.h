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
 * Clunk RewriteCache — two-tier persistent rewrite cache.
 *
 * Mirrors Souper's `MemCachingSolver` (in-process LRU) + `ExternalCachingSolver`
 * (file-backed) two-tier design and Minotaur's persistent rewrite cache.
 * The cache maps
 *   (LHS structural hash, RHS structural hash) → VerificationResult
 * so that an SMT query already answered for a given (LHS, RHS) pair is never
 * re-asked, neither within a single process nor across runs.
 *
 *   Cold-cache compile  → warm-cache compile speedup is the dominant
 *   deployability lever for any superoptimiser that uses an SMT verifier
 *   (Souper paper Figure 5: 6×; Minotaur paper Table 3: ~780×).
 *
 * Headlines:
 *  - In-process tier: LRU map (default capacity 4096 entries), O(1) get/put,
 *    thread-safe (multiple readers + writers via a single mutex; lookups are
 *    the hot path and promote the entry to the MRU position).
 *  - File-backed tier: append-only, human-readable, tab-separated, one entry
 *    per line — `lhs_hash\trhs_hash\tstatus\tmessage\ttimestamp_ms`. Loaded
 *    lazily on the first in-memory miss, then indexed in RAM.
 *  - Negative results are cached (status != Equivalent) — most LHS→RHS pairs
 *    don't verify, and caching that fact avoids re-running Z3 on them every
 *    run (Souper §2.12).
 *
 * Thread safety: every public method takes a process-wide mutex. Lookups are
 * the hot path; for the LRU update we mutate `lru_list_` under the lock. The
 * mutex is held briefly — file I/O is done outside the lookup path (only on
 * `put`, `warmup`, `flush`, and the lazy one-shot `load_file_index`).
 *
 * This file provides the RewriteCache class. Entries keyed by structural
 * hashes allow SMT verification results to persist across runs and avoid
 * re-asking the same query.
 */
#include <cstdint>
#include <fstream>
#include <list>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>

#include "clunk/Search/SMTVerifier.h"  // VerificationResult

namespace clunk::search {

// A rewrite cache entry: the verification result for a (LHS, RHS) pair,
// keyed by their structural hashes (`StochasticSearch::structural_hash`).
struct CacheEntry {
    uint64_t lhs_hash = 0;
    uint64_t rhs_hash = 0;
    VerificationResult::Status status = VerificationResult::Unknown;
    std::string message;          // short human-readable note (e.g. Z3 reason)
    int64_t timestamp_ms = 0;     // epoch millis when the entry was added
};

// Two-tier rewrite cache: in-process LRU + optional file-backed persistence.
//
// The in-process tier is a hard-bounded LRU; the file-backed tier is
// append-only and loaded lazily into an in-RAM index on the first lookup
// miss (so a cache that is opened but never queried pays no parse cost).
class RewriteCache {
public:
    explicit RewriteCache(size_t in_memory_capacity = 4096);
    ~RewriteCache();

    RewriteCache(const RewriteCache&) = delete;
    RewriteCache& operator=(const RewriteCache&) = delete;
    RewriteCache(RewriteCache&&) = delete;
    RewriteCache& operator=(RewriteCache&&) = delete;

    // Open a file-backed cache for persistence. Entries are appended on
    // put() (and flush()) and indexed on the first lookup miss via
    // load_file_index(). The file format is one CacheEntry per line,
    // tab-separated:
    //   `lhs_hash\trhs_hash\tstatus\tmessage\ttimestamp_ms`
    // `message` is a tab-free single token stream; embedded tabs/newlines
    // are replaced with single spaces on write to keep the format flat.
    // Returns false if the file can't be opened for append+read.
    bool open_file(const std::string& path);

    // Close the file (no more appends). Called by the destructor.
    void close();

    // Look up a (LHS, RHS) pair. Returns the cached entry if found,
    // nullopt otherwise. Checks the in-process tier first; on miss falls
    // back to the file-backed tier (loading it lazily on first use). On a
    // file-backed hit, the entry is promoted into the in-process tier so
    // subsequent lookups of the same key are O(1) and don't re-parse.
    std::optional<CacheEntry> lookup(uint64_t lhs_hash, uint64_t rhs_hash) const;

    // Insert a (LHS, RHS) pair into the cache. Writes to both tiers.
    // If the in-process tier is at capacity, the LRU entry is evicted.
    // The file-backed tier (if open) is appended-to immediately and
    // flushed, so a crash after `put` does not lose the entry.
    void put(uint64_t lhs_hash, uint64_t rhs_hash, const VerificationResult& result);

    // Bulk-load entries from the file-backed tier into the in-process tier.
    // Useful at startup to warm the cache. Returns the number of entries
    // promoted (capped by in-memory capacity).
    size_t warmup();

    // Save the in-process tier to the file-backed tier (dumps all entries,
    // not just new ones). Returns the number of entries saved. No-op if
    // no file is open.
    size_t flush();

    // Statistics counters (per-cache-instance, never reset).
    struct Stats {
        size_t in_memory_hits = 0;        // lookup found entry in LRU
        size_t file_hits = 0;             // lookup found entry in file index
        size_t misses = 0;                // lookup found nothing
        size_t puts = 0;                  // put() calls
        size_t file_load_errors = 0;      // malformed lines skipped on load
        size_t in_memory_entries = 0;     // current LRU size
        size_t file_entries = 0;          // entries in file_index_ (after load)
    };
    const Stats& stats() const { return stats_; }

private:
    // 128-bit (lhs, rhs) key packed into a 64-bit value via golden-ratio
    // mixing. Collisions are astronomically unlikely for the structural
    // hash (FNV-1a over the opcode+operand stream) and would at worst
    // cause a wrong cache hit — which the SMT verifier downstream would
    // catch (the cache is an optimisation, not a soundness gate).
    static uint64_t make_key(uint64_t lhs_hash, uint64_t rhs_hash) {
        return lhs_hash ^ (rhs_hash * 0x9E3779B97F4A7C15ULL);
    }

    // Lazily parse the file-backed cache into `file_index_`. Idempotent
    // (guarded by `file_index_loaded_`). Called by `lookup` on the first
    // in-memory miss and by `warmup`. Marks `file_index_loaded_ = true`
    // even if the file is missing/empty so we don't retry the parse on
    // every lookup. Thread-safe via `mutex_`.
    void load_file_index() const;

    // Append one entry to `file_out_` (no lock — caller holds `mutex_`).
    void append_line(const CacheEntry& e) const;

    // Promote one entry to the MRU position in the in-process tier
    // (no lock — caller holds `mutex_`).
    void touch_in_memory(std::list<CacheEntry>::iterator it) const;

    mutable std::mutex mutex_;
    size_t capacity_;

    // In-process LRU. `lru_list_` owns the entries (front = MRU).
    // `in_memory_` is the lookup index into the list. The iterator stored
    // in the map remains valid for the lifetime of the entry (std::list
    // iterators are not invalidated by other inserts/erases).
    mutable std::list<CacheEntry> lru_list_;
    mutable std::unordered_map<uint64_t,
                              std::pair<CacheEntry, std::list<CacheEntry>::iterator>>
        in_memory_;

    // File-backed tier.
    std::string file_path_;
    mutable std::ofstream file_out_;
    mutable std::unordered_map<uint64_t, CacheEntry> file_index_;
    mutable bool file_index_loaded_ = false;

    mutable Stats stats_;
};

} // namespace clunk::search
