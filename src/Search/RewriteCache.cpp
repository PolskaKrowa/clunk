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
 * Clunk RewriteCache — implementation. See RewriteCache.h for the design.
 *
 * Two-tier persistent cache. The in-process tier is
 * an LRU (std::list + unordered_map, O(1) get/put); the file-backed tier is
 * an append-only tab-separated text file, lazily parsed on the first
 * in-memory miss.
 *
 * File format (one entry per line, tab-separated, human-readable):
 *   lhs_hash\trhs_hash\tstatus\tmessage\ttimestamp_ms
 *
 * `status` is serialised as the integer enum value (0..3) for compactness
 * and unambiguous parsing. `message` is a free-form string with embedded
 * tabs/newlines replaced by single spaces so the line stays flat.
 *
 * Thread safety: every public method takes `mutex_`. The hot path
 * (`lookup`) holds the lock for the duration of the LRU probe + (on miss)
 * the file-index probe. The file is opened once and kept open for the
 * lifetime of the cache; `put` appends under the lock and flushes, so a
 * crash after `put` does not lose data.
 */
#include "clunk/Search/RewriteCache.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <utility>

namespace clunk::search {

namespace {

// Replace any tab or newline in `s` with a single space so the serialised
// line stays a single tab-separated record.
std::string sanitise_message(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        if (c == '\t' || c == '\n' || c == '\r') {
            out.push_back(' ');
        } else {
            out.push_back(c);
        }
    }
    return out;
}

int64_t now_ms() {
    using namespace std::chrono;
    return duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
}

// Parse one tab-separated cache line into `out`. Returns false on any
// parse error (caller should bump file_load_errors and skip the line).
bool parse_line(const std::string& line, CacheEntry& out) {
    // Fields: lhs_hash \t rhs_hash \t status \t message \t timestamp_ms
    // `message` may itself contain spaces (but not tabs — those were
    // sanitised on write). Split on the first 4 tabs; the remainder is
    // the message body up to the final tab + timestamp.
    std::istringstream iss(line);
    std::string lhs_s, rhs_s, status_s, ts_s;
    if (!std::getline(iss, lhs_s, '\t')) return false;
    if (!std::getline(iss, rhs_s, '\t')) return false;
    if (!std::getline(iss, status_s, '\t')) return false;
    if (!std::getline(iss, ts_s, '\t')) return false;
    // The remainder of the stream is the `message` field (may be empty).
    std::string msg;
    std::getline(iss, msg);

    // All numeric fields must parse cleanly (strtoull with full-consumption
    // check) so a truncated/corrupt line is rejected rather than silently
    // producing a wrong cache entry.
    auto parse_u64 = [](const std::string& s, uint64_t& val) -> bool {
        if (s.empty()) return false;
        errno = 0;
        char* end = nullptr;
        unsigned long long v = std::strtoull(s.c_str(), &end, 10);
        if (errno != 0 || end == s.c_str() || *end != '\0') return false;
        val = static_cast<uint64_t>(v);
        return true;
    };
    auto parse_i64 = [](const std::string& s, int64_t& val) -> bool {
        if (s.empty()) return false;
        errno = 0;
        char* end = nullptr;
        long long v = std::strtoll(s.c_str(), &end, 10);
        if (errno != 0 || end == s.c_str() || *end != '\0') return false;
        val = static_cast<int64_t>(v);
        return true;
    };

    uint64_t lhs = 0, rhs = 0;
    int64_t ts = 0;
    if (!parse_u64(lhs_s, lhs)) return false;
    if (!parse_u64(rhs_s, rhs)) return false;
    if (!parse_i64(ts_s, ts)) return false;

    // Status must be a single-digit integer 0..3 (matches the
    // VerificationResult::Status enum order).
    if (status_s.size() != 1) return false;
    int st = status_s[0] - '0';
    if (st < 0 || st > 3) return false;

    out.lhs_hash = lhs;
    out.rhs_hash = rhs;
    out.status = static_cast<VerificationResult::Status>(st);
    out.message = msg;
    out.timestamp_ms = ts;
    return true;
}

std::string serialise_line(const CacheEntry& e) {
    std::ostringstream oss;
    oss << e.lhs_hash << '\t'
        << e.rhs_hash << '\t'
        << static_cast<int>(e.status) << '\t'
        << e.timestamp_ms << '\t'
        << sanitise_message(e.message);
    return oss.str();
}

} // namespace

RewriteCache::RewriteCache(size_t in_memory_capacity)
    : capacity_(in_memory_capacity == 0 ? 1 : in_memory_capacity) {}

RewriteCache::~RewriteCache() {
    close();
}

bool RewriteCache::open_file(const std::string& path) {
    std::lock_guard<std::mutex> lk(mutex_);
    file_path_ = path;
    if (file_out_.is_open()) {
        file_out_.flush();
        file_out_.close();
    }
    // Open in append mode so existing entries are preserved. We never
    // truncate — the file is the persistent record across runs.
    file_out_.open(path, std::ios::app);
    if (!file_out_.is_open()) {
        return false;
    }
    // Reset the lazy index so the next lookup re-parses the file (it may
    // have grown since the previous load).
    file_index_.clear();
    file_index_loaded_ = false;
    return true;
}

void RewriteCache::close() {
    std::lock_guard<std::mutex> lk(mutex_);
    if (file_out_.is_open()) {
        file_out_.flush();
        file_out_.close();
    }
}

void RewriteCache::load_file_index() const {
    if (file_index_loaded_) return;
    file_index_loaded_ = true;  // set first so a parse failure doesn't retry

    if (file_path_.empty()) return;

    std::ifstream in(file_path_);
    if (!in.is_open()) return;

    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        CacheEntry e;
        if (!parse_line(line, e)) {
            ++stats_.file_load_errors;
            continue;
        }
        // Last-write-wins: a duplicate key keeps the most recent entry.
        file_index_[make_key(e.lhs_hash, e.rhs_hash)] = e;
    }
    stats_.file_entries = file_index_.size();
}

void RewriteCache::append_line(const CacheEntry& e) const {
    if (!file_out_.is_open()) return;
    file_out_ << serialise_line(e) << '\n';
    file_out_.flush();
}

void RewriteCache::touch_in_memory(std::list<CacheEntry>::iterator it) const {
    // Move the entry to the front of the LRU list (MRU position).
    // std::list::splice is O(1) and does NOT invalidate iterators —
    // crucial because `in_memory_` stores iterators into this list.
    lru_list_.splice(lru_list_.begin(), lru_list_, it);
}

std::optional<CacheEntry> RewriteCache::lookup(uint64_t lhs_hash,
                                                uint64_t rhs_hash) const {
    std::lock_guard<std::mutex> lk(mutex_);
    uint64_t key = make_key(lhs_hash, rhs_hash);

    // Tier 1: in-process LRU.
    auto it = in_memory_.find(key);
    if (it != in_memory_.end()) {
        ++stats_.in_memory_hits;
        touch_in_memory(it->second.second);
        return it->second.first;
    }

    // Tier 2: file-backed index (loaded lazily).
    if (!file_index_loaded_) {
        load_file_index();
    }
    auto fit = file_index_.find(key);
    if (fit != file_index_.end()) {
        ++stats_.file_hits;
        // Promote into the in-process tier so future lookups hit tier 1.
        // Insert at MRU; evict LRU if over capacity.
        lru_list_.push_front(fit->second);
        in_memory_[key] = {fit->second, lru_list_.begin()};
        if (in_memory_.size() > capacity_) {
            // Evict the LRU (back of the list).
            auto lru_it = std::prev(lru_list_.end());
            uint64_t lru_key = make_key(lru_it->lhs_hash, lru_it->rhs_hash);
            in_memory_.erase(lru_key);
            lru_list_.pop_back();
        }
        stats_.in_memory_entries = in_memory_.size();
        return fit->second;
    }

    ++stats_.misses;
    return std::nullopt;
}

void RewriteCache::put(uint64_t lhs_hash, uint64_t rhs_hash,
                       const VerificationResult& result) {
    std::lock_guard<std::mutex> lk(mutex_);
    uint64_t key = make_key(lhs_hash, rhs_hash);

    CacheEntry entry;
    entry.lhs_hash = lhs_hash;
    entry.rhs_hash = rhs_hash;
    entry.status = result.status;
    entry.message = result.message.empty() ? result.z3_reason : result.message;
    entry.timestamp_ms = now_ms();

    // Tier 1: insert / update the in-process LRU.
    auto it = in_memory_.find(key);
    if (it != in_memory_.end()) {
        // Update in place; refresh MRU.
        it->second.first = entry;
        // Replace the list node's value too so the list and map stay in sync.
        *it->second.second = entry;
        touch_in_memory(it->second.second);
    } else {
        lru_list_.push_front(entry);
        in_memory_[key] = {entry, lru_list_.begin()};
        if (in_memory_.size() > capacity_) {
            auto lru_it = std::prev(lru_list_.end());
            uint64_t lru_key = make_key(lru_it->lhs_hash, lru_it->rhs_hash);
            in_memory_.erase(lru_key);
            lru_list_.pop_back();
        }
    }
    stats_.in_memory_entries = in_memory_.size();

    // Tier 2: append to the file (and update the file index so the next
    // lookup sees the new entry without re-parsing).
    append_line(entry);
    if (file_index_loaded_) {
        file_index_[key] = entry;
        stats_.file_entries = file_index_.size();
    }

    ++stats_.puts;
}

size_t RewriteCache::warmup() {
    std::lock_guard<std::mutex> lk(mutex_);
    if (!file_index_loaded_) {
        // load_file_index is itself const + idempotent; call it under our lock.
        load_file_index();
    }
    size_t promoted = 0;
    for (const auto& [key, entry] : file_index_) {
        if (in_memory_.count(key)) continue;  // already present
        if (in_memory_.size() >= capacity_) break;  // LRU full
        lru_list_.push_front(entry);
        in_memory_[key] = {entry, lru_list_.begin()};
        ++promoted;
    }
    stats_.in_memory_entries = in_memory_.size();
    return promoted;
}

size_t RewriteCache::flush() {
    std::lock_guard<std::mutex> lk(mutex_);
    if (!file_out_.is_open()) return 0;

    // Dump every in-process entry to the file. We don't deduplicate against
    // the existing file content — the lookup path already treats the file
    // as last-write-wins, so duplicates are harmless (and rare in practice
    // because `put` already appends).
    size_t saved = 0;
    for (const auto& entry : lru_list_) {
        file_out_ << serialise_line(entry) << '\n';
        ++saved;
    }
    file_out_.flush();

    // Refresh the file index so subsequent lookups see the just-saved
    // entries without a re-parse.
    if (file_index_loaded_) {
        for (const auto& entry : lru_list_) {
            file_index_[make_key(entry.lhs_hash, entry.rhs_hash)] = entry;
        }
        stats_.file_entries = file_index_.size();
    }
    return saved;
}

} // namespace clunk::search
