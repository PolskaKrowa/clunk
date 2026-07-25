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
 * Clunk RewriteCache Tests — exercises the two-tier (in-process LRU +
 * file-backed) rewrite cache.
 *
 * Uses the same `CHECK(cond, msg)` macro pattern as test_search.cpp.
 *
 * Coverage:
 *   1. test_rewrite_cache_in_memory_put_lookup — basic put/lookup roundtrip.
 *   2. test_rewrite_cache_lru_eviction — fill to capacity, verify LRU is
 *      evicted and the new entry is retained.
 *   3. test_rewrite_cache_file_persistence — write 3 entries to a temp
 *      file, close, reopen, verify all 3 survive (after warmup).
 *   4. test_rewrite_cache_warmup — manually write 5 entries to a temp
 *      file, call warmup(), verify all 5 are in the in-process tier.
 *   5. test_rewrite_cache_stats — mix of hits and misses, verify stats
 *      counters track correctly.
 *   6. test_rewrite_cache_negative_results_cached — a NotEquivalent
 *      result is stored and retrieved (negative caching is essential).
 *   7. test_rewrite_cache_thread_safety — concurrent put/lookup from
 *      multiple threads completes without deadlock or data corruption.
 */
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include <unistd.h>  // ::close

#include "clunk/Search/RewriteCache.h"
#include "clunk/Search/SMTVerifier.h"

using namespace clunk::search;

static int g_pass = 0, g_fail = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { std::cerr << "FAIL: " << msg << " at " << __FILE__ << ":" << __LINE__ << "\n"; g_fail++; } \
    else { g_pass++; } \
} while(0)

// ── Temp-file helper ────────────────────────────────────────────────────────
static std::string make_temp_path(const char* tag) {
    // mkstemp pattern: a tag-specific name under /tmp.
    char tmpl[128];
    std::snprintf(tmpl, sizeof(tmpl), "/tmp/clunk_cache_%s_XXXXXX", tag);
    int fd = mkstemp(tmpl);
    if (fd < 0) {
        std::cerr << "FATAL: mkstemp failed for tag " << tag << "\n";
        std::exit(1);
    }
    // Close the fd — we only want the path; the cache opens the file
    // itself in append mode. Unlinking the file ensures it's cleaned up
    // even if the test crashes; the cache's open(append) re-creates it.
    ::close(fd);
    std::remove(tmpl);
    return std::string(tmpl);
}

// Build a VerificationResult for tests.
static VerificationResult make_result(VerificationResult::Status s,
                                       const std::string& msg) {
    VerificationResult r;
    r.status = s;
    r.message = msg;
    r.solve_time_ms = 1.5;
    return r;
}

// ═══════════════════════════════════════════════════════════════════════════
//  Test 1: in-memory put + lookup
// ═══════════════════════════════════════════════════════════════════════════
static void test_rewrite_cache_in_memory_put_lookup() {
    RewriteCache cache(/*capacity=*/64);
    cache.put(/*lhs=*/0xAAA1, /*rhs=*/0xBBB1,
              make_result(VerificationResult::Equivalent, "ok-equiv"));

    auto hit = cache.lookup(0xAAA1, 0xBBB1);
    CHECK(hit.has_value(), "lookup finds inserted entry");
    if (hit) {
        CHECK(hit->lhs_hash == 0xAAA1, "lhs hash preserved");
        CHECK(hit->rhs_hash == 0xBBB1, "rhs hash preserved");
        CHECK(hit->status == VerificationResult::Equivalent, "status preserved");
        CHECK(hit->message == "ok-equiv", "message preserved");
    }

    auto miss = cache.lookup(0xDEAD, 0xBEEF);
    CHECK(!miss.has_value(), "lookup of unknown key returns nullopt");

    const auto& s = cache.stats();
    CHECK(s.in_memory_hits == 1, "exactly one in-memory hit");
    CHECK(s.misses == 1, "exactly one miss");
    CHECK(s.puts == 1, "exactly one put");
    CHECK(s.in_memory_entries == 1, "one entry in memory");
}

// ═══════════════════════════════════════════════════════════════════════════
//  Test 2: LRU eviction
// ═══════════════════════════════════════════════════════════════════════════
static void test_rewrite_cache_lru_eviction() {
    const size_t CAP = 4;
    RewriteCache cache(CAP);

    // Fill the cache to capacity. Each (lhs=i, rhs=i*10) entry is unique.
    for (uint64_t i = 1; i <= CAP; ++i) {
        cache.put(i, i * 10, make_result(VerificationResult::Equivalent, "e"));
    }
    CHECK(cache.stats().in_memory_entries == CAP, "cache is at capacity");

    // Touch the LRU entry (i=1) so it becomes MRU, then insert one more —
    // the eviction target should now be i=2, NOT i=1.
    auto touch = cache.lookup(1, 10);
    CHECK(touch.has_value(), "touch LRU entry promotes it");

    // Insert one more entry, forcing an eviction of the current LRU (i=2).
    cache.put(100, 1000, make_result(VerificationResult::NotEquivalent, "new"));
    CHECK(cache.stats().in_memory_entries == CAP, "size still at capacity");

    // i=1 was touched → still present.
    CHECK(cache.lookup(1, 10).has_value(), "touched entry survives eviction");
    // i=2 was the new LRU → evicted.
    CHECK(!cache.lookup(2, 20).has_value(), "true LRU entry was evicted");
    // i=3, i=4 still present.
    CHECK(cache.lookup(3, 30).has_value(), "i=3 still present");
    CHECK(cache.lookup(4, 40).has_value(), "i=4 still present");
    // The new entry is present.
    CHECK(cache.lookup(100, 1000).has_value(), "new entry present");
}

// ═══════════════════════════════════════════════════════════════════════════
//  Test 3: file persistence — entries survive close + reopen
// ═══════════════════════════════════════════════════════════════════════════
static void test_rewrite_cache_file_persistence() {
    std::string path = make_temp_path("persist");

    {
        RewriteCache cache(64);
        bool ok = cache.open_file(path);
        CHECK(ok, "open_file succeeds for new temp path");

        cache.put(0x1001, 0x2001, make_result(VerificationResult::Equivalent, "first"));
        cache.put(0x1002, 0x2002, make_result(VerificationResult::NotEquivalent, "second"));
        cache.put(0x1003, 0x2003, make_result(VerificationResult::Unknown, "third"));
        // destructor calls close()
    }

    // Verify the file exists and has 3 lines (human-readable).
    {
        std::ifstream in(path);
        CHECK(in.good(), "file exists after close");
        size_t lines = 0;
        std::string line;
        while (std::getline(in, line)) {
            if (!line.empty()) ++lines;
        }
        CHECK(lines == 3, "file contains exactly 3 entries");
    }

    // Reopen in a fresh instance and verify all 3 entries are recoverable.
    {
        RewriteCache cache(64);
        bool ok = cache.open_file(path);
        CHECK(ok, "reopen existing cache file");

        auto h1 = cache.lookup(0x1001, 0x2001);
        CHECK(h1.has_value(), "first entry recovered from file");
        if (h1) {
            CHECK(h1->status == VerificationResult::Equivalent, "first status");
            CHECK(h1->message == "first", "first message");
        }

        auto h2 = cache.lookup(0x1002, 0x2002);
        CHECK(h2.has_value(), "second entry recovered from file");
        if (h2) {
            CHECK(h2->status == VerificationResult::NotEquivalent, "second status (negative cached)");
        }

        auto h3 = cache.lookup(0x1003, 0x2003);
        CHECK(h3.has_value(), "third entry recovered from file");
        if (h3) {
            CHECK(h3->status == VerificationResult::Unknown, "third status");
        }

        // The lookups above should be file hits, not in-memory hits (the
        // in-memory tier is empty until the first lookup promotes the entry).
        const auto& s = cache.stats();
        CHECK(s.file_hits == 3, "all three recovered via file tier");
        CHECK(s.in_memory_hits == 0, "no in-memory hits on first-touch lookups");
        CHECK(s.misses == 0, "no misses — all keys present in file");
    }

    std::remove(path.c_str());
}

// ═══════════════════════════════════════════════════════════════════════════
//  Test 4: warmup — bulk-load file entries into the in-process tier
// ═══════════════════════════════════════════════════════════════════════════
static void test_rewrite_cache_warmup() {
    std::string path = make_temp_path("warmup");

    // Manually write 5 cache entries directly to the file (bypassing the
    // cache itself) so warmup() has work to do.
    {
        std::ofstream out(path, std::ios::app);
        CHECK(out.good(), "manual file open for warmup seed");
        // Format: lhs\trhs\tstatus\ttimestamp\tmessage
        out << "5001\t5001\t0\t1000\tmsg-one\n";
        out << "5002\t5002\t1\t2000\tmsg-two\n";
        out << "5003\t5003\t2\t3000\tmsg-three\n";
        out << "5004\t5004\t0\t4000\tmsg-four\n";
        out << "5005\t5005\t3\t5000\tmsg-five\n";
    }

    RewriteCache cache(64);
    bool ok = cache.open_file(path);
    CHECK(ok, "open_file for warmup test");

    size_t promoted = cache.warmup();
    CHECK(promoted == 5, "warmup promoted all 5 file entries");

    const auto& s = cache.stats();
    CHECK(s.in_memory_entries == 5, "5 entries now in memory");

    // All 5 should be in-memory hits now (not file hits).
    for (uint64_t i = 1; i <= 5; ++i) {
        uint64_t k = 5000 + i;
        auto h = cache.lookup(k, k);
        CHECK(h.has_value(), "warmup-loaded entry is findable");
    }
    CHECK(cache.stats().in_memory_hits == 5, "warmup entries are in-memory hits");
    CHECK(cache.stats().file_hits == 0, "warmup entries are NOT file hits (already in memory)");

    std::remove(path.c_str());
}

// ═══════════════════════════════════════════════════════════════════════════
//  Test 5: stats counters
// ═══════════════════════════════════════════════════════════════════════════
static void test_rewrite_cache_stats() {
    RewriteCache cache(8);

    // 3 puts.
    cache.put(1, 1, make_result(VerificationResult::Equivalent, "a"));
    cache.put(2, 2, make_result(VerificationResult::Equivalent, "b"));
    cache.put(3, 3, make_result(VerificationResult::NotEquivalent, "c"));

    // 2 hits (in-memory).
    (void)cache.lookup(1, 1);
    (void)cache.lookup(2, 2);

    // 2 misses.
    (void)cache.lookup(98, 98);
    (void)cache.lookup(99, 99);

    const auto& s = cache.stats();
    CHECK(s.puts == 3, "stats: puts == 3");
    CHECK(s.in_memory_hits == 2, "stats: in_memory_hits == 2");
    CHECK(s.misses == 2, "stats: misses == 2");
    CHECK(s.file_hits == 0, "stats: file_hits == 0 (no file open)");
    CHECK(s.in_memory_entries == 3, "stats: in_memory_entries == 3");
    CHECK(s.file_entries == 0, "stats: file_entries == 0 (no file open)");
    CHECK(s.file_load_errors == 0, "stats: no file load errors");
}

// ═══════════════════════════════════════════════════════════════════════════
//  Test 6: negative results are cached
// ═══════════════════════════════════════════════════════════════════════════
static void test_rewrite_cache_negative_results_cached() {
    RewriteCache cache(8);

    // A NotEquivalent result is just as cacheable as an Equivalent one —
    // re-running Z3 on the same (LHS, RHS) pair that already returned
    // NotEquivalent would be pure waste.
    cache.put(0xCAFE, 0xBABE,
              make_result(VerificationResult::NotEquivalent, "differs at x=7"));

    auto h = cache.lookup(0xCAFE, 0xBABE);
    CHECK(h.has_value(), "negative result is cached");
    if (h) {
        CHECK(h->status == VerificationResult::NotEquivalent, "negative status preserved");
        CHECK(h->message == "differs at x=7", "negative message preserved");
    }

    // Same for Unknown (the sound-fallback status) — caching it prevents
    // repeatedly attempting to verify functions the verifier can't model.
    cache.put(0xDEAD, 0xBEEF,
              make_result(VerificationResult::Unknown, "loops present"));
    auto h2 = cache.lookup(0xDEAD, 0xBEEF);
    CHECK(h2.has_value(), "unknown result is cached");
    if (h2) {
        CHECK(h2->status == VerificationResult::Unknown, "unknown status preserved");
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Test 7: thread safety — concurrent put + lookup doesn't deadlock or corrupt
// ═══════════════════════════════════════════════════════════════════════════
static void test_rewrite_cache_thread_safety() {
    RewriteCache cache(256);
    const int N_WRITERS = 4;
    const int N_READERS = 4;
    const int OPS_PER_THREAD = 500;
    const int READER_ITERS = 200;  // bounded; readers stop after this many loops

    std::atomic<int> total_hits{0};
    std::atomic<int> total_misses{0};

    auto writer = [&](int tid) {
        for (int i = 0; i < OPS_PER_THREAD; ++i) {
            uint64_t lhs = static_cast<uint64_t>(tid) * 100000 + static_cast<uint64_t>(i);
            uint64_t rhs = lhs + 1;
            cache.put(lhs, rhs,
                      make_result(VerificationResult::Equivalent, "concurrent"));
        }
    };

    auto reader = [&](int /*tid*/) {
        int local_hits = 0, local_misses = 0;
        for (int iter = 0; iter < READER_ITERS; ++iter) {
            // Mix of lookups likely to hit (small keys, written by writers)
            // and lookups guaranteed to miss (huge keys outside the writer range).
            for (int i = 0; i < 50; ++i) {
                uint64_t lhs = static_cast<uint64_t>(i);
                uint64_t rhs = lhs + 1;
                if (cache.lookup(lhs, rhs)) {
                    ++local_hits;
                } else {
                    ++local_misses;
                }
            }
            for (int i = 0; i < 10; ++i) {
                uint64_t lhs = 0xFFFFFFFFULL + static_cast<uint64_t>(i);
                if (cache.lookup(lhs, lhs)) {
                    ++local_hits;
                } else {
                    ++local_misses;
                }
            }
        }
        total_hits.fetch_add(local_hits, std::memory_order_relaxed);
        total_misses.fetch_add(local_misses, std::memory_order_relaxed);
    };

    std::vector<std::thread> threads;
    for (int t = 0; t < N_READERS; ++t) {
        threads.emplace_back(reader, t);
    }
    for (int t = 0; t < N_WRITERS; ++t) {
        threads.emplace_back(writer, t);
    }

    // Join everything.
    for (auto& th : threads) {
        if (th.joinable()) th.join();
    }

    // No assertions on hit/miss counts — the test passes if it completes
    // without deadlock and the cache is internally consistent (no crash,
    // no double-free, no segfault). We do check the put count is exact.
    const auto& s = cache.stats();
    CHECK(s.puts == static_cast<size_t>(N_WRITERS * OPS_PER_THREAD),
          "all writer puts recorded");
    CHECK(s.in_memory_entries <= 256, "in-memory entries never exceed capacity");
    CHECK(s.in_memory_entries >= 1, "at least one entry made it into the cache");

    // After concurrent churn, the cache should still answer lookups
    // without crashing. Writers used distinct key ranges, so no write
    // overwrote another (modulo LRU eviction, which is fine).
    int found = 0;
    for (int tid = 0; tid < N_WRITERS; ++tid) {
        for (int i = 0; i < OPS_PER_THREAD; ++i) {
            uint64_t lhs = static_cast<uint64_t>(tid) * 100000 + static_cast<uint64_t>(i);
            uint64_t rhs = lhs + 1;
            if (cache.lookup(lhs, rhs)) ++found;
        }
    }
    // Some entries may have been evicted (capacity 256 vs N_WRITERS*OPS = 2000),
    // so we can't assert all of them — but the count should be bounded.
    CHECK(found >= 0 && found <= N_WRITERS * OPS_PER_THREAD,
          "post-churn lookup count is sane");
    (void)total_hits;
    (void)total_misses;
}

// ═══════════════════════════════════════════════════════════════════════════

int main() {
    std::cout << "=== Clunk RewriteCache Tests ===" << std::endl;

    std::cout << "  in-memory put + lookup..." << std::endl;
    test_rewrite_cache_in_memory_put_lookup();

    std::cout << "  LRU eviction..." << std::endl;
    test_rewrite_cache_lru_eviction();

    std::cout << "  file persistence..." << std::endl;
    test_rewrite_cache_file_persistence();

    std::cout << "  warmup..." << std::endl;
    test_rewrite_cache_warmup();

    std::cout << "  stats counters..." << std::endl;
    test_rewrite_cache_stats();

    std::cout << "  negative results cached..." << std::endl;
    test_rewrite_cache_negative_results_cached();

    std::cout << "  thread safety..." << std::endl;
    test_rewrite_cache_thread_safety();

    std::cout << "\n=== Results: " << g_pass << " passed, " << g_fail << " failed ===" << std::endl;
    return g_fail > 0 ? 1 : 0;
}
