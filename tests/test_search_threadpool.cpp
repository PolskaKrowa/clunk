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
 * Clunk ThreadPool Tests — basic correctness checks for the
 * header-only clunk::search::ThreadPool.
 *
 * Verifies:
 *   - submit() returns a future that resolves to the callable's result.
 *   - Multiple submitted tasks all complete.
 *   - The destructor joins all worker threads (no leak / hang).
 *   - Exception propagation through the future.
 *   - Worker count is honoured.
 */
#include <algorithm>
#include <atomic>
#include <chrono>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

#include "clunk/Search/ThreadPool.h"

static int g_pass = 0, g_fail = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { std::cerr << "FAIL: " << msg << " at " << __FILE__ << ":" << __LINE__ << "\n"; g_fail++; } \
    else { g_pass++; } \
} while(0)

using namespace clunk::search;

void test_basic_submit() {
    ThreadPool pool(2);
    auto f = pool.submit([] { return 42; });
    CHECK(f.get() == 42, "basic submit returns value");
}

void test_worker_count() {
    ThreadPool pool(4);
    CHECK(pool.worker_count() == 4, "pool has 4 workers");
    ThreadPool pool2(1);
    CHECK(pool2.worker_count() == 1, "pool has 1 worker");
}

void test_default_construction() {
    // Default num_threads=0 should pick hardware_concurrency (clamped to [1, 32]).
    ThreadPool pool;
    CHECK(pool.worker_count() >= 1, "default pool has >= 1 worker");
    CHECK(pool.worker_count() <= 32, "default pool has <= 32 workers");
}

void test_many_tasks() {
    ThreadPool pool(4);
    std::vector<std::future<int>> futs;
    for (int i = 0; i < 100; ++i) {
        futs.push_back(pool.submit([i] { return i * i; }));
    }
    int sum = 0;
    for (auto& f : futs) sum += f.get();
    // sum of i^2 for i in [0,100) = 99*100*199/6 = 328350
    CHECK(sum == 328350, "100 tasks all complete with correct sum");
}

void test_exception_propagation() {
    ThreadPool pool(2);
    auto f = pool.submit([]() -> int {
        throw std::runtime_error("boom");
        return 0;
    });
    bool caught = false;
    try {
        (void)f.get();
    } catch (const std::runtime_error&) {
        caught = true;
    }
    CHECK(caught, "exception propagates through future");
}

void test_concurrent_counter() {
    // Verify that tasks actually run on different threads by counting
    // distinct thread IDs.
    ThreadPool pool(4);
    std::mutex mu;
    std::vector<std::thread::id> ids;
    std::vector<std::future<void>> futs;
    for (int i = 0; i < 50; ++i) {
        futs.push_back(pool.submit([&] {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            std::lock_guard<std::mutex> lk(mu);
            ids.push_back(std::this_thread::get_id());
        }));
    }
    for (auto& f : futs) f.get();
    std::sort(ids.begin(), ids.end());
    ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
    // We can't guarantee all 4 workers were used (scheduling), but at
    // least 2 should be — otherwise the pool isn't actually parallel.
    CHECK(ids.size() >= 2, "tasks ran on at least 2 distinct threads");
}

void test_destructor_joins() {
    // If the destructor doesn't join, this will crash or hang under TSan.
    std::thread::id pool_thread;
    {
        ThreadPool pool(2);
        pool_thread = std::this_thread::get_id();
        auto f = pool.submit([&] {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        });
        f.get();
    }
    // If we got here without hanging, the destructor joined cleanly.
    CHECK(true, "destructor joins all workers");
}

void test_clamped_workers() {
    // Requesting more than 32 workers should be clamped to 32.
    ThreadPool pool(1024);
    CHECK(pool.worker_count() == 32, "worker count clamped to 32");
    // Requesting 0 should yield >= 1.
    ThreadPool pool2(0);
    CHECK(pool2.worker_count() >= 1, "0 workers -> >= 1");
}

int main() {
    std::cout << "=== Clunk ThreadPool Tests ===" << std::endl;

    std::cout << "  Basic submit..." << std::endl;
    test_basic_submit();

    std::cout << "  Worker count..." << std::endl;
    test_worker_count();

    std::cout << "  Default construction..." << std::endl;
    test_default_construction();

    std::cout << "  Many tasks..." << std::endl;
    test_many_tasks();

    std::cout << "  Exception propagation..." << std::endl;
    test_exception_propagation();

    std::cout << "  Concurrent counter..." << std::endl;
    test_concurrent_counter();

    std::cout << "  Destructor joins..." << std::endl;
    test_destructor_joins();

    std::cout << "  Clamped workers..." << std::endl;
    test_clamped_workers();

    std::cout << "\n=== Results: " << g_pass << " passed, " << g_fail << " failed ===" << std::endl;
    return g_fail > 0 ? 1 : 0;
}
