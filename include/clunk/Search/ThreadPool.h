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
 * Clunk Search ThreadPool — a minimal header-only C++17 thread pool.
 *
 * Design:
 *   - Fixed number of worker threads (default = hardware_concurrency,
 *     clamped to [1, 32]). Workers are created in the constructor and
 *     joined in the destructor (graceful shutdown).
 *   - Thread-safe task queue: mutex + condition_variable. Lock-free is
 *     overkill for population evaluation (at most a few hundred tasks).
 *   - submit(fn) -> std::future<T>: wraps fn in a packaged_task, enqueues
 *     it, returns the future. Works with any callable and return type.
 *   - Destructor sets a shutdown flag, notifies all workers, and joins
 *     them. Pending tasks are NOT guaranteed to run; callers that need
 *     completion must wait on their futures before the pool destructs.
 *
 * The pool is intentionally simple — no priorities, no cancellation, no
 * dynamic resizing. For the evolutionary search's evaluate_population,
 * we submit N analysis tasks and wait on all futures, which is exactly
 * the pattern this pool supports efficiently.
 */
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace clunk::search {

class ThreadPool {
public:
    explicit ThreadPool(std::size_t num_threads = 0)
        : shutdown_(false) {
        if (num_threads == 0) {
            num_threads = std::thread::hardware_concurrency();
            if (num_threads == 0) num_threads = 1;
        }
        // Clamp to a sane upper bound to avoid oversubscription on big
        // boxes and to keep the population-evaluation benchmark honest.
        if (num_threads > 32) num_threads = 32;
        if (num_threads < 1) num_threads = 1;

        workers_.reserve(num_threads);
        for (std::size_t i = 0; i < num_threads; ++i) {
            workers_.emplace_back([this] { worker_loop(); });
        }
    }

    ~ThreadPool() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            shutdown_ = true;
        }
        cv_.notify_all();
        for (auto& w : workers_) {
            if (w.joinable()) w.join();
        }
    }

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;
    ThreadPool(ThreadPool&&) = delete;
    ThreadPool& operator=(ThreadPool&&) = delete;

    // Submit a callable. Returns a future for the callable's result.
    template <typename F, typename... Args>
    auto submit(F&& f, Args&&... args) -> std::future<
        typename std::invoke_result<F, Args...>::type>
    {
        using R = typename std::invoke_result<F, Args...>::type;

        // Bind arguments into a zero-arg callable. We use shared_ptr to
        // share the packaged_task between the returned future and the
        // queue (std::function requires CopyConstructible, and
        // packaged_task is move-only).
        auto task = std::make_shared<std::packaged_task<R()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...));
        std::future<R> fut = task->get_future();

        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (shutdown_) {
                // Pool is shutting down — drop the task. The future will
                // never be satisfied; caller should check is_shutdown()
                // before submitting in shutdown paths.
                return fut;
            }
            tasks_.emplace([task]() { (*task)(); });
        }
        cv_.notify_one();
        return fut;
    }

    std::size_t worker_count() const noexcept {
        return workers_.size();
    }

    bool is_shutdown() const noexcept {
        // Best-effort snapshot; do not rely on this for synchronisation.
        return shutdown_;
    }

    // Number of tasks currently queued (not including in-flight). Best-effort.
    std::size_t queued_task_count() {
        std::lock_guard<std::mutex> lock(mutex_);
        return tasks_.size();
    }

private:
    void worker_loop() {
        while (true) {
            std::function<void()> task;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_.wait(lock, [this] {
                    return shutdown_ || !tasks_.empty();
                });
                if (shutdown_ && tasks_.empty()) return;
                if (!tasks_.empty()) {
                    task = std::move(tasks_.front());
                    tasks_.pop();
                }
            }
            if (task) task();
        }
    }

    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::queue<std::function<void()>> tasks_;
    std::vector<std::thread> workers_;
    bool shutdown_;
};

} // namespace clunk::search
