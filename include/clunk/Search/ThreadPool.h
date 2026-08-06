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

    // ── Work-stealing helpers (for nested parallelism) ────────────────
    //
    // When a pool worker thread submits subtasks to its OWN pool and then
    // blocks on a future, the pool can deadlock: if all N workers are
    // blocked waiting on subtasks, no worker is left to run them. The
    // standard fix is to let the blocking thread help drain the queue
    // while it waits — so an in-flight worker that submits more work
    // participates in running that work instead of idling.
    //
    // try_run_one(): pop and run at most one queued task on the CALLING
    //                thread. Returns true if a task ran, false if the
    //                queue was empty. Safe to call from any thread,
    //                including pool workers and external callers. Does
    //                NOT block — purely best-effort.
    //
    // run_until(future): repeatedly try_run_one() (with a short sleep
    //                     when the queue is empty) until `fut` is in a
    //                     ready state. If the queue is empty AND no
    //                     workers are processing (best-effort), it falls
    //                     back to fut.wait() to avoid a busy-spin. This
    //                     is the preferred wait primitive when the
    //                     caller may be a pool worker.
    //
    // Both helpers honour shutdown_: once shutdown_ is set they stop
    // trying to run queued tasks (the destructor will drain the queue
    // and join).
    bool try_run_one() {
        std::function<void()> task;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (tasks_.empty()) return false;
            task = std::move(tasks_.front());
            tasks_.pop();
        }
        if (task) task();
        return true;
    }

    template <typename Fut>
    void run_until(Fut& fut) {
        while (true) {
            // Fast path: future is ready, we're done.
            if (fut.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
                return;
            }
            // Try to steal some work while we wait — this is what
            // prevents deadlock when the caller is itself a pool worker
            // that submitted subtasks to this same pool.
            bool stole = false;
            for (int i = 0; i < 8; ++i) {
                if (try_run_one()) {
                    stole = true;
                    break;
                }
            }
            if (!stole) {
                // Queue is empty — let the pool workers handle it.
                // Short sleep to avoid a hard busy-spin.
                fut.wait_for(std::chrono::milliseconds(1));
            }
        }
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
