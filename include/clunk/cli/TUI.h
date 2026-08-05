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
 * Clunk ncurses TUI — a live view of the superoptimiser's progress.
 *
 * Two-panel layout:
 *
 *   ┌── Function list (left) ──┬── Detail (right) ────────────────┐
 *   │▶ f_const    [done]  3.2x│ define i32 @f_const(i32 %x) {    │
 *   │   f_mul      [round 4]   │ entry:                           │
 *   │   f_affine   [start]     │   ret i32 1                      │
 *   │   main       [pending]   │ }                                │
 *   │                          │                                  │
 *   │                          │ Round: 2   Adopted: 2            │
 *   │                          │ Score: -3.0 → -1.0 (3.0x)        │
 *   │                          │ Verified: yes                    │
 *   │                          │ Elapsed: 1.2s                    │
 *   └──────────────────────────┴──────────────────────────────────┘
 *   [↑/↓] navigate  [Enter] pin  [q] quit  [r] toggle raw IR
 *
 * The TUI subscribes to the pipeline's rich progress callback and
 * updates its display on every event. The pipeline runs on a worker
 * thread; the TUI runs on the main thread and polls for events /
 * keystrokes via a 50ms timeout in getch().
 *
 * Keyboard:
 *   ↑/k         Move selection up
 *   ↓/j         Move selection down
 *   Enter/Space Pin the currently-selected function (force the detail
 *               panel to stay on it even as new events arrive)
 *   Tab         Toggle between "follow latest" and "pinned" modes
 *   r           Toggle raw IR vs pretty-printed IR (TODO: pretty mode)
 *   q/Ctrl-C    Quit (sends SIGINT to the pipeline worker; the
 *               pipeline's time budget then expires on the next round)
 *
 * Thread safety:
 *   - The TUI owns a shared state struct (TUIState) guarded by a mutex.
 *   - The rich progress callback (fired from the pipeline worker
 *     thread) takes the lock, updates the state, releases the lock.
 *   - The main thread's render loop takes the lock, snapshots the
 *     state, releases the lock, then draws — never holds the lock
 *     across a drawing call.
 */
#include <atomic>
#include <chrono>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "clunk/Pipeline.h"

namespace clunk::cli {

// One row in the function list.
struct FunctionRow {
    std::string name;
    std::string stage;          // "pending", "start", "round", "done", "skip"
    std::string current_best_ir;
    size_t round = 0;
    size_t improvements_adopted = 0;
    double score_original = 0.0;
    double score_current = 0.0;
    double improvement_ratio = 1.0;
    bool verified = false;
    std::string message;
    double elapsed_ms = 0.0;
    std::chrono::steady_clock::time_point last_update;
};

// Shared state between the TUI render loop and the pipeline worker.
struct TUIState {
    std::mutex mutex;
    std::vector<FunctionRow> rows;
    std::map<std::string, size_t> name_to_index;  // lookup by function name
    std::string last_event_stage;
    std::string last_event_function;
    std::chrono::steady_clock::time_point last_event_time;
    bool pipeline_done = false;
    std::string pipeline_summary;  // filled in when pipeline_done = true
};

class TUI {
public:
    TUI();
    ~TUI();

    // Initialise ncurses. Returns false if ncurses couldn't be
    // initialised (e.g. not a tty); the caller should fall back to
    // the non-TUI mode in that case.
    bool init();

    // The rich progress callback to install on the pipeline. Captures
    // `this` and dispatches to handle_event().
    clunk::RichProgressCallback make_callback();

    // Run the render loop until the pipeline finishes AND the user
    // presses q. Call this on the main thread after starting the
    // pipeline on a worker thread.
    void run_loop(std::atomic<bool>& pipeline_running);

    // Mark the pipeline as done. Called by the main thread when the
    // pipeline worker joins.
    void mark_done(const std::string& summary);

private:
    void handle_event(const clunk::RichProgressEvent& ev);
    void render();
    void render_help_bar();
    void handle_key(int ch);

    // ncurses window handles
    void* win_list_ = nullptr;   // left panel (function list)
    void* win_detail_ = nullptr; // right panel (current function detail)
    int cols_ = 0, rows_ = 0;

    // Selection state
    size_t selected_ = 0;        // index into rows_
    bool pinned_ = false;        // if true, keep selection on selected_ even on new events
    bool show_raw_ir_ = true;    // toggle raw IR vs pretty (TODO)
    bool quit_requested_ = false;

    // Shared state (updated by the callback, read by the render loop)
    TUIState state_;

    // Initialise / shutdown ncurses
    bool ncurses_initialised_ = false;
};

} // namespace clunk::cli
