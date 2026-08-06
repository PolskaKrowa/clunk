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
 * Clunk ncurses TUI — implementation.
 * See include/clunk/cli/TUI.h for the contract.
 *
 * Layout:
 *
 *   ┌── Function list (left, 30 cols)  ──┬── Detail (right, rest)  ────┐
 *   │ ▶ name1   [stage]  ratio          │ define ... @name1(...) {    │
 *   │   name2   [stage]  ratio           │   ...                       │
 *   │   ...                              │ }                           │
 *   │                                    │                             │
 *   │                                    │ Round: N   Adopted: M       │
 *   │                                    │ Score: -X → -Y (Z.Zx)       │
 *   │                                    │ Verified: yes/no            │
 *   │                                    │ Elapsed: S.Ss               │
 *   ├────────────────────────────────────┴─────────────────────────────┤
 *   │ [↑/↓] nav  [Tab] pin  [r] raw IR  [q] quit                       │
 *   └──────────────────────────────────────────────────────────────────┘
 *
 * The pipeline runs on a worker thread; the TUI's main loop polls for
 * keystrokes with a 50ms timeout (via timeout()) and re-renders on
 * every iteration. The rich progress callback fires from the worker
 * thread, takes the state mutex, updates the relevant row, releases.
 */

#include "clunk/cli/TUI.h"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <sstream>

// We use the wide-character ncurses header (curses.h) directly. The
// TUI only uses ASCII characters in its own chrome (arrows are
// rendered as ASCII "> " to avoid encoding issues), so we don't need
// the wide-character input path.
#include <ncurses.h>

namespace clunk::cli {

namespace {

// Truncate a string to fit a width, appending "..." if truncated.
std::string truncate(const std::string& s, size_t width) {
    if (s.size() <= width) return s;
    if (width < 3) return s.substr(0, width);
    return s.substr(0, width - 3) + "...";
}

// Format a double as a fixed-point string with `prec` decimals.
std::string fmt_double(double v, int prec = 2) {
    char buf[32];
    snprintf(buf, sizeof(buf), "%.*f", prec, v);
    return buf;
}

// Stage → short tag for the list display.
const char* stage_tag(const std::string& stage) {
    if (stage == "pending") return "pending";
    if (stage == "start")   return "start ";
    if (stage == "patterns") return "pattn ";
    if (stage == "inline")  return "inlin ";
    if (stage == "round")   return "round ";
    if (stage == "vector")  return "vector";
    if (stage == "hole")    return "hole  ";
    if (stage == "algo-pre") return "algo  ";
    if (stage == "mem")     return "mem   ";
    if (stage == "prune")   return "prune ";
    if (stage == "pow2")    return "pow2  ";
    if (stage == "mining")  return "mine  ";
    if (stage == "egraph")  return "egraph";
    if (stage == "verify")  return "verify";
    if (stage == "done")    return "done  ";
    if (stage == "skip")    return "skip  ";
    if (stage == "cap")     return "cap   ";
    if (stage == "gpu")     return "gpu   ";
    if (stage == "pipeline") return "pipe  ";
    return "???   ";
}

} // anonymous namespace

TUI::TUI() {}
TUI::~TUI() {
    if (ncurses_initialised_) {
        // Restore the terminal.
        endwin();
    }
}

bool TUI::init() {
    // initscr() can fail when stdin isn't a tty.
    if (initscr() == nullptr) return false;
    ncurses_initialised_ = true;

    cbreak();              // disable line buffering
    noecho();              // don't echo keystrokes
    keypad(stdscr, TRUE);  // enable arrow keys
    curs_set(0);           // hide the cursor
    timeout(50);           // non-blocking getch with 50ms timeout

    getmaxyx(stdscr, rows_, cols_);

    // Layout:
    //   ┌──────────────────────────────────────────────┐
    //   │  list panel  │  detail panel                 │  (rows - 4 high)
    //   │              │                               │
    //   ├──────────────┴───────────────────────────────┤
    //   │  Overall: [████████░░░░░░] 5/10  12.3s/30s   │  (2 high)
    //   ├──────────────────────────────────────────────┤
    //   │  [↑/↓] nav  [Tab/p] pin  [q] quit            │  (1 high, help)
    //   └──────────────────────────────────────────────┘
    int list_w = std::min(40, cols_ / 3);
    int detail_w = cols_ - list_w - 2;  // -2 for the separator + borders
    if (detail_w < 10) detail_w = 10;
    int help_h = 1;
    int status_h = 2;
    int panel_h = rows_ - help_h - status_h;
    if (panel_h < 3) panel_h = 3;

    win_list_ = newwin(panel_h, list_w, 0, 0);
    win_detail_ = newwin(panel_h, detail_w, 0, list_w + 1);
    win_status_ = newwin(status_h, cols_, panel_h, 0);

    return true;
}

clunk::RichProgressCallback TUI::make_callback() {
    return [this](const clunk::RichProgressEvent& ev) {
        handle_event(ev);
    };
}

void TUI::handle_event(const clunk::RichProgressEvent& ev) {
    std::lock_guard<std::mutex> lock(state_.mutex);

    // Update module-level progress fields (always present in every event).
    state_.module_total_functions = ev.module_total_functions;
    state_.module_done_functions = ev.module_done_functions;
    state_.module_elapsed_ms = ev.module_elapsed_ms;
    state_.module_time_budget_ms = ev.module_time_budget_ms;

    // "pipeline" stage events are module-level only (no function row).
    if (ev.stage == "pipeline") {
        state_.last_event_stage = ev.stage;
        state_.last_event_function = ev.function_name;
        state_.last_event_time = std::chrono::steady_clock::now();
        return;
    }

    // Find or create the row for this function.
    auto it = state_.name_to_index.find(ev.function_name);
    size_t idx;
    if (it == state_.name_to_index.end()) {
        FunctionRow row;
        row.name = ev.function_name;
        row.stage = "pending";
        state_.rows.push_back(std::move(row));
        idx = state_.rows.size() - 1;
        state_.name_to_index[ev.function_name] = idx;
    } else {
        idx = it->second;
    }

    FunctionRow& row = state_.rows[idx];
    row.name = ev.function_name;
    // Stages flow: pending → start → (round | vector | hole | ...) → done/skip.
    // Don't let a "start" overwrite a "done"/"skip".
    if (ev.stage == "start") {
        row.stage = "start";
        row.fn_progress = 0.05;  // just started
    } else if (ev.stage == "done") {
        row.stage = "done";
        row.fn_progress = 1.0;
    } else if (ev.stage == "skip") {
        row.stage = "skip";
        row.fn_progress = 1.0;  // skipped = complete (not pending)
    } else if (ev.stage != "pipeline" && ev.stage != "cap") {
        // Don't overwrite "done"/"skip" with later sub-stages.
        if (row.stage != "done" && row.stage != "skip") {
            row.stage = ev.stage;
        }
        // Update per-function progress from the round number.
        // progress = 1 - 1/(round+2), capped at 0.99.
        if (ev.round > 0) {
            row.fn_progress = std::min(0.99, 1.0 - 1.0 / static_cast<double>(ev.round + 2));
        }
    }
    if (!ev.current_best_ir.empty()) {
        row.current_best_ir = ev.current_best_ir;
    }
    row.round = ev.round;
    row.improvements_adopted = ev.improvements_adopted;
    row.score_original = ev.score_original;
    row.score_current = ev.score_current;
    row.improvement_ratio = ev.improvement_ratio;
    row.verified = ev.verified;
    if (!ev.message.empty()) row.message = ev.message;
    row.elapsed_ms = ev.elapsed_ms;
    row.last_update = std::chrono::steady_clock::now();

    state_.last_event_stage = ev.stage;
    state_.last_event_function = ev.function_name;
    state_.last_event_time = std::chrono::steady_clock::now();

    // Auto-follow: if not pinned and a new function appeared, select it.
    if (!pinned_ && ev.stage == "start") {
        selected_ = idx;
    }
}

void TUI::run_loop(std::atomic<bool>& pipeline_running) {
    while (!quit_requested_) {
        // Handle keystrokes.
        int ch = getch();
        if (ch != ERR) handle_key(ch);

        // Render.
        render();

        // If the pipeline is done AND the user has pressed q, exit.
        // (We keep rendering after the pipeline finishes so the user
        // can inspect the final state — they press q when ready.)
        if (state_.pipeline_done && quit_requested_) break;
        // Also exit if the pipeline is done and the user pressed q
        // before this loop iteration noticed.
        if (!pipeline_running.load() && quit_requested_) break;
    }
}

void TUI::mark_done(const std::string& summary) {
    std::lock_guard<std::mutex> lock(state_.mutex);
    state_.pipeline_done = true;
    state_.pipeline_summary = summary;
}

void TUI::handle_key(int ch) {
    std::lock_guard<std::mutex> lock(state_.mutex);
    size_t n = state_.rows.size();
    switch (ch) {
    case 'q':
    case 'Q':
    case 27:  // ESC
        quit_requested_ = true;
        break;
    case KEY_UP:
    case 'k':
    case 'K':
        if (n > 0 && selected_ > 0) --selected_;
        break;
    case KEY_DOWN:
    case 'j':
    case 'J':
        if (n > 0 && selected_ + 1 < n) ++selected_;
        break;
    case 'p':
    case 'P':
    case 9:   // Tab
    case '\n': // Enter
    case ' ':
        pinned_ = !pinned_;
        break;
    case 'r':
    case 'R':
        show_raw_ir_ = !show_raw_ir_;
        break;
    case KEY_RESIZE: {
        getmaxyx(stdscr, rows_, cols_);
        // Recreate the windows at the new size.
        if (win_list_) delwin(static_cast<WINDOW*>(win_list_));
        if (win_detail_) delwin(static_cast<WINDOW*>(win_detail_));
        if (win_status_) delwin(static_cast<WINDOW*>(win_status_));
        int list_w = std::min(40, cols_ / 3);
        int detail_w = cols_ - list_w - 2;
        if (detail_w < 10) detail_w = 10;
        int help_h = 1;
        int status_h = 2;
        int panel_h = rows_ - help_h - status_h;
        if (panel_h < 3) panel_h = 3;
        win_list_ = newwin(panel_h, list_w, 0, 0);
        win_detail_ = newwin(panel_h, detail_w, 0, list_w + 1);
        win_status_ = newwin(status_h, cols_, panel_h, 0);
        break;
    }
    default:
        break;
    }
}

void TUI::render() {
    std::lock_guard<std::mutex> lock(state_.mutex);

    render_function_list();
    render_detail_panel();
    render_progress_bars();
    render_help_bar();
}

void TUI::render_function_list() {
    WINDOW* wl = static_cast<WINDOW*>(win_list_);
    auto& rows = state_.rows;
    auto sel = selected_;
    bool done = state_.pipeline_done;

    werase(wl);
    box(wl, 0, 0);
    // Header: show done/total count.
    std::ostringstream hdr;
    hdr << " Functions (" << state_.module_done_functions
        << "/" << state_.module_total_functions << ") ";
    mvwprintw(wl, 0, 2, "%s", hdr.str().c_str());

    int list_h, list_w;
    getmaxyx(wl, list_h, list_w);
    int row_y = 1;
    int max_rows = list_h - 2;  // -2 for top/bottom border
    // Scroll if the selection is off-screen.
    size_t scroll_top = 0;
    if (sel >= static_cast<size_t>(max_rows)) {
        scroll_top = sel - max_rows + 1;
    }
    for (size_t i = scroll_top;
         i < rows.size() && row_y < list_h - 1;
         ++i, ++row_y) {
        const auto& r = rows[i];
        bool is_sel = (i == sel);
        // Highlight the selected row.
        if (is_sel) {
            wattron(wl, A_REVERSE);
        }
        // Format: " >name          [stage] ratio"
        // Show a mini progress indicator: █ for done/skip, ▓ for
        // in-progress, ░ for pending.
        const char* progress_char = "░";  // pending
        if (r.stage == "done" || r.stage == "skip") {
            progress_char = "█";
        } else if (r.stage == "start" || r.stage == "round" ||
                   r.stage == "patterns" || r.stage == "inline" ||
                   r.stage == "vector" || r.stage == "hole" ||
                   r.stage == "algo-pre" || r.stage == "mem" ||
                   r.stage == "prune" || r.stage == "pow2" ||
                   r.stage == "mining" || r.stage == "egraph" ||
                   r.stage == "verify") {
            progress_char = "▓";
        }
        std::ostringstream ss;
        ss << (is_sel ? "> " : "  ")
           << progress_char << " "
           << truncate(r.name, list_w - 20);
        // Pad to a fixed width so the stage tag aligns.
        std::string left = ss.str();
        while (static_cast<int>(left.size()) < list_w - 14) left += ' ';
        // Append stage + ratio.
        ss.str("");
        ss << "[" << stage_tag(r.stage) << "]"
           << (r.improvement_ratio > 0.01
                   ? fmt_double(r.improvement_ratio, 1) + "x"
                   : "    ");
        std::string right = ss.str();
        // Truncate to fit.
        std::string line = left + right;
        if (static_cast<int>(line.size()) > list_w - 2) {
            line = line.substr(0, list_w - 2);
        }
        mvwprintw(wl, row_y, 1, "%s", line.c_str());
        if (is_sel) {
            wattroff(wl, A_REVERSE);
        }
    }

    // If the pipeline is done, show a "DONE" indicator at the bottom
    // of the list.
    if (done && row_y < list_h - 1) {
        mvwprintw(wl, list_h - 2, 1, "── pipeline done ──");
    }

    wrefresh(wl);
}

void TUI::render_detail_panel() {
    WINDOW* wd = static_cast<WINDOW*>(win_detail_);
    auto& rows = state_.rows;
    auto sel = selected_;

    werase(wd);
    box(wd, 0, 0);

    int detail_h, detail_w;
    getmaxyx(wd, detail_h, detail_w);

    if (rows.empty()) {
        mvwprintw(wd, detail_h / 2, (detail_w - 32) / 2,
                  "(no functions yet — waiting for pipeline)");
        wrefresh(wd);
        return;
    }

    if (sel >= rows.size()) sel = rows.size() - 1;
    const FunctionRow& r = rows[sel];
    mvwprintw(wd, 0, 2, " %s ", r.name.c_str());

    // Stats line.
    int y = 1;
    std::ostringstream stats;
    stats << "Stage: " << r.stage
          << "  Round: " << r.round
          << "  Adopted: " << r.improvements_adopted;
    mvwprintw(wd, y, 1, "%s", truncate(stats.str(), detail_w - 2).c_str());
    ++y;

    std::ostringstream scores;
    scores << "Score: " << fmt_double(r.score_original, 2)
           << " → " << fmt_double(r.score_current, 2)
           << "  (" << fmt_double(r.improvement_ratio, 2) << "x)"
           << "  Verified: " << (r.verified ? "yes" : "no");
    mvwprintw(wd, y, 1, "%s", truncate(scores.str(), detail_w - 2).c_str());
    ++y;

    std::ostringstream elapsed;
    elapsed << "Elapsed: " << fmt_double(r.elapsed_ms / 1000.0, 2) << "s";
    if (!r.message.empty()) {
        elapsed << "   Last: " << r.message;
    }
    mvwprintw(wd, y, 1, "%s", truncate(elapsed.str(), detail_w - 2).c_str());
    ++y;

    // ── Per-function progress bar ───────────────────────────────────
    if (r.stage != "pending") {
        std::ostringstream fn_label;
        fn_label << "Fn progress: ";
        std::ostringstream fn_right;
        if (r.stage == "done" || r.stage == "skip") {
            fn_right << (r.stage == "done" ? "complete" : "skipped");
        } else {
            fn_right << static_cast<int>(r.fn_progress * 100) << "%";
        }
        draw_progress_bar(wd, y, 1, detail_w - 2, r.fn_progress,
                          fn_label.str(), fn_right.str());
        ++y;
    }

    // Separator.
    mvwhline(wd, y, 1, ACS_HLINE, detail_w - 2);
    ++y;

    // IR text.
    const std::string& ir = r.current_best_ir;
    if (ir.empty()) {
        mvwprintw(wd, y, 1, "(no IR snapshot yet)");
    } else {
        // Split the IR into lines and render each, scrolling if needed.
        std::vector<std::string> lines;
        std::istringstream iss(ir);
        std::string line;
        while (std::getline(iss, line)) {
            lines.push_back(line);
        }
        int avail_h = detail_h - y - 2;  // -2 for border + this line
        if (avail_h < 1) avail_h = 1;
        // Simple scrolling: if the IR is longer than avail_h, show the
        // LAST avail_h lines (the ret instruction is usually at the end
        // and is the most interesting part for live preview).
        size_t start = 0;
        if (lines.size() > static_cast<size_t>(avail_h)) {
            start = lines.size() - static_cast<size_t>(avail_h);
        }
        for (size_t i = start; i < lines.size() && y < detail_h - 1; ++i, ++y) {
            mvwprintw(wd, y, 1, "%s",
                      truncate(lines[i], detail_w - 2).c_str());
        }
    }

    wrefresh(wd);
}

void TUI::render_progress_bars() {
    WINDOW* ws = static_cast<WINDOW*>(win_status_);
    werase(ws);
    box(ws, 0, 0);

    int status_h, status_w;
    getmaxyx(ws, status_h, status_w);
    (void)status_h;  // only status_w is used below

    // Overall progress bar: done/total functions.
    double mod_frac = 0.0;
    if (state_.module_total_functions > 0) {
        mod_frac = static_cast<double>(state_.module_done_functions) /
                   static_cast<double>(state_.module_total_functions);
    }
    if (state_.pipeline_done) mod_frac = 1.0;

    std::ostringstream label;
    label << "Overall: ";
    std::ostringstream right;
    right << state_.module_done_functions << "/" << state_.module_total_functions << " fns";
    // Add time info.
    double elapsed_s = state_.module_elapsed_ms / 1000.0;
    if (state_.module_time_budget_ms > 0.0) {
        double budget_s = state_.module_time_budget_ms / 1000.0;
        right << "  " << fmt_double(elapsed_s, 1) << "s/"
              << fmt_double(budget_s, 1) << "s";
        // If we're over budget, show the overage in red.
        if (elapsed_s > budget_s) {
            right << " (OVER)";
        }
    } else {
        right << "  " << fmt_double(elapsed_s, 1) << "s";
    }

    draw_progress_bar(ws, 1, 1, status_w - 2, mod_frac,
                      label.str(), right.str());

    wrefresh(ws);
}

void TUI::draw_progress_bar(void* win_ptr, int y, int x, int width,
                             double fraction, const std::string& label,
                             const std::string& right_text) {
    WINDOW* win = static_cast<WINDOW*>(win_ptr);
    if (fraction < 0.0) fraction = 0.0;
    if (fraction > 1.0) fraction = 1.0;

    // Layout: [label ][████████░░░░░░][ right_text]
    // The bar takes whatever width is left after the label and right text.
    int label_len = static_cast<int>(label.size());
    int right_len = static_cast<int>(right_text.size());
    // Minimum bar width of 10 chars; if there's not enough room, skip the bar.
    int bar_width = width - label_len - right_len - 2;  // -2 for [ ]
    if (bar_width < 10) {
        // Not enough room — just print label + right_text.
        std::string line = label + right_text;
        mvwprintw(win, y, x, "%s", truncate(line, width).c_str());
        return;
    }

    // Draw the label.
    mvwprintw(win, y, x, "%s", label.c_str());

    // Draw the bar: [████████░░░░░░]
    int filled = static_cast<int>(fraction * bar_width);
    int bar_x = x + label_len;
    mvwaddch(win, y, bar_x, '[');
    for (int i = 0; i < bar_width; ++i) {
        if (i < filled) {
            mvwaddch(win, y, bar_x + 1 + i, ACS_CKBOARD);  // █
        } else {
            mvwaddch(win, y, bar_x + 1 + i, '.');  // ░ (ASCII fallback)
        }
    }
    mvwaddch(win, y, bar_x + 1 + bar_width, ']');

    // Draw the right text.
    if (!right_text.empty()) {
        mvwprintw(win, y, bar_x + 1 + bar_width + 1, " %s", right_text.c_str());
    }
}

void TUI::render_help_bar() {
    // Help line at the bottom of the screen.
    move(rows_ - 1, 0);
    clrtoeol();
    std::ostringstream help;
    help << "[↑/↓] nav  [Tab/p] pin (" << (pinned_ ? "on" : "off") << ")"
         << "  [r] raw IR (" << (show_raw_ir_ ? "on" : "off") << ")"
         << "  [q] quit";
    if (state_.pipeline_done) {
        help << "  ── PIPELINE DONE ──";
    }
    attron(A_REVERSE);
    printw("%s", truncate(help.str(), cols_).c_str());
    attroff(A_REVERSE);
    refresh();
}

} // namespace clunk::cli
