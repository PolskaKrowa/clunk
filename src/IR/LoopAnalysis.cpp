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
 * Clunk LoopAnalysis — natural-loop detection. See IR/LoopAnalysis.h.
 */
#include "clunk/IR/LoopAnalysis.h"

#include <algorithm>
#include <map>
#include <unordered_map>

namespace clunk::ir {

namespace {

// Reachable blocks in DFS order from the entry, plus the successor map.
struct Cfg {
    std::vector<std::string> order;                       // reachable, entry first
    std::unordered_map<std::string, std::vector<std::string>> succs;
    std::unordered_map<std::string, std::vector<std::string>> preds;
};

Cfg build_cfg(const Function& fn) {
    Cfg cfg;
    if (fn.blocks().empty()) return cfg;
    std::unordered_set<std::string> seen;
    std::vector<std::string> stack = {fn.blocks().front()->name()};
    while (!stack.empty()) {
        auto name = stack.back();
        stack.pop_back();
        if (!seen.insert(name).second) continue;
        cfg.order.push_back(name);
        auto bb = fn.block(name);
        if (!bb) continue;
        for (auto& s : bb->successors()) {
            // Only record edges to blocks that actually exist.
            if (fn.block(s)) {
                cfg.succs[name].push_back(s);
                stack.push_back(s);
            }
        }
    }
    for (auto& [from, tos] : cfg.succs) {
        for (auto& to : tos) cfg.preds[to].push_back(from);
    }
    return cfg;
}

// Iterative dominator sets (small CFGs — set-based dataflow is fine).
std::unordered_map<std::string, std::unordered_set<std::string>>
compute_dominators(const Cfg& cfg) {
    std::unordered_map<std::string, std::unordered_set<std::string>> dom;
    if (cfg.order.empty()) return dom;
    const auto& entry = cfg.order.front();
    std::unordered_set<std::string> all(cfg.order.begin(), cfg.order.end());
    dom[entry] = {entry};
    for (size_t i = 1; i < cfg.order.size(); ++i) dom[cfg.order[i]] = all;

    bool changed = true;
    while (changed) {
        changed = false;
        for (size_t i = 1; i < cfg.order.size(); ++i) {
            const auto& b = cfg.order[i];
            std::unordered_set<std::string> next;
            bool first = true;
            auto pit = cfg.preds.find(b);
            if (pit != cfg.preds.end()) {
                for (auto& p : pit->second) {
                    auto dit = dom.find(p);
                    if (dit == dom.end()) continue;  // unreachable pred
                    if (first) {
                        next = dit->second;
                        first = false;
                    } else {
                        std::unordered_set<std::string> merged;
                        for (auto& d : next) {
                            if (dit->second.count(d)) merged.insert(d);
                        }
                        next = std::move(merged);
                    }
                }
            }
            next.insert(b);
            if (next != dom[b]) {
                dom[b] = std::move(next);
                changed = true;
            }
        }
    }
    return dom;
}

} // anonymous namespace

std::vector<NaturalLoop> find_natural_loops(const Function& fn) {
    auto cfg = build_cfg(fn);
    if (cfg.order.empty()) return {};
    auto dom = compute_dominators(cfg);

    // Back-edges u -> h where h dominates u. std::map keeps header order
    // deterministic.
    std::map<std::string, NaturalLoop> loops;
    for (auto& u : cfg.order) {
        auto sit = cfg.succs.find(u);
        if (sit == cfg.succs.end()) continue;
        for (auto& h : sit->second) {
            if (!dom[u].count(h)) continue;
            auto& loop = loops[h];
            loop.header = h;
            loop.blocks.insert(h);
            if (std::find(loop.latches.begin(), loop.latches.end(), u) ==
                loop.latches.end()) {
                loop.latches.push_back(u);
            }
            // Grow the body backwards from the latch to the header.
            std::vector<std::string> stack = {u};
            while (!stack.empty()) {
                auto b = stack.back();
                stack.pop_back();
                if (!loop.blocks.insert(b).second) continue;
                auto pit = cfg.preds.find(b);
                if (pit == cfg.preds.end()) continue;
                for (auto& p : pit->second) stack.push_back(p);
            }
        }
    }

    std::vector<NaturalLoop> out;
    out.reserve(loops.size());
    for (auto& [h, loop] : loops) {
        // Preheader: the unique out-of-loop predecessor of the header.
        auto pit = cfg.preds.find(h);
        std::vector<std::string> outside;
        if (pit != cfg.preds.end()) {
            for (auto& p : pit->second) {
                if (!loop.contains(p)) outside.push_back(p);
            }
        }
        loop.preheader = outside.size() == 1 ? outside.front() : "";
        out.push_back(std::move(loop));
    }
    return out;
}

bool has_back_edge(const Function& fn) {
    auto cfg = build_cfg(fn);
    // Colour DFS: 0 = unvisited, 1 = on stack, 2 = done.
    std::unordered_map<std::string, int> colour;
    struct Frame { std::string bb; size_t next; };
    if (cfg.order.empty()) return false;
    for (auto& start : cfg.order) {
        if (colour[start] != 0) continue;
        std::vector<Frame> stack = {{start, 0}};
        colour[start] = 1;
        while (!stack.empty()) {
            auto& fr = stack.back();
            auto& ss = cfg.succs[fr.bb];
            if (fr.next < ss.size()) {
                auto s = ss[fr.next++];
                if (colour[s] == 1) return true;
                if (colour[s] == 0) {
                    colour[s] = 1;
                    stack.push_back({s, 0});
                }
            } else {
                colour[fr.bb] = 2;
                stack.pop_back();
            }
        }
    }
    return false;
}

} // namespace clunk::ir
