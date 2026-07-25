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
 * Clunk MCACostModel — llvm-mca-backed candidate ranking.
 * See include/clunk/Evaluator/MCACostModel.h for the design contract.
 */
#include "clunk/Evaluator/MCACostModel.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <vector>

#include <unistd.h>

#include "clunk/IR/BasicBlock.h"
#include "clunk/IR/Instruction.h"

namespace clunk::evaluator {

namespace {

// Run a command, discarding output. Returns true on exit code 0.
bool run_silent(const std::string& cmd) {
    const std::string full = cmd + " >/dev/null 2>&1";
    return std::system(full.c_str()) == 0;
}

// Run a command and capture stdout. Returns nullopt-style empty string on
// failure (callers treat missing markers as failure anyway).
std::string run_capture(const std::string& cmd) {
    const std::string full = cmd + " 2>/dev/null";
    FILE* pipe = popen(full.c_str(), "r");
    if (!pipe) return {};
    std::string out;
    char buf[512];
    while (fgets(buf, sizeof(buf), pipe)) out += buf;
    pclose(pipe);
    return out;
}

// Extract "<label>: <number>" from llvm-mca's summary block.
bool parse_metric(const std::string& text, const std::string& label,
                  double& out) {
    auto pos = text.find(label);
    if (pos == std::string::npos) return false;
    pos += label.size();
    while (pos < text.size() && (text[pos] == ':' || text[pos] == ' '))
        ++pos;
    try {
        out = std::stod(text.substr(pos));
        return true;
    } catch (...) {
        return false;
    }
}

// A self-deleting temporary file.
struct TempFile {
    std::string path;
    explicit TempFile(const char* suffix) {
        char tmpl[] = "/tmp/clunk_mca_XXXXXX";
        int fd = mkstemp(tmpl);
        if (fd >= 0) {
            close(fd);
            path = std::string(tmpl) + suffix;
            std::rename(tmpl, path.c_str());
        }
    }
    ~TempFile() {
        if (!path.empty()) std::remove(path.c_str());
    }
    TempFile(const TempFile&) = delete;
    TempFile& operator=(const TempFile&) = delete;
};

} // anonymous namespace

// ── Availability probe ──────────────────────────────────────────────────────

bool MCACostModel::is_available() {
    static const bool available = [] {
        return run_silent("llc --version") && run_silent("llvm-mca --version");
    }();
    return available;
}

// ── IR rendering ────────────────────────────────────────────────────────────

std::string MCACostModel::render_module(const ir::Function& fn) {
    std::string body = fn.to_string();

    // clunk intrinsics measure as the real LLVM SIMD sequences they model.
    const std::string kClunkPrefix = "@clunk.vector.reduce.";
    const std::string kLLVMPrefix = "@llvm.vector.reduce.";
    for (size_t pos = 0; (pos = body.find(kClunkPrefix, pos)) != std::string::npos;) {
        body.replace(pos, kClunkPrefix.size(), kLLVMPrefix);
        pos += kLLVMPrefix.size();
    }

    // Declare every called function so the module is self-contained.
    std::ostringstream declares;
    std::vector<std::string> declared;
    for (auto& bb : fn.blocks()) {
        for (auto& inst : bb->instructions()) {
            if (!inst || inst->opcode() != ir::Opcode::Call) continue;
            auto it = inst->metadata().find("callee");
            if (it == inst->metadata().end()) continue;
            std::string callee = it->second;
            const std::string clunk_stem = "clunk.vector.reduce.";
            if (callee.rfind(clunk_stem, 0) == 0) {
                callee = "llvm.vector.reduce." + callee.substr(clunk_stem.size());
            }
            bool seen = false;
            for (auto& d : declared) {
                if (d == callee) { seen = true; break; }
            }
            if (seen) continue;
            declared.push_back(callee);
            declares << "declare " << inst->type()->to_string() << " @" << callee
                     << "(";
            for (size_t i = 0; i < inst->num_operands(); ++i) {
                if (i > 0) declares << ", ";
                declares << inst->operand(i)->type()->to_string();
            }
            declares << ")\n";
        }
    }

    return declares.str() + body + "\n";
}

// ── Measurement ─────────────────────────────────────────────────────────────

MCACostModel::Measurement MCACostModel::measure(const ir::Function& fn) const {
    Measurement m;
    if (!is_available()) {
        m.error = "llc / llvm-mca not available";
        return m;
    }

    const std::string module_text = render_module(fn);
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = cache_.find(module_text);
        if (it != cache_.end()) {
            ++stats_.cache_hits;
            return it->second;
        }
    }

    TempFile ll(".ll"), asm_out(".s");
    if (ll.path.empty() || asm_out.path.empty()) {
        m.error = "cannot create temporary files";
        return m;
    }
    {
        FILE* f = std::fopen(ll.path.c_str(), "w");
        if (!f) {
            m.error = "cannot write temporary IR file";
            return m;
        }
        std::fwrite(module_text.data(), 1, module_text.size(), f);
        std::fclose(f);
    }

    // llc: IR → native asm. `timeout` guards against pathological inputs.
    if (!run_silent("timeout 10 llc " + ll.path + " -O2 -o " + asm_out.path)) {
        std::lock_guard<std::mutex> lock(mutex_);
        ++stats_.measurements;
        ++stats_.failures;
        m.error = "llc failed to lower the IR";
        cache_[module_text] = m;  // failures cache too (deterministic)
        return m;
    }

    // llvm-mca: asm → cycle estimate over 100 iterations of the block.
    const std::string report = run_capture(
        "timeout 10 llvm-mca " + asm_out.path + " --iterations=100");
    double cycles = 0.0, uops = 0.0, ipc = 0.0;
    if (!parse_metric(report, "Total Cycles", cycles) || cycles <= 0.0) {
        std::lock_guard<std::mutex> lock(mutex_);
        ++stats_.measurements;
        ++stats_.failures;
        m.error = "llvm-mca produced no cycle count";
        cache_[module_text] = m;
        return m;
    }
    parse_metric(report, "Total uOps", uops);
    parse_metric(report, "IPC", ipc);

    m.ok = true;
    m.total_cycles = cycles;
    m.uops = uops;
    m.ipc = ipc;

    std::lock_guard<std::mutex> lock(mutex_);
    ++stats_.measurements;
    cache_[module_text] = m;
    return m;
}

double MCACostModel::compare(const ir::Function& original,
                             const ir::Function& candidate) const {
    auto mo = measure(original);
    auto mc = measure(candidate);
    if (!mo.ok || !mc.ok || mc.total_cycles <= 0.0) return 0.0;
    return mo.total_cycles / mc.total_cycles;
}

} // namespace clunk::evaluator
