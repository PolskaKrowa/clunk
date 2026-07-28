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
 * Clunk AliveVerifier — see include/clunk/Search/AliveVerifier.h for the
 * design contract.
 *
 * Two verification backends:
 *   1. In-process via dlopen'd libAlive2.so (preferred, no subprocess
 *      overhead, no temp-file I/O)
 *   2. Out-of-process via alive-tv binary (fallback, uses the same
 *      popen + RAII temp files pattern as MCACostModel)
 */
#include "clunk/Search/AliveVerifier.h"
#include "clunk/Search/AliveDynamicLoader.h"

#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <sstream>
#include <algorithm>
#include <unordered_set>

#include <unistd.h>

#include "clunk/IR/BasicBlock.h"
#include "clunk/IR/Module.h"

namespace clunk::search {

namespace {

// ── Subprocess helpers (out-of-process fallback) ─────────────────────────

// Run a command, discarding output. Returns true on exit code 0.
bool run_silent(const std::string& cmd) {
    const std::string full = cmd + " >/dev/null 2>&1";
    return std::system(full.c_str()) == 0;
}

// Run a command and capture combined stdout+stderr plus its exit code.
std::string run_capture(const std::string& cmd, int& exit_code) {
    FILE* pipe = popen((cmd + " 2>&1").c_str(), "r");
    if (!pipe) {
        exit_code = -1;
        return {};
    }
    std::string out;
    char buf[512];
    while (fgets(buf, sizeof(buf), pipe)) out += buf;
    int status = pclose(pipe);
    exit_code = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
    return out;
}

// Quote a path for embedding in a shell command line.
std::string shell_quote(const std::string& s) {
    std::string q = "'";
    for (char c : s) {
        if (c == '\'') q += "'\\''";
        else q += c;
    }
    q += '\'";
    return q;
}

// A self-deleting temporary file.
struct TempFile {
    std::string path;
    explicit TempFile(const char* suffix) {
        char tmpl[] = "/tmp/clunk_alive_XXXXXX";
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

bool write_file(const std::string& path, const std::string& text) {
    FILE* f = std::fopen(path.c_str(), "w");
    if (!f) return false;
    std::fwrite(text.data(), 1, text.size(), f);
    std::fclose(f);
    return true;
}

} // anonymous namespace

// ── Availability probes ─────────────────────────────────────────────────────

bool AliveVerifier::is_lib_available() {
    return AliveDynamicLoader::instance().alive_loaded();
}

bool AliveVerifier::is_alive_tv_available(const std::string& alive_tv_path) {
    static std::unordered_set<std::string> known_available;
    static std::unordered_set<std::string> known_unavailable;
    if (known_available.count(alive_tv_path)) return true;
    if (known_unavailable.count(alive_tv_path)) return false;

    bool available = run_silent(shell_quote(alive_tv_path) + " --version");
    if (available) known_available.insert(alive_tv_path);
    else known_unavailable.insert(alive_tv_path);
    return available;
}

bool AliveVerifier::is_available(const AliveConfig& config) {
    return is_lib_available() || is_alive_tv_available(config.alive_tv_path);
}

// ── Constructor ─────────────────────────────────────────────────────────────
AliveVerifier::AliveVerifier(AliveConfig config) : config_(std::move(config)) {
    // Apply optional in-process configuration.
    if (is_lib_available()) {
        if (config_.timeout_ms > 0 && ALIVE_API(alive_set_timeout)) {
            ALIVE_API(alive_set_timeout)(config_.timeout_ms);
        }
        if (ALIVE_API(alive_set_disable_undef_input)) {
            ALIVE_API(alive_set_disable_undef_input)(config_.disable_undef_input ? 1 : 0);
        }
    }
}

// ── render_standalone_module() ───────────────────────────────────────────
std::string AliveVerifier::render_standalone_module(
        const ir::Function& fn, const ir::Module* module_ctx) const {
    std::ostringstream out;

    std::string triple = config_.target_triple;
    std::string datalayout = config_.target_datalayout;
    if (module_ctx && module_ctx->has_target()) {
        if (triple.empty()) triple = module_ctx->target().triple;
        if (datalayout.empty()) datalayout = module_ctx->target().datalayout;
    }
    if (triple.empty()) triple = "x86_64-unknown-linux-gnu";
    if (datalayout.empty()) {
        datalayout =
            "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128";
    }
    out << "target triple = \"" << triple << "\"\n";
    out << "target datalayout = \"" << datalayout << "\"\n\n";

    // Declare every function `fn` calls so the module is self-contained.
    std::vector<std::string> declared;
    for (const auto& bb : fn.blocks()) {
        for (const auto& inst : bb->instructions()) {
            if (!inst || inst->opcode() != ir::Opcode::Call) continue;
            auto it = inst->metadata().find("callee");
            if (it == inst->metadata().end()) continue;
            const std::string& callee = it->second;
            if (std::find(declared.begin(), declared.end(), callee) != declared.end())
                continue;
            declared.push_back(callee);
            out << "declare " << inst->type()->to_string() << " @" << callee << "(";
            for (size_t i = 0; i < inst->num_operands(); ++i) {
                if (i > 0) out << ", ";
                out << inst->operand(i)->type()->to_string();
            }
            out << ")\n";
        }
    }
    if (!declared.empty()) out << "\n";

    out << fn.to_string() << "\n";
    return out.str();
}

// ── In-process verification (libAlive2.so) ──────────────────────────────
AliveResult AliveVerifier::verify_inprocess(const std::string& src_ir,
                                              const std::string& tgt_ir) const {
    AliveResult result;
    auto t0 = std::chrono::steady_clock::now();

    alive_result* ar = ALIVE_API(alive_tvs_verify)(src_ir.c_str(), tgt_ir.c_str());

    auto t1 = std::chrono::steady_clock::now();
    result.time_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();

    if (!ar) {
        result.status = AliveResult::Error;
        result.message = "alive_tvs_verify returned null";
        return result;
    }

    int status = ALIVE_API(alive_result_status)(ar);
    const char* msg = ALIVE_API(alive_result_message)(ar);

    // Capture extra JSON if available.
    if (ALIVE_API(alive_result_json)) {
        const char* json = ALIVE_API(alive_result_json)(ar);
        if (json) result.raw_output = json;
    }
    if (msg) result.message = msg;

    switch (status) {
        case 0: // ALIVE_STATUS_CORRECT
            result.status = AliveResult::Verified;
            if (result.message.empty())
                result.message = "Alive2 (in-process) proved refinement holds";
            break;
        case 1: // ALIVE_STATUS_INCORRECT
            result.status = AliveResult::Refuted;
            if (result.message.empty())
                result.message = "Alive2 (in-process) REFUTED refinement — soundness bug";
            break;
        case 2: // ALIVE_STATUS_UNKNOWN
            result.status = AliveResult::Unknown;
            if (result.message.empty())
                result.message = "Alive2 (in-process) could not determine a verdict";
            break;
        case 3: // ALIVE_STATUS_ERROR
            result.status = AliveResult::Error;
            if (result.message.empty())
                result.message = "Alive2 (in-process) internal error";
            break;
        case 4: // ALIVE_STATUS_UNSUPPORTED
            result.status = AliveResult::Unknown;
            if (result.message.empty())
                result.message = "Alive2 (in-process) unsupported construct";
            break;
        default:
            result.status = AliveResult::Error;
            result.message = "Alive2 (in-process) returned unknown status " +
                              std::to_string(status);
            break;
    }

    ALIVE_API(alive_result_free)(ar);
    return result;
}

// ── classify_output() (subprocess path) ─────────────────────────────────
AliveResult::Status AliveVerifier::classify_output(const std::string& output,
                                                      int exit_code) {
    auto contains = [&](const char* needle) {
        return output.find(needle) != std::string::npos;
    };

    if (contains("doesn't verify") || contains("Mismatch") ||
        contains("value mismatch") || contains("Source is more defined")) {
        return AliveResult::Refuted;
    }
    if (contains("seems to be correct")) {
        return AliveResult::Verified;
    }
    if (contains("Timeout") || contains("timed out") || contains("unknown")) {
        return AliveResult::Unknown;
    }
    if (exit_code == 0) {
        return AliveResult::Unknown;
    }
    return AliveResult::Error;
}

// ── run_alive_tv() (subprocess path) ─────────────────────────────────────
AliveResult AliveVerifier::run_alive_tv(const std::string& lhs_path,
                                          const std::string& rhs_path) const {
    AliveResult result;
    auto t0 = std::chrono::steady_clock::now();

    std::ostringstream cmd;
    cmd << shell_quote(config_.alive_tv_path);
    if (config_.timeout_ms > 0) {
        std::ostringstream secs;
        secs.setf(std::ios::fixed);
        secs.precision(3);
        secs << (config_.timeout_ms / 1000.0);
        cmd.str("");
        cmd << "timeout " << secs.str() << "s "
            << shell_quote(config_.alive_tv_path);
    }
    for (const auto& arg : config_.extra_args) {
        cmd << " " << shell_quote(arg);
    }
    cmd << " " << shell_quote(lhs_path) << " " << shell_quote(rhs_path);

    int exit_code = -1;
    std::string output = run_capture(cmd.str(), exit_code);

    auto t1 = std::chrono::steady_clock::now();
    result.time_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    result.exit_code = exit_code;
    result.raw_output = output;

    if (exit_code == 124) {
        result.status = AliveResult::Unknown;
        result.message = "alive-tv timed out after " +
                          std::to_string(config_.timeout_ms) + "ms";
        return result;
    }

    result.status = classify_output(output, exit_code);
    switch (result.status) {
        case AliveResult::Verified:
            result.message = "alive-tv proved refinement holds";
            break;
        case AliveResult::Refuted:
            result.message = "alive-tv REFUTED refinement — soundness bug";
            break;
        case AliveResult::Unknown:
            result.message = "alive-tv could not determine a verdict";
            break;
        case AliveResult::Error:
        default:
            result.message = "alive-tv errored (exit " +
                              std::to_string(exit_code) + ")";
            break;
    }
    return result;
}

// ── verify() ──────────────────────────────────────────────────────────────
AliveResult AliveVerifier::verify(const ir::Function& original,
                                    const ir::Function& candidate,
                                    const ir::Module* module_ctx) const {
    // Prefer in-process path.
    if (is_lib_available()) {
        const std::string src_text = render_standalone_module(original, module_ctx);
        std::string tgt_text;
        if (candidate.name() != original.name()) {
            ir::Function renamed = candidate;
            renamed.set_name(original.name());
            tgt_text = render_standalone_module(renamed, module_ctx);
        } else {
            tgt_text = render_standalone_module(candidate, module_ctx);
        }
        return verify_inprocess(src_text, tgt_text);
    }

    // Fallback: out-of-process alive-tv binary.
    if (!is_alive_tv_available(config_.alive_tv_path)) {
        AliveResult r;
        r.status = AliveResult::NotAvailable;
        r.message = "no Alive2 backend found (tried libAlive2.so and " +
                     config_.alive_tv_path + ")";
        return r;
    }

    TempFile src(".ll"), tgt(".ll");
    if (src.path.empty() || tgt.path.empty()) {
        AliveResult r;
        r.status = AliveResult::Error;
        r.message = "cannot create temporary files for alive-tv";
        return r;
    }
    return verify_to_files(original, candidate, src.path, tgt.path, module_ctx);
}

// ── verify_to_files() ─────────────────────────────────────────────────────
AliveResult AliveVerifier::verify_to_files(
        const ir::Function& original, const ir::Function& candidate,
        const std::string& original_ll_path, const std::string& candidate_ll_path,
        const ir::Module* module_ctx) const {
    // If in-process is available, use it (ignoring the file paths —
    // the in-process path works directly with strings, which is strictly
    // better since it avoids temp-file I/O).
    if (is_lib_available()) {
        const std::string src_text = render_standalone_module(original, module_ctx);
        std::string tgt_text;
        if (candidate.name() != original.name()) {
            ir::Function renamed = candidate;
            renamed.set_name(original.name());
            tgt_text = render_standalone_module(renamed, module_ctx);
        } else {
            tgt_text = render_standalone_module(candidate, module_ctx);
        }
        return verify_inprocess(src_text, tgt_text);
    }

    // Out-of-process: write .ll files and invoke alive-tv.
    if (!is_alive_tv_available(config_.alive_tv_path)) {
        AliveResult r;
        r.status = AliveResult::NotAvailable;
        r.message = "no Alive2 backend found (tried libAlive2.so and " +
                     config_.alive_tv_path + ")";
        return r;
    }

    const std::string src_text = render_standalone_module(original, module_ctx);

    std::string tgt_text;
    if (candidate.name() != original.name()) {
        ir::Function renamed = candidate;
        renamed.set_name(original.name());
        tgt_text = render_standalone_module(renamed, module_ctx);
    } else {
        tgt_text = render_standalone_module(candidate, module_ctx);
    }

    if (!write_file(original_ll_path, src_text) ||
        !write_file(candidate_ll_path, tgt_text)) {
        AliveResult r;
        r.status = AliveResult::Error;
        r.message = "cannot write temporary IR files for alive-tv";
        return r;
    }

    return run_alive_tv(original_ll_path, candidate_ll_path);
}

} // namespace clunk::search
