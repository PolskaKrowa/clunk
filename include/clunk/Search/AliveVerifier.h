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
 * Clunk Alive2 Verifier — soundness backstop via Alive2.
 *
 * Alive2 (https://github.com/AliveToolkit/alive2, PLDI'21) is the
 * community gold-standard translation-validation tool for LLVM IR: given
 * a "source" and a "target" function, Alive2 proves refinement
 * (`src >= tgt`) using a bounded, memory-model-aware SMT encoding that
 * covers *far* more of LLVM's semantics than clunk's own `SMTVerifier`
 * (which is intentionally conservative and bails out to `Unknown` on
 * memory ops, calls, floating point and unbounded loops — see
 * SMTVerifier.h's soundness policy). Alive2 also catches something
 * SMTVerifier structurally cannot: whether clunk's *own emitted text* is
 * even valid, parseable LLVM IR, since it runs the output through
 * LLVM's real `.ll` parser.
 *
 * This class is an OPTIONAL extra gate, not a replacement for
 * SMTVerifier. It supports two verification backends:
 *
 *   1. **In-process** (preferred): dlopens libAlive2.so at runtime and
 *      calls `alive_tvs_verify()` directly. This avoids subprocess
 *      overhead, temp-file I/O, and shell-quoting issues. Requires a
 *      compatible libAlive2.so on the library search path. Follows the
 *      same dlopen pattern as Z3DynamicLoader — no link-time dependency.
 *
 *   2. **Out-of-process** (fallback): shells out to a prebuilt `alive-tv`
 *      binary, exactly like scripts/diff_test_alive.sh does, but wired
 *      into the pipeline itself instead of a shell script. Used when
 *      the shared library is not found but the CLI binary is.
 *
 * The verifier auto-detects availability at first use (cached):
 *   - If libAlive2.so is found via dlopen → in-process path
 *   - Else if alive-tv binary is on PATH → subprocess path
 *   - Else → NotAvailable
 *
 * AliveVerifier is used as a SECOND opinion: a candidate is only
 * ever adopted if `SMTVerifier` (or a sound-by-construction rewrite)
 * already blessed it. AliveVerifier::verify() is then an additional
 * check that can, at most, veto adoption (if it proves refinement
 * fails) or flag a `Warn` state (if alive-tv itself errored/timed
 * out) — it never causes an otherwise-unverified candidate to be
 * adopted.
 */
#include <string>
#include <vector>
#include <cstdint>
#include "clunk/IR/Function.h"

namespace clunk::ir { class Module; }

namespace clunk::search {

// ── Alive2 configuration ─────────────────────────────────────────────────
struct AliveConfig {
    // Path to (or bare name of) the alive-tv binary for the out-of-process
    // fallback. Looked up via PATH if not absolute.
    std::string alive_tv_path = "alive-tv";

    // Wall-clock timeout for a single verification, in milliseconds.
    // 0 = no timeout. Applies to both in-process and subprocess paths.
    unsigned timeout_ms = 30000;

    // Extra command-line arguments forwarded verbatim to alive-tv
    // (subprocess path only), e.g. {"--disable-undef-input"}.
    std::vector<std::string> extra_args;

    // Target triple / datalayout to embed in the synthesised modules.
    // Falls back to a generic little-endian x86_64 layout if empty.
    std::string target_triple;
    std::string target_datalayout;

    // Whether to disable undefined input variables.
    // Applies to the in-process path when alive_set_disable_undef_input
    // is available.
    bool disable_undef_input = false;
};

// ── Verification result ──────────────────────────────────────────────────
struct AliveResult {
    enum Status {
        Verified,     // Alive2 proved src >= tgt (refinement holds)
        Refuted,      // Alive2 proved refinement does NOT hold
        Unknown,      // Alive2 timed out / gave up (e.g. loop bound)
        Error,        // Alive2 crashed, or its output could not be parsed
        NotAvailable  // No Alive2 backend found (neither lib nor binary)
    };

    Status status = NotAvailable;
    std::string message;      // human-readable summary / diagnostic
    std::string raw_output;   // full stdout+stderr (subprocess) or JSON
    double time_ms = 0.0;
    int exit_code = -1;

    bool refines() const { return status == Verified; }
};

// ── AliveVerifier ─────────────────────────────────────────────────────────
class AliveVerifier {
public:
    explicit AliveVerifier(AliveConfig config = {});

    // True iff ANY Alive2 backend is available (libAlive2.so via dlopen,
    // or alive-tv binary on PATH). Cached per-process.
    static bool is_available(const AliveConfig& config = {});

    // True iff the in-process libAlive2.so backend is loaded.
    static bool is_lib_available();

    // True iff the out-of-process alive-tv binary is on PATH.
    static bool is_alive_tv_available(const std::string& alive_tv_path = "alive-tv");

    const AliveConfig& config() const { return config_; }

    // Check that `candidate` is a refinement of `original`.
    //
    // Automatically selects the best available backend:
    //   - In-process (libAlive2.so) if loaded
    //   - Out-of-process (alive-tv binary) otherwise
    //
    // Returns Status::NotAvailable if no backend is found.
    AliveResult verify(const ir::Function& original,
                        const ir::Function& candidate,
                        const ir::Module* module_ctx = nullptr) const;

    // Same as verify(), but writes/keeps the two temporary .ll files at
    // `original_ll_path` / `candidate_ll_path` (subprocess path only;
    // in-process path ignores these and works directly with strings).
    AliveResult verify_to_files(const ir::Function& original,
                                 const ir::Function& candidate,
                                 const std::string& original_ll_path,
                                 const std::string& candidate_ll_path,
                                 const ir::Module* module_ctx = nullptr) const;

private:
    AliveConfig config_;

    // Render `fn` as a standalone module: target triple/datalayout,
    // `declare` stubs for every function `fn` calls (best-effort — void
    // and integer/pointer signatures only), then `fn` itself.
    std::string render_standalone_module(const ir::Function& fn,
                                          const ir::Module* module_ctx) const;

    // ── In-process verification via dlopen'd libAlive2.so ────────────
    AliveResult verify_inprocess(const std::string& src_ir,
                                   const std::string& tgt_ir) const;

    // ── Out-of-process verification via alive-tv binary ──────────────
    // Run `alive-tv <lhs> <rhs> [extra_args...]`, capturing stdout+stderr
    // and the exit code, and classify the result.
    AliveResult run_alive_tv(const std::string& lhs_path,
                              const std::string& rhs_path) const;

    // Parse alive-tv's textual output into a Status when the exit code
    // alone is ambiguous.
    static AliveResult::Status classify_output(const std::string& output,
                                                 int exit_code);
};

} // namespace clunk::search
