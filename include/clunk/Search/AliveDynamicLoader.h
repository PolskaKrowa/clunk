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
 * Clunk AliveDynamicLoader — runtime dlopen integration for Alive2.
 *
 * Mirrors the pattern established by SMTVerifier.cpp's Z3DynamicLoader:
 *   - Opaque forward-declared types (no link-time dependency on Alive2)
 *   - Function-pointer typedefs for every C API function we use
 *   - Singleton that dlopens libAlive2.so and resolves all symbols
 *   - REQUIRED symbols: if any fails to resolve, alive_loaded() returns
 *     false and all function pointers remain nullptr
 *   - Optional symbols: resolved best-effort; missing = feature disabled
 *   - Convenience macro ALIVE_API(name) = singleton.name##_fn
 *
 * Alive2 is an LLVM-based tool. The shared library must be built against
 * a compatible LLVM version. This is why it's loaded at runtime — Clunk
 * itself has no LLVM dependency at link time.
 *
 * When Alive2's library is not found, AliveDynamicLoader degrades
 * gracefully (alive_loaded() == false) and AliveVerifier falls back to
 * the out-of-process alive-tv binary subprocess path — exactly the same
 * "degrade gracefully" contract SMTVerifier uses for Z3.
 */

#include <string>

// Forward-declare the opaque Alive2 types (defined in vendor/alive2/alive2_c.h).
struct alive_result;

namespace clunk::search {

// ── Function-pointer typedefs ──────────────────────────────────────────────

using alive_result_status_t       = int (*)(const alive_result*);
using alive_result_message_t      = const char* (*)(const alive_result*);
using alive_result_json_t         = const char* (*)(const alive_result*);
using alive_result_source_lines_t = int (*)(const alive_result*);
using alive_result_target_lines_t = int (*)(const alive_result*);
using alive_result_free_t         = void (*)(alive_result*);

using alive_tvs_verify_t          = alive_result* (*)(const char*, const char*);
using alive_set_timeout_t         = void (*)(unsigned);
using alive_set_disable_undef_input_t = void (*)(int);
using alive_version_t             = const char* (*)(void);

// ── Dynamic loader (singleton) ──────────────────────────────────────────────

class AliveDynamicLoader {
public:
    static AliveDynamicLoader& instance() {
        static AliveDynamicLoader inst;
        return inst;
    }

    bool alive_loaded() const { return loaded_; }

    // ── REQUIRED function pointers ──────────────────────────────────────
    alive_result_status_t       alive_result_status_fn       = nullptr;
    alive_result_message_t      alive_result_message_fn      = nullptr;
    alive_result_free_t         alive_result_free_fn         = nullptr;
    alive_tvs_verify_t          alive_tvs_verify_fn          = nullptr;

    // ── OPTIONAL function pointers ─────────────────────────────────────
    alive_result_json_t         alive_result_json_fn         = nullptr;
    alive_result_source_lines_t alive_result_source_lines_fn = nullptr;
    alive_result_target_lines_t alive_result_target_lines_fn = nullptr;
    alive_set_timeout_t         alive_set_timeout_fn         = nullptr;
    alive_set_disable_undef_input_t alive_set_disable_undef_input_fn = nullptr;
    alive_version_t             alive_version_fn             = nullptr;

private:
    AliveDynamicLoader();
    AliveDynamicLoader(const AliveDynamicLoader&) = delete;
    AliveDynamicLoader& operator=(const AliveDynamicLoader&) = delete;

    void load();

    void* handle_ = nullptr;
    bool  loaded_ = false;
};

// Convenience macro — short alias for the singleton's function pointer members.
#define ALIVE_API(name) (clunk::search::AliveDynamicLoader::instance().name##_fn)

} // namespace clunk::search