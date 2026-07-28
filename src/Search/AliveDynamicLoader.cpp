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
 * Clunk AliveDynamicLoader — implementation.
 *
 * Tries several sonames for the Alive2 shared library (distributions
 * and build configurations differ). If loading or resolution of any
 * REQUIRED symbol fails, alive_loaded() returns false and all function
 * pointers remain nullptr — AliveVerifier then falls back to the
 * out-of-process alive-tv binary path.
 */

#include "clunk/Search/AliveDynamicLoader.h"

#include <dlfcn.h>

namespace clunk::search {

AliveDynamicLoader::AliveDynamicLoader() { load(); }

void AliveDynamicLoader::load() {
    // Try several sonames — distributions and build configs differ.
    // RTLD_LOCAL to avoid polluting the global symbol namespace.
    // RTLD_NOW to fail fast on unresolved symbols (Alive2 depends on
    // LLVM, and we want an immediate error rather than a crash later).
    static const char* names[] = {
        "libAlive2.so",
        "libalive2.so",
        "libAlive2.so.0",
        "libalive2.so.0",
        "libalive2.dylib",     // macOS
        nullptr
    };

    for (int i = 0; names[i]; ++i) {
        handle_ = dlopen(names[i], RTLD_NOW | RTLD_LOCAL);
        if (handle_) break;
    }

    if (!handle_) {
        loaded_ = false;
        return;
    }

    // Resolve REQUIRED symbols. If any is missing, fall back entirely.
    #define RESOLVE(name)                                               \
        do {                                                            \
            name##_fn = reinterpret_cast<name##_t>(                     \
                dlsym(handle_, #name));                                  \
            if (!name##_fn) {                                           \
                loaded_ = false;                                        \
                return;                                                 \
            }                                                           \
        } while (0)

    // Resolve OPTIONAL symbols (best-effort — nullptr is acceptable).
    #define RESOLVE_OPTIONAL(name)                                      \
        do {                                                            \
            name##_fn = reinterpret_cast<name##_t>(                     \
                dlsym(handle_, #name));                                  \
        } while (0)

    RESOLVE(alive_result_status);
    RESOLVE(alive_result_message);
    RESOLVE(alive_result_free);
    RESOLVE(alive_tvs_verify);

    // Optional symbols
    RESOLVE_OPTIONAL(alive_result_json);
    RESOLVE_OPTIONAL(alive_result_source_lines);
    RESOLVE_OPTIONAL(alive_result_target_lines);
    RESOLVE_OPTIONAL(alive_set_timeout);
    RESOLVE_OPTIONAL(alive_set_disable_undef_input);
    RESOLVE_OPTIONAL(alive_version);

    #undef RESOLVE
    #undef RESOLVE_OPTIONAL

    loaded_ = true;
}

} // namespace clunk::search
