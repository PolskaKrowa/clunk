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
 * Clunk SMT Verifier — RAII wrappers for Z3 C API objects.
 *
 * Z3's C API requires manual reference counting for any Z3_ast / Z3_sort /
 * Z3_solver / Z3_params / Z3_model that the caller wishes to keep alive past
 * the immediate "make → use → forget" scope. The wrappers below encapsulate
 * the inc_ref/dec_ref dance so that ASTs are freed automatically when the
 * guard goes out of scope.
 *
 * These wrappers are header-only and depend ONLY on the opaque Z3 types
 * forward-declared by SMTVerifier.cpp's translation unit. They use the
 * Z3_inc_ref / Z3_dec_ref function-pointer members exposed by
 * Z3DynamicLoader (also declared in SMTVerifier.cpp). Because they reference
 * symbols defined in SMTVerifier.cpp's TU, they MUST be included AFTER the
 * Z3DynamicLoader declaration — but since they are templates / inline
 * functions called from SMTVerifier.cpp, the linker resolves them at link
 * time against the singleton.
 *
 * To avoid the include-ordering problem, the guards are designed as templates
 * parameterised on the function-pointer types. The SMTVerifier.cpp TU creates
 * concrete instances using the resolved Z3DynamicLoader members.
 */
#include <cstddef>
#include <functional>
#include <utility>

namespace clunk::search::z3raii {

// ── Function-pointer typedefs (mirror SMTVerifier.cpp's) ─────────────────────
// We use void* for opaque Z3 types so this header doesn't need the full Z3
// forward declarations. The SMTVerifier.cpp TU casts to the real Z3 types
// when constructing guards.

using IncRefFn  = void (*)(void* ctx, void* obj);
using DecRefFn  = void (*)(void* ctx, void* obj);

// ── Generic RAII guard ───────────────────────────────────────────────────────
//
// Holds a Z3 object plus the inc_ref/dec_ref function pointers. On
// destruction, calls dec_ref(ctx, obj) if obj is non-null.
//
// Move-only (no copies — would double-free). Null-state is well-defined.
//
class Z3RefGuard {
public:
    Z3RefGuard() = default;

    Z3RefGuard(void* ctx, void* obj, IncRefFn inc, DecRefFn dec)
        : ctx_(ctx), obj_(obj), inc_(inc), dec_(dec) {
        if (obj_ && inc_) inc_(ctx_, obj_);
    }

    ~Z3RefGuard() {
        release_internal();
    }

    Z3RefGuard(const Z3RefGuard&) = delete;
    Z3RefGuard& operator=(const Z3RefGuard&) = delete;

    Z3RefGuard(Z3RefGuard&& other) noexcept
        : ctx_(other.ctx_), obj_(other.obj_),
          inc_(other.inc_), dec_(other.dec_) {
        other.obj_ = nullptr;
    }

    Z3RefGuard& operator=(Z3RefGuard&& other) noexcept {
        if (this != &other) {
            release_internal();
            ctx_  = other.ctx_;
            obj_  = other.obj_;
            inc_  = other.inc_;
            dec_  = other.dec_;
            other.obj_ = nullptr;
        }
        return *this;
    }

    void* get() const { return obj_; }

    // Releases ownership of the managed object and returns the raw pointer.
    // The caller is now responsible for calling dec_ref when done.
    void* release() {
        void* tmp = obj_;
        // Note: we don't dec_ref on release — caller now owns the ref we added.
        obj_ = nullptr;
        return tmp;
    }

    void reset(void* ctx = nullptr, void* obj = nullptr,
               IncRefFn inc = nullptr, DecRefFn dec = nullptr) {
        release_internal();
        ctx_ = ctx;
        obj_ = obj;
        inc_ = inc;
        dec_ = dec;
        if (obj_ && inc_) inc_(ctx_, obj_);
    }

    explicit operator bool() const { return obj_ != nullptr; }

private:
    void release_internal() {
        if (obj_ && dec_ && ctx_) dec_(ctx_, obj_);
        obj_ = nullptr;
    }

    void*    ctx_ = nullptr;
    void*    obj_ = nullptr;
    IncRefFn inc_ = nullptr;
    DecRefFn dec_ = nullptr;
};

} // namespace clunk::search::z3raii
