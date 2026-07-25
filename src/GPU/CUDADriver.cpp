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
 * Clunk GPU CUDADriver — runtime-loaded CUDA driver API bindings.
 *
 * Implementation mirrors the Z3 dlopen pattern in src/Search/SMTVerifier.cpp:
 * we dlopen("libcuda.so.1") on first use, resolve a small set of driver
 * symbols via dlsym, and fall back gracefully if anything is missing.
 *
 * There is NO link-time dependency on libcuda.
 */
#include "clunk/GPU/CUDADriver.h"

#include <cstdlib>
#include <dlfcn.h>
#include <mutex>
#include <sstream>
#include <unordered_map>

namespace clunk::gpu {

// ── Opaque CUDA types (we never include cuda.h) ─────────────────────────────
//
// These match the typedefs in cuda.h: CUdevice is int, CUcontext/CUmodule/
// CUstream/CUfunction are opaque pointers. We deliberately use `void*`
// rather than distinct struct types because the driver API returns them
// as void*-compatible pointers and we never dereference them.
//
using CUdevice = int;
using CUcontext = void*;
using CUmodule = void*;
using CUstream = void*;
using CUfunction = void*;

// ── Function-pointer types for the driver symbols we use ────────────────────

using cuInit_t               = CUresult (*)(unsigned int);
using cuDeviceGetCount_t     = CUresult (*)(int*);
using cuDeviceGet_t          = CUresult (*)(CUdevice*, int);
using cuDeviceComputeCapability_t = CUresult (*)(int*, int*, CUdevice);
using cuDevicePrimaryCtxRetain_t = CUresult (*)(CUcontext*, CUdevice);
using cuDevicePrimaryCtxRelease_t = CUresult (*)(CUdevice);
using cuModuleLoadDataEx_t   = CUresult (*)(CUmodule*, const void*,
                                             unsigned int, void**,
                                             void**);
using cuModuleUnload_t       = CUresult (*)(CUmodule);
using cuModuleGetFunction_t  = CUresult (*)(CUfunction*, CUmodule,
                                             const char*);
using cuLaunchKernel_t       = CUresult (*)(CUfunction,
                                             unsigned int, unsigned int, unsigned int,
                                             unsigned int, unsigned int, unsigned int,
                                             unsigned int, CUstream,
                                             void**, void**);
using cuStreamSynchronize_t  = CUresult (*)(CUstream);
using cuCtxSynchronize_t     = CUresult (*)();

// ── Dynamic loader singleton ────────────────────────────────────────────────
//
// We resolve the symbols exactly once at process start. The handle is
// deliberately never dlclose'd: the CUDA driver registers atexit hooks
// and teardown callbacks that race with dlclose; leaking is the safe
// option (mirrors the SMTVerifier::Z3DynamicLoader pattern).
//
struct CUDALoader {
    void* handle = nullptr;
    bool loaded = false;

    cuInit_t                     cuInit_fn = nullptr;
    cuDeviceGetCount_t           cuDeviceGetCount_fn = nullptr;
    cuDeviceGet_t                cuDeviceGet_fn = nullptr;
    cuDeviceComputeCapability_t  cuDeviceComputeCapability_fn = nullptr;
    cuDevicePrimaryCtxRetain_t   cuDevicePrimaryCtxRetain_fn = nullptr;
    cuDevicePrimaryCtxRelease_t  cuDevicePrimaryCtxRelease_fn = nullptr;
    cuModuleLoadDataEx_t         cuModuleLoadDataEx_fn = nullptr;
    cuModuleUnload_t             cuModuleUnload_fn = nullptr;
    cuModuleGetFunction_t        cuModuleGetFunction_fn = nullptr;
    cuLaunchKernel_t             cuLaunchKernel_fn = nullptr;
    cuStreamSynchronize_t        cuStreamSynchronize_fn = nullptr;
    cuCtxSynchronize_t           cuCtxSynchronize_fn = nullptr;

    static CUDALoader& instance() {
        static CUDALoader inst;
        static std::once_flag flag;
        std::call_once(flag, []() { inst.load(); });
        return inst;
    }

    void load() {
        // Try a few well-known names. RTLD_NOW means we resolve all
        // symbols immediately (rather than lazily on first call), which
        // is what we want — we'd rather detect a missing symbol now than
        // crash later when calling it.
        static const char* names[] = {
            "libcuda.so.1",
            "libcuda.so",
            "libcuda.dylib",       // macOS
            "libcuda.1.dylib",
            "nvcuda.dll",          // Windows
            nullptr
        };

        for (int i = 0; names[i]; ++i) {
            handle = dlopen(names[i], RTLD_NOW | RTLD_GLOBAL);
            if (handle) break;
        }
        if (!handle) {
            loaded = false;
            return;
        }

        // Resolve every symbol. If any is missing, fall back entirely.
        // Note: cuDeviceComputeCapability is deprecated in CUDA 12 but
        // still present; if it's missing we'll treat the device as sm_80.
        #define RESOLVE(name)                                                \
            do {                                                             \
                name##_fn = reinterpret_cast<name##_t>(                      \
                    dlsym(handle, #name));                                   \
                if (!name##_fn) {                                            \
                    loaded = false;                                          \
                    return;                                                  \
                }                                                            \
            } while (0)

        RESOLVE(cuInit);
        RESOLVE(cuDeviceGetCount);
        RESOLVE(cuDeviceGet);
        RESOLVE(cuDevicePrimaryCtxRetain);
        RESOLVE(cuModuleLoadDataEx);
        RESOLVE(cuModuleUnload);
        RESOLVE(cuModuleGetFunction);
        RESOLVE(cuLaunchKernel);
        RESOLVE(cuStreamSynchronize);
        RESOLVE(cuCtxSynchronize);

        // Optional: cuDeviceComputeCapability is deprecated in CUDA 12.
        // We tolerate its absence (then device_compute_capability returns 0).
        cuDeviceComputeCapability_fn =
            reinterpret_cast<cuDeviceComputeCapability_t>(
                dlsym(handle, "cuDeviceComputeCapability"));
        // Optional: cuDevicePrimaryCtxRelease
        cuDevicePrimaryCtxRelease_fn =
            reinterpret_cast<cuDevicePrimaryCtxRelease_t>(
                dlsym(handle, "cuDevicePrimaryCtxRelease"));

        #undef RESOLVE

        loaded = true;
    }
};

// Convenience macro: short alias for the singleton's function pointer members
#define CUDAAPI(name) (CUDALoader::instance().name##_fn)

// ── ModuleHandle ────────────────────────────────────────────────────────────

void ModuleHandle::release() {
    if (!mod_) return;
    if (CUDALoader::instance().loaded && CUDAAPI(cuModuleUnload)) {
        // Best-effort unload; ignore errors
        CUDAAPI(cuModuleUnload)(mod_);
    }
    mod_ = nullptr;
}

// ── CUDADriver ──────────────────────────────────────────────────────────────

CUDADriver::CUDADriver() {
    // Force symbol resolution now (try_load is idempotent).
    try_load();
}

CUDADriver& CUDADriver::instance() {
    static CUDADriver inst;
    return inst;
}

void CUDADriver::try_load() {
    if (tried_load_) return;
    tried_load_ = true;

    // Touching the loader singleton triggers dlopen + dlsym.
    (void)CUDALoader::instance();

    if (!CUDALoader::instance().loaded) {
        loaded_ = false;
        last_error_ = "libcuda not available";
        return;
    }

    // Initialise the driver and probe device count.
    CUresult r = CUDAAPI(cuInit)(0);
    if (r != CUresult::CUDA_SUCCESS) {
        loaded_ = false;
        last_error_ = "cuInit failed";
        return;
    }

    int n = 0;
    r = CUDAAPI(cuDeviceGetCount)(&n);
    if (r != CUresult::CUDA_SUCCESS) {
        loaded_ = false;
        last_error_ = "cuDeviceGetCount failed";
        return;
    }
    device_count_ = n;
    loaded_ = (n > 0);
    if (!loaded_) {
        last_error_ = "no CUDA devices";
    }
}

bool CUDADriver::driver_loaded() const {
    return CUDALoader::instance().loaded;
}

bool CUDADriver::has_gpu() const {
    return CUDALoader::instance().loaded &&
           CUDADriver::instance().device_count_ > 0;
}

int CUDADriver::device_count() const {
    if (!CUDALoader::instance().loaded) return 0;
    return CUDADriver::instance().device_count_;
}

unsigned CUDADriver::device_compute_capability(int device_idx) const {
    if (!has_gpu()) return 0;
    if (device_idx < 0 || device_idx >= device_count_) return 0;

    // Use the deprecated cuDeviceComputeCapability if available; otherwise
    // return a sensible default (sm_80) so downstream code can still
    // pick a sensible ArchLimits entry.
    auto fn = CUDAAPI(cuDeviceComputeCapability);
    if (!fn) return 80;

    int major = 0, minor = 0;
    CUresult r = fn(&major, &minor, device_idx);
    if (r != CUresult::CUDA_SUCCESS) return 80;
    return static_cast<unsigned>(major * 10 + minor);
}

// ── compile_ptx ─────────────────────────────────────────────────────────────

ModuleHandle CUDADriver::compile_ptx(const std::string& ptx) {
    if (!has_gpu()) {
        last_error_ = "no CUDA device available";
        return ModuleHandle();
    }
    if (ptx.empty()) {
        last_error_ = "empty PTX input";
        return ModuleHandle();
    }

    // Retain the primary context on first use
    if (primary_ctx_device_idx_ == static_cast<unsigned>(-1)) {
        CUcontext ctx = nullptr;
        CUresult r = CUDAAPI(cuDevicePrimaryCtxRetain)(&ctx, 0);
        if (r != CUresult::CUDA_SUCCESS) {
            last_error_ = "cuDevicePrimaryCtxRetain failed";
            return ModuleHandle();
        }
        primary_ctx_ = ctx;
        primary_ctx_device_idx_ = 0;
    }

    CUmodule mod = nullptr;
    // cuModuleLoadDataEx takes (module*, image, numOptions, options, optValues).
    // We pass numOptions=0 for default behaviour (no JIT options).
    CUresult r = CUDAAPI(cuModuleLoadDataEx)(&mod, ptx.data(),
                                              0, nullptr, nullptr);
    if (r != CUresult::CUDA_SUCCESS) {
        std::ostringstream oss;
        oss << "cuModuleLoadDataEx failed (code " << static_cast<int>(r) << ")";
        last_error_ = oss.str();
        return ModuleHandle();
    }
    return ModuleHandle(mod);
}

// ── launch_kernel ───────────────────────────────────────────────────────────

bool CUDADriver::launch_kernel(ModuleHandle& mod,
                                const std::string& kernel_name,
                                unsigned grid_x, unsigned grid_y, unsigned grid_z,
                                unsigned block_x, unsigned block_y, unsigned block_z,
                                unsigned shared_mem_bytes,
                                const std::vector<void*>& kernel_args)
{
    if (!has_gpu()) {
        last_error_ = "no CUDA device available";
        return false;
    }
    if (!mod.valid()) {
        last_error_ = "invalid module handle";
        return false;
    }

    CUfunction func = nullptr;
    CUresult r = CUDAAPI(cuModuleGetFunction)(&func, mod.get(),
                                                kernel_name.c_str());
    if (r != CUresult::CUDA_SUCCESS) {
        last_error_ = "cuModuleGetFunction failed for '" + kernel_name + "'";
        return false;
    }

    // Build the args array (void**) from the vector<void*>.
    std::vector<void*> arg_ptrs = kernel_args;
    void** args_ptr = arg_ptrs.empty() ? nullptr : arg_ptrs.data();

    r = CUDAAPI(cuLaunchKernel)(func,
                                  grid_x, grid_y, grid_z,
                                  block_x, block_y, block_z,
                                  shared_mem_bytes,
                                  nullptr,  // default stream
                                  args_ptr,
                                  nullptr);  // no extra params
    if (r != CUresult::CUDA_SUCCESS) {
        std::ostringstream oss;
        oss << "cuLaunchKernel failed (code " << static_cast<int>(r) << ")";
        last_error_ = oss.str();
        return false;
    }
    return true;
}

// ── synchronize ─────────────────────────────────────────────────────────────

bool CUDADriver::synchronize() {
    if (!has_gpu()) {
        last_error_ = "no CUDA device available";
        return false;
    }
    // Prefer cuCtxSynchronize (synchronises the primary context)
    CUresult r = CUDAAPI(cuCtxSynchronize)();
    if (r != CUresult::CUDA_SUCCESS) {
        last_error_ = "cuCtxSynchronize failed";
        return false;
    }
    return true;
}

} // namespace clunk::gpu
