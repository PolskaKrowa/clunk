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
 * Clunk GPU CUDADriver — runtime-loaded CUDA driver API bindings.
 *
 * This class mirrors the Z3 dlopen pattern used by SMTVerifier: we
 * dlopen("libcuda.so.1") on first use and resolve the small subset of
 * CUDA driver symbols we need. If libcuda is not present on the host
 * (or any required symbol is missing), every public method falls back
 * gracefully:
 *
 *   - has_gpu() returns false
 *   - compile_ptx() returns an invalid ModuleHandle
 *   - launch_kernel() returns false
 *   - synchronize() returns false
 *
 * There is NO link-time dependency on libcuda. Linking clunk_core on a
 * machine without CUDA installed works fine. The driver is a singleton
 * accessed via CUDADriver::instance().
 *
 * Thread-safety: instance() is thread-safe (call_once). All other
 * methods assume external synchronisation. In practice the pipeline
 * calls these serially from a single thread.
 */
#include <memory>
#include <string>
#include <vector>

namespace clunk::gpu {

// CUDA error code (mirrors CUresult from cuda.h)
enum class CUresult : int {
    CUDA_SUCCESS                = 0,
    CUDA_ERROR_INVALID_VALUE    = 1,
    CUDA_ERROR_OUT_OF_MEMORY    = 2,
    CUDA_ERROR_NOT_INITIALIZED  = 3,
    CUDA_ERROR_DEINITIALIZED    = 4,
    CUDA_ERROR_NO_DEVICE        = 100,
    CUDA_ERROR_INVALID_DEVICE   = 101,
    CUDA_ERROR_NOT_FOUND        = 500,  // libcuda / symbol not found
    CUDA_ERROR_UNKNOWN          = 999,
};

// Exception thrown on CUDA errors when callers request exceptions.
// Note: the driver itself NEVER throws — it returns error codes / false.
// This struct exists for callers that prefer to translate returns.
struct CUDAException {
    CUresult code;
    std::string message;
};

// RAII handle for a CUmodule. Movable but not copyable.
class ModuleHandle {
public:
    ModuleHandle() = default;
    explicit ModuleHandle(void* mod) : mod_(mod) {}
    ~ModuleHandle() { release(); }

    ModuleHandle(ModuleHandle&& other) noexcept : mod_(other.mod_) {
        other.mod_ = nullptr;
    }
    ModuleHandle& operator=(ModuleHandle&& other) noexcept {
        if (this != &other) {
            release();
            mod_ = other.mod_;
            other.mod_ = nullptr;
        }
        return *this;
    }
    ModuleHandle(const ModuleHandle&) = delete;
    ModuleHandle& operator=(const ModuleHandle&) = delete;

    void* get() const { return mod_; }
    bool valid() const { return mod_ != nullptr; }
    void release();

private:
    void* mod_ = nullptr;
};

class CUDADriver {
public:
    // Singleton accessor (thread-safe, lazily initialised on first call).
    static CUDADriver& instance();

    // ── Capability queries ────────────────────────────────────────────────

    // Returns true iff libcuda was loaded AND at least one CUDA-capable
    // device is present.
    bool has_gpu() const;

    // Returns true iff libcuda.so.1 was successfully loaded and every
    // required symbol resolved. Does not imply a device is present.
    bool driver_loaded() const;

    int device_count() const;
    unsigned device_compute_capability(int device_idx) const;

    // ── Compilation & launch ──────────────────────────────────────────────

    // Compile a PTX string to a CUmodule. Returns an invalid handle on
    // failure (no GPU, driver not loaded, or PTX error). Never throws.
    //
    // The caller owns the returned handle and must keep it alive for the
    // lifetime of any kernel launched against it.
    ModuleHandle compile_ptx(const std::string& ptx);

    // Launch a kernel by name. Returns false on any failure or if no GPU.
    // kernel_args is a vector of opaque void* pointers to argument values
    // (matching the cuLaunchKernel ABI).
    bool launch_kernel(ModuleHandle& mod,
                       const std::string& kernel_name,
                       unsigned grid_x, unsigned grid_y, unsigned grid_z,
                       unsigned block_x, unsigned block_y, unsigned block_z,
                       unsigned shared_mem_bytes,
                       const std::vector<void*>& kernel_args);

    // Block the calling thread until all outstanding GPU work completes.
    // Returns false on failure or if no GPU.
    bool synchronize();

    // Last error message (for diagnostics). Empty if no error.
    const std::string& last_error() const { return last_error_; }

private:
    CUDADriver();
    CUDADriver(const CUDADriver&) = delete;
    CUDADriver& operator=(const CUDADriver&) = delete;

    // Lazy-load libcuda and resolve symbols. Idempotent.
    void try_load();

    // Set the last-error message and return the given value (template
    // trick so callers can do `return fail<bool>("...");`).
    template <typename T>
    T fail(const std::string& msg, T ret) {
        last_error_ = msg;
        return ret;
    }

    bool tried_load_ = false;
    bool loaded_ = false;
    void* libcuda_handle_ = nullptr;
    int device_count_ = 0;
    unsigned primary_ctx_device_idx_ = static_cast<unsigned>(-1);
    void* primary_ctx_ = nullptr;  // CUcontext
    mutable std::string last_error_;
};

} // namespace clunk::gpu
