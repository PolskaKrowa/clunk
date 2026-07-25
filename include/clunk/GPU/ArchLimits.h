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
 * Clunk GPU ArchLimits — per-SM hardware limits for occupancy modelling.
 *
 * Covers the major NVIDIA CUDA architectures clunk targets:
 *   sm_70 (Volta), sm_75 (Turing), sm_80 (Ampere GA100),
 *   sm_86 (Ampere GA10x), sm_89 (Ada Lovelace AD102),
 *   sm_90 (Hopper GH100).
 *
 * Constants sourced from the CUDA Programming Guide and the
 * cudaOccupancyMaxActiveBlocksPerMultiprocessor reference values.
 */
#include <cstdint>
#include <string>

namespace clunk::gpu {

// ── GPU architectures ──────────────────────────────────────────────────────
enum class GpuArch : uint8_t {
    SM_70 = 0,
    SM_75 = 1,
    SM_80 = 2,
    SM_86 = 3,
    SM_89 = 4,
    SM_90 = 5,
    Unknown = 0xFF
};

// ── Per-SM hardware limits ──────────────────────────────────────────────────
//
// `register_alloc_granularity` is the per-thread rounding multiple applied
// by the hardware (e.g. on sm_80, a kernel using 33 registers is charged
// as 36 per thread — rounded up to the next multiple of 4).
//
// `register_alloc_unit` is the warp-level allocation quantum (in registers
// per warp). On all currently-supported architectures this is 256 (i.e.
// 8 regs/lane × 32 lanes, or equivalently 32 regs/lane × 8 lanes for
// half-warps on older arches).
//
struct ArchLimits {
    GpuArch arch = GpuArch::Unknown;
    const char* name = "unknown";        // e.g. "sm_80"
    unsigned compute_capability = 80;    // e.g. 80
    unsigned ptx_isa_version_major = 7;  // PTX ISA version supported
    unsigned ptx_isa_version_minor = 0;

    // Threading limits
    unsigned max_threads_per_sm = 2048;
    unsigned max_warps_per_sm = 64;
    unsigned max_blocks_per_sm = 32;
    unsigned warp_size = 32;

    // Register file
    unsigned max_registers_per_thread = 255;
    unsigned max_registers_per_sm = 65536;
    unsigned register_alloc_granularity = 4;    // per-thread round-up multiple
    unsigned register_alloc_unit = 256;          // per-warp allocation quantum

    // Shared memory (bytes)
    unsigned max_shared_memory_per_sm = 167936;     // ~164 KB
    unsigned max_shared_memory_per_block = 167936;  // ~164 KB

    // Optional hardware features
    bool has_cp_async = false;       // cp.async (sm_80+)
    bool has_wgmma = false;          // wgmma (sm_90 only)
    bool has_cluster = false;        // thread block clusters (sm_90+)
    bool has_tensor_cores = false;   // mma.sync (sm_70+)
};

// ── Occupancy calculation result ────────────────────────────────────────────
//
// Extended with the additional fields the GPU cost model
// needs to reason about occupancy cliffs (occupancy_ratio, threads_per_sm,
// shared_mem_per_block, registers_per_sm, max_threads_per_sm,
// max_blocks_per_sm). All new fields default to 0 and are populated by
// compute_occupancy(); pre-existing fields keep their existing semantics
struct OccupancyInfo {
    GpuArch arch = GpuArch::Unknown;

    unsigned threads_per_block = 0;
    unsigned warps_per_block = 0;

    unsigned regs_per_thread = 0;
    unsigned regs_per_thread_rounded = 0;
    unsigned regs_per_warp = 0;
    unsigned regs_per_block = 0;
    unsigned regs_per_sm_used = 0;

    unsigned blocks_per_sm = 0;
    unsigned achieved_warps_per_sm = 0;
    unsigned theoretical_max_warps_per_sm = 0;

    double occupancy_pct = 0.0;  // 0.0 - 100.0

    // Which limit dominated the blocks_per_sm calculation.
    // Bitmask: 1=warps, 2=blocks, 4=regs, 8=threads, 16=shared_mem
    unsigned limited_by = 0;

    // ── Occupancy ratio and derived fields ───────────────────────────────────────────────
    //
    // `occupancy_ratio` is `occupancy_pct / 100.0` — i.e. 0.0 to 1.0.
    // This is the field the cost model asks for when comparing two
    // candidate rewrites; the cliff penalty (see OccupancyModel.h) is
    // computed as `max(0, prev_ratio - curr_ratio) * 100.0`.
    double occupancy_ratio = 0.0;

    // Achieved threads per SM (= blocks_per_sm * threads_per_block).
    unsigned threads_per_sm = 0;

    // Echo of the input shared_mem_per_block, for downstream consumers.
    unsigned shared_mem_per_block = 0;

    // Per-SM hardware limits echoed back (used by OccupancyModel and
    // by callers that don't want to call get_arch_limits separately).
    unsigned registers_per_sm = 65536;   // = ArchLimits::max_registers_per_sm
    unsigned max_threads_per_sm = 2048;  // = ArchLimits::max_threads_per_sm
    unsigned max_blocks_per_sm = 32;     // = ArchLimits::max_blocks_per_sm
};

// ── Lookup helpers ──────────────────────────────────────────────────────────

ArchLimits get_arch_limits(GpuArch arch);

// Parse "sm_80" / "sm80" / "80" → GpuArch::SM_80. Returns Unknown on miss.
GpuArch parse_gpu_arch(const std::string& name);

// "sm_80" for SM_80, "unknown" otherwise.
std::string gpu_arch_name(GpuArch arch);

// Convert an ArchDescriptor::compute_capability (e.g. 80) to GpuArch.
GpuArch arch_from_compute_capability(unsigned cc);

// ── Occupancy calculator ────────────────────────────────────────────────────
//
// Implements the standard NVIDIA occupancy calculation:
//   1. Round regs_per_thread up to the next multiple of
//      ArchLimits::register_alloc_granularity.
//   2. Compute regs_per_warp = regs_per_thread_rounded * warp_size, rounded
//      up to a multiple of register_alloc_unit.
//   3. Compute regs_per_block = regs_per_warp * warps_per_block.
//   4. blocks_per_sm =
//        min(max_warps_per_sm / warps_per_block,
//            max_blocks_per_sm,
//            max_registers_per_sm / regs_per_block,
//            max_threads_per_sm / threads_per_block,
//            max_shared_memory_per_sm / shared_mem_per_block)
//   5. achieved_warps_per_sm = blocks_per_sm * warps_per_block
//   6. occupancy_pct = 100 * achieved_warps_per_sm / max_warps_per_sm
//
OccupancyInfo compute_occupancy(GpuArch arch,
                                 unsigned threads_per_block,
                                 unsigned registers_per_thread,
                                 unsigned shared_mem_per_block);

} // namespace clunk::gpu