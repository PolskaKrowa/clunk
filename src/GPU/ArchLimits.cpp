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
 * Clunk GPU ArchLimits — per-SM hardware limits for occupancy modelling.
 */
#include "clunk/GPU/ArchLimits.h"

#include <algorithm>
#include <cctype>
#include <string>

namespace clunk::gpu {

// ── Architecture table ──────────────────────────────────────────────────────
//
// Sources (CUDA C++ Programming Guide, "Compute Capabilities" table):
//   sm_70 (Volta GV100):   2048 thr/SM, 64 warps/SM, 32 blk/SM,
//                          65536 regs/SM, granularity 4, 96 KB shared/SM.
//   sm_75 (Turing TU10x):  1024 thr/SM, 32 warps/SM, 16 blk/SM,
//                          65536 regs/SM, granularity 2, 64 KB shared/SM.
//   sm_80 (Ampere GA100):  2048 thr/SM, 64 warps/SM, 32 blk/SM,
//                          65536 regs/SM, granularity 4, 164 KB shared/SM.
//   sm_86 (Ampere GA10x):  1536 thr/SM, 48 warps/SM, 16 blk/SM,
//                          65536 regs/SM, granularity 4, 100 KB shared/SM.
//   sm_89 (Ada AD102):     1536 thr/SM, 48 warps/SM, 24 blk/SM,
//                          65536 regs/SM, granularity 2, 228 KB shared/SM.
//   sm_90 (Hopper GH100):  2048 thr/SM, 64 warps/SM, 32 blk/SM,
//                          65536 regs/SM, granularity 4, 227 KB shared/SM.
//
namespace {

constexpr unsigned KB = 1024;

struct ArchRow {
    GpuArch arch;
    const char* name;
    unsigned cc;
    unsigned ptx_major, ptx_minor;
    unsigned max_threads_per_sm;
    unsigned max_warps_per_sm;
    unsigned max_blocks_per_sm;
    unsigned warp_size;
    unsigned max_regs_per_thread;
    unsigned max_regs_per_sm;
    unsigned reg_granularity;
    unsigned reg_alloc_unit;
    unsigned smem_per_sm;
    unsigned smem_per_block;
    bool has_cp_async;
    bool has_wgmma;
    bool has_cluster;
    bool has_tensor_cores;
};

const ArchRow& lookup_row(GpuArch arch) {
    static const ArchRow rows[] = {
        { GpuArch::SM_70, "sm_70", 70, 6, 3,
          2048, 64, 32, 32, 255, 65536, 4, 256,
          96 * KB, 96 * KB,
          false, false, false, true },
        { GpuArch::SM_75, "sm_75", 75, 6, 4,
          1024, 32, 16, 32, 255, 65536, 2, 256,
          64 * KB, 64 * KB,
          false, false, false, true },
        { GpuArch::SM_80, "sm_80", 80, 7, 0,
          2048, 64, 32, 32, 255, 65536, 4, 256,
          164 * KB, 164 * KB,
          true, false, false, true },
        { GpuArch::SM_86, "sm_86", 86, 7, 1,
          1536, 48, 16, 32, 255, 65536, 4, 256,
          100 * KB, 99 * KB,
          true, false, false, true },
        { GpuArch::SM_89, "sm_89", 89, 7, 4,
          1536, 48, 24, 32, 255, 65536, 2, 256,
          228 * KB, 227 * KB,
          true, false, false, true },
        { GpuArch::SM_90, "sm_90", 90, 8, 0,
          2048, 64, 32, 32, 255, 65536, 4, 256,
          227 * KB, 227 * KB,
          true, true, true, true },
    };
    for (const auto& r : rows) {
        if (r.arch == arch) return r;
    }
    // Default: sm_80 (the most widely-deployed compute capability today).
    static const ArchRow fallback = rows[2];
    return fallback;
}

} // anonymous namespace

// ── get_arch_limits ─────────────────────────────────────────────────────────

ArchLimits get_arch_limits(GpuArch arch) {
    const ArchRow& r = lookup_row(arch);
    ArchLimits l;
    l.arch = r.arch;
    l.name = r.name;
    l.compute_capability = r.cc;
    l.ptx_isa_version_major = r.ptx_major;
    l.ptx_isa_version_minor = r.ptx_minor;
    l.max_threads_per_sm = r.max_threads_per_sm;
    l.max_warps_per_sm = r.max_warps_per_sm;
    l.max_blocks_per_sm = r.max_blocks_per_sm;
    l.warp_size = r.warp_size;
    l.max_registers_per_thread = r.max_regs_per_thread;
    l.max_registers_per_sm = r.max_regs_per_sm;
    l.register_alloc_granularity = r.reg_granularity;
    l.register_alloc_unit = r.reg_alloc_unit;
    l.max_shared_memory_per_sm = r.smem_per_sm;
    l.max_shared_memory_per_block = r.smem_per_block;
    l.has_cp_async = r.has_cp_async;
    l.has_wgmma = r.has_wgmma;
    l.has_cluster = r.has_cluster;
    l.has_tensor_cores = r.has_tensor_cores;
    return l;
}

// ── parse_gpu_arch ──────────────────────────────────────────────────────────

GpuArch parse_gpu_arch(const std::string& name) {
    // Accept "sm_70", "sm70", "sm-70", "70", "SM_80", etc.
    std::string s;
    s.reserve(name.size());
    for (char c : name) {
        if (c == '-' || c == '_') continue;
        s.push_back(static_cast<char>(std::toupper(static_cast<unsigned char>(c))));
    }
    // Strip leading "SM" if present
    if (s.size() >= 2 && s[0] == 'S' && s[1] == 'M') {
        s = s.substr(2);
    }
    if (s == "70") return GpuArch::SM_70;
    if (s == "75") return GpuArch::SM_75;
    if (s == "80") return GpuArch::SM_80;
    if (s == "86") return GpuArch::SM_86;
    if (s == "89") return GpuArch::SM_89;
    if (s == "90") return GpuArch::SM_90;
    return GpuArch::Unknown;
}

std::string gpu_arch_name(GpuArch arch) {
    switch (arch) {
        case GpuArch::SM_70: return "sm_70";
        case GpuArch::SM_75: return "sm_75";
        case GpuArch::SM_80: return "sm_80";
        case GpuArch::SM_86: return "sm_86";
        case GpuArch::SM_89: return "sm_89";
        case GpuArch::SM_90: return "sm_90";
        case GpuArch::Unknown: return "unknown";
    }
    return "unknown";
}

GpuArch arch_from_compute_capability(unsigned cc) {
    switch (cc) {
        case 70: return GpuArch::SM_70;
        case 75: return GpuArch::SM_75;
        case 80: return GpuArch::SM_80;
        case 86: return GpuArch::SM_86;
        case 89: return GpuArch::SM_89;
        case 90: return GpuArch::SM_90;
        default:
            if (cc >= 90) return GpuArch::SM_90;
            if (cc >= 89) return GpuArch::SM_89;
            if (cc >= 86) return GpuArch::SM_86;
            if (cc >= 80) return GpuArch::SM_80;
            if (cc >= 75) return GpuArch::SM_75;
            if (cc >= 70) return GpuArch::SM_70;
            return GpuArch::Unknown;
    }
}

// ── compute_occupancy ───────────────────────────────────────────────────────

OccupancyInfo compute_occupancy(GpuArch arch,
                                 unsigned threads_per_block,
                                 unsigned registers_per_thread,
                                 unsigned shared_mem_per_block)
{
    OccupancyInfo info;
    info.arch = arch;
    info.threads_per_block = threads_per_block;
    info.regs_per_thread = registers_per_thread;
    info.shared_mem_per_block = shared_mem_per_block;

    ArchLimits l = get_arch_limits(arch);

    // Echo per-SM hardware limits into the result struct so
    // downstream consumers (OccupancyModel::occupancy_penalty, callers
    // that don't want to call get_arch_limits separately) have them
    // available without a second lookup.
    info.registers_per_sm  = l.max_registers_per_sm;
    info.max_threads_per_sm = l.max_threads_per_sm;
    info.max_blocks_per_sm  = l.max_blocks_per_sm;

    if (threads_per_block == 0) {
        info.occupancy_pct = 0.0;
        info.occupancy_ratio = 0.0;
        return info;
    }

    unsigned warp_size = l.warp_size > 0 ? l.warp_size : 32;
    info.warps_per_block = (threads_per_block + warp_size - 1) / warp_size;
    info.theoretical_max_warps_per_sm = l.max_warps_per_sm;

    // Step 1: round up regs per thread to granularity
    unsigned reg_gran = l.register_alloc_granularity > 0
                          ? l.register_alloc_granularity : 1;
    unsigned regs_rounded = registers_per_thread;
    if (regs_rounded == 0) regs_rounded = 1;
    regs_rounded = ((regs_rounded + reg_gran - 1) / reg_gran) * reg_gran;
    if (regs_rounded > l.max_registers_per_thread) {
        regs_rounded = l.max_registers_per_thread;
    }
    info.regs_per_thread_rounded = regs_rounded;

    // Step 2: regs per warp, rounded up to allocation unit
    unsigned reg_unit = l.register_alloc_unit > 0
                          ? l.register_alloc_unit : 256;
    unsigned regs_per_warp = regs_rounded * warp_size;
    regs_per_warp = ((regs_per_warp + reg_unit - 1) / reg_unit) * reg_unit;
    info.regs_per_warp = regs_per_warp;

    // Step 3: regs per block
    unsigned regs_per_block = regs_per_warp * info.warps_per_block;
    info.regs_per_block = regs_per_block;

    // Step 4: blocks_per_sm = min of all limits
    unsigned blocks_by_warps   = info.warps_per_block > 0
        ? l.max_warps_per_sm / info.warps_per_block : 0;
    unsigned blocks_by_blocks  = l.max_blocks_per_sm;
    unsigned blocks_by_regs    = regs_per_block > 0
        ? l.max_registers_per_sm / regs_per_block : 0;
    unsigned blocks_by_threads = threads_per_block > 0
        ? l.max_threads_per_sm / threads_per_block : 0;
    unsigned blocks_by_smem    = shared_mem_per_block > 0
        ? l.max_shared_memory_per_sm / shared_mem_per_block
        : l.max_blocks_per_sm;

    unsigned blocks_per_sm = std::min({blocks_by_warps, blocks_by_blocks,
                                       blocks_by_regs, blocks_by_threads,
                                       blocks_by_smem});
    info.blocks_per_sm = blocks_per_sm;

    // Track which limit dominated
    unsigned limited_by = 0;
    unsigned cur_min = blocks_per_sm;
    if (blocks_by_warps   == cur_min) limited_by |= 1u;
    if (blocks_by_blocks  == cur_min) limited_by |= 2u;
    if (blocks_by_regs    == cur_min) limited_by |= 4u;
    if (blocks_by_threads == cur_min) limited_by |= 8u;
    if (blocks_by_smem    == cur_min) limited_by |= 16u;
    info.limited_by = limited_by;

    info.regs_per_sm_used = blocks_per_sm * regs_per_block;

    // Step 5: achieved warps per SM
    info.achieved_warps_per_sm = blocks_per_sm * info.warps_per_block;

    // Also report achieved threads per SM.
    info.threads_per_sm = blocks_per_sm * threads_per_block;

    // Step 6: occupancy %
    if (l.max_warps_per_sm > 0) {
        info.occupancy_pct = 100.0
            * static_cast<double>(info.achieved_warps_per_sm)
            / static_cast<double>(l.max_warps_per_sm);
        if (info.occupancy_pct > 100.0) info.occupancy_pct = 100.0;
    }

    // 0.0–1.0 ratio (the field the cost model asks for).
    info.occupancy_ratio = info.occupancy_pct / 100.0;

    return info;
}

} // namespace clunk::gpu
