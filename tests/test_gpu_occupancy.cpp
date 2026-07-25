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
 * Clunk GPU Occupancy Tests — ArchLimits table + compute_occupancy.
 *
 * Verifies the per-SM hardware constants against known NVIDIA values
 * (CUDA Programming Guide, "Compute Capabilities" table) and exercises
 * the occupancy calculator on a range of (block_size, regs, smem)
 * combinations across sm_70/75/80/86/89/90.
 */
#include <iostream>
#include <string>

#include "clunk/GPU/ArchLimits.h"

static int g_pass = 0, g_fail = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { std::cerr << "FAIL: " << msg << " at " << __FILE__ << ":" << __LINE__ << "\n"; g_fail++; } \
    else { g_pass++; } \
} while(0)

using namespace clunk::gpu;

// ── Test 1: ArchLimits table has correct known values ──────────────────────

static void test_arch_limits_known_values() {
    // sm_70 (Volta GV100)
    {
        ArchLimits l = get_arch_limits(GpuArch::SM_70);
        CHECK(l.max_warps_per_sm == 64, "sm_70 max_warps_per_sm=64");
        CHECK(l.max_threads_per_sm == 2048, "sm_70 max_threads_per_sm=2048");
        CHECK(l.max_blocks_per_sm == 32, "sm_70 max_blocks_per_sm=32");
        CHECK(l.max_registers_per_thread == 255, "sm_70 max_regs_per_thread=255");
        CHECK(l.register_alloc_granularity == 4, "sm_70 reg_granularity=4");
        CHECK(l.warp_size == 32, "sm_70 warp_size=32");
        CHECK(l.has_tensor_cores, "sm_70 has tensor cores");
        CHECK(!l.has_wgmma, "sm_70 has no wgmma");
    }

    // sm_75 (Turing)
    {
        ArchLimits l = get_arch_limits(GpuArch::SM_75);
        CHECK(l.max_warps_per_sm == 32, "sm_75 max_warps_per_sm=32");
        CHECK(l.max_threads_per_sm == 1024, "sm_75 max_threads_per_sm=1024");
        CHECK(l.max_blocks_per_sm == 16, "sm_75 max_blocks_per_sm=16");
        CHECK(l.register_alloc_granularity == 2, "sm_75 reg_granularity=2");
    }

    // sm_80 (Ampere GA100)
    {
        ArchLimits l = get_arch_limits(GpuArch::SM_80);
        CHECK(l.max_warps_per_sm == 64, "sm_80 max_warps_per_sm=64");
        CHECK(l.max_threads_per_sm == 2048, "sm_80 max_threads_per_sm=2048");
        CHECK(l.max_blocks_per_sm == 32, "sm_80 max_blocks_per_sm=32");
        CHECK(l.register_alloc_granularity == 4, "sm_80 reg_granularity=4");
        CHECK(l.has_cp_async, "sm_80 has cp.async");
        CHECK(!l.has_wgmma, "sm_80 has no wgmma");
    }

    // sm_86 (Ampere GA10x)
    {
        ArchLimits l = get_arch_limits(GpuArch::SM_86);
        CHECK(l.max_warps_per_sm == 48, "sm_86 max_warps_per_sm=48");
        CHECK(l.max_threads_per_sm == 1536, "sm_86 max_threads_per_sm=1536");
        CHECK(l.max_blocks_per_sm == 16, "sm_86 max_blocks_per_sm=16");
        CHECK(l.register_alloc_granularity == 4, "sm_86 reg_granularity=4");
    }

    // sm_89 (Ada AD102)
    {
        ArchLimits l = get_arch_limits(GpuArch::SM_89);
        CHECK(l.max_warps_per_sm == 48, "sm_89 max_warps_per_sm=48");
        CHECK(l.max_threads_per_sm == 1536, "sm_89 max_threads_per_sm=1536");
        CHECK(l.max_blocks_per_sm == 24, "sm_89 max_blocks_per_sm=24");
        CHECK(l.register_alloc_granularity == 2, "sm_89 reg_granularity=2");
    }

    // sm_90 (Hopper GH100)
    {
        ArchLimits l = get_arch_limits(GpuArch::SM_90);
        CHECK(l.max_warps_per_sm == 64, "sm_90 max_warps_per_sm=64");
        CHECK(l.max_threads_per_sm == 2048, "sm_90 max_threads_per_sm=2048");
        CHECK(l.max_blocks_per_sm == 32, "sm_90 max_blocks_per_sm=32");
        CHECK(l.register_alloc_granularity == 4, "sm_90 reg_granularity=4");
        CHECK(l.has_wgmma, "sm_90 has wgmma");
        CHECK(l.has_cluster, "sm_90 has cluster");
        CHECK(l.ptx_isa_version_major == 8, "sm_90 PTX ISA major=8");
    }
}

// ── Test 2: parse_gpu_arch handles common variants ─────────────────────────

static void test_parse_gpu_arch() {
    CHECK(parse_gpu_arch("sm_80") == GpuArch::SM_80, "sm_80 parsed");
    CHECK(parse_gpu_arch("sm80")  == GpuArch::SM_80, "sm80 parsed");
    CHECK(parse_gpu_arch("SM_90") == GpuArch::SM_90, "SM_90 parsed (case)");
    CHECK(parse_gpu_arch("80")    == GpuArch::SM_80, "80 parsed");
    CHECK(parse_gpu_arch("sm-89") == GpuArch::SM_89, "sm-89 parsed");
    CHECK(parse_gpu_arch("bogus") == GpuArch::Unknown, "bogus → Unknown");
}

// ── Test 3: arch_from_compute_capability ───────────────────────────────────

static void test_arch_from_cc() {
    CHECK(arch_from_compute_capability(70) == GpuArch::SM_70, "cc=70 → SM_70");
    CHECK(arch_from_compute_capability(75) == GpuArch::SM_75, "cc=75 → SM_75");
    CHECK(arch_from_compute_capability(80) == GpuArch::SM_80, "cc=80 → SM_80");
    CHECK(arch_from_compute_capability(86) == GpuArch::SM_86, "cc=86 → SM_86");
    CHECK(arch_from_compute_capability(89) == GpuArch::SM_89, "cc=89 → SM_89");
    CHECK(arch_from_compute_capability(90) == GpuArch::SM_90, "cc=90 → SM_90");
    // Unknown caps above 90 → clamp to SM_90
    CHECK(arch_from_compute_capability(95) == GpuArch::SM_90, "cc=95 → SM_90");
    CHECK(arch_from_compute_capability(0)  == GpuArch::Unknown, "cc=0 → Unknown");
}

// ── Test 4: compute_occupancy for high-occupancy case ──────────────────────
//
// On sm_80 with 256 threads/block, 32 regs/thread, 0 shared mem:
//   - warps_per_block = 8
//   - regs_per_thread_rounded = 32 (already multiple of 4)
//   - regs_per_warp = 32*32 = 1024 (already multiple of 256)
//   - regs_per_block = 1024*8 = 8192
//   - blocks_by_warps   = 64/8 = 8
//   - blocks_by_blocks  = 32
//   - blocks_by_regs    = 65536/8192 = 8
//   - blocks_by_threads = 2048/256 = 8
//   - blocks_by_smem    = unlimited
//   - blocks_per_sm = min(8, 32, 8, 8, ∞) = 8
//   - achieved_warps = 8*8 = 64
//   - occupancy = 64/64 = 100%
//
static void test_occupancy_full_sm80() {
    OccupancyInfo info = compute_occupancy(GpuArch::SM_80,
                                            256,  /* threads_per_block */
                                            32,   /* regs_per_thread */
                                            0);   /* shared_mem */
    CHECK(info.warps_per_block == 8, "warps_per_block=8");
    CHECK(info.regs_per_thread_rounded == 32, "regs_per_thread_rounded=32");
    CHECK(info.regs_per_warp == 1024, "regs_per_warp=1024");
    CHECK(info.regs_per_block == 8192, "regs_per_block=8192");
    CHECK(info.blocks_per_sm == 8, "blocks_per_sm=8");
    CHECK(info.achieved_warps_per_sm == 64, "achieved_warps=64");
    CHECK(info.theoretical_max_warps_per_sm == 64, "max_warps=64");
    CHECK(info.occupancy_pct > 99.9, "occupancy ~100%");
    CHECK((info.limited_by & 1u) != 0, "limited by warps");
    CHECK((info.limited_by & 4u) != 0, "limited by regs");
    CHECK((info.limited_by & 8u) != 0, "limited by threads");
}

// ── Test 5: register pressure limits occupancy ─────────────────────────────
//
// On sm_80 with 256 threads/block, 128 regs/thread:
//   - warps_per_block = 8
//   - regs_per_thread_rounded = 128
//   - regs_per_warp = 128*32 = 4096 (multiple of 256)
//   - regs_per_block = 4096*8 = 32768
//   - blocks_by_regs = 65536/32768 = 2
//   - blocks_per_sm = 2
//   - achieved_warps = 2*8 = 16
//   - occupancy = 16/64 = 25%
//
static void test_occupancy_reg_limited_sm80() {
    OccupancyInfo info = compute_occupancy(GpuArch::SM_80,
                                            256, 128, 0);
    CHECK(info.blocks_per_sm == 2, "blocks_per_sm=2 (reg-limited)");
    CHECK(info.achieved_warps_per_sm == 16, "achieved_warps=16");
    CHECK(info.occupancy_pct > 24.9 && info.occupancy_pct < 25.1,
          "occupancy ~25%");
    CHECK((info.limited_by & 4u) != 0, "limited by regs");
}

// ── Test 6: sm_89 caps blocks_per_sm at 24  ──────────────────
//
// The old code used max_blocks_per_sm=32 for all arches. On sm_89 it
// should be 24.
//
static void test_occupancy_sm89_block_cap() {
    // 64 threads/block (2 warps), low regs, no smem → block-limited
    OccupancyInfo info = compute_occupancy(GpuArch::SM_89,
                                            64, 16, 0);
    // blocks_by_warps = 48/2 = 24
    // blocks_by_blocks = 24
    // blocks_by_regs = 65536 / (16*32=512 → rounded to 512) / 2 = 64 → not limit
    // blocks_by_threads = 1536/64 = 24
    // blocks_per_sm = min(24, 24, 64, 24, ∞) = 24
    CHECK(info.blocks_per_sm <= 24, "blocks_per_sm capped at 24 for sm_89");
    CHECK(info.achieved_warps_per_sm <= 48, "achieved_warps ≤ 48 for sm_89");
    CHECK(info.theoretical_max_warps_per_sm == 48, "sm_89 max_warps=48");
}

// ── Test 7: sm_75 max_threads=1024 limits block size ───────────────────────

static void test_occupancy_sm75_thread_limit() {
    // 1024 threads/block, 32 regs, 0 smem
    // warps_per_block = 32
    // blocks_by_warps = 32/32 = 1
    // blocks_by_threads = 1024/1024 = 1
    // blocks_per_sm = 1
    OccupancyInfo info = compute_occupancy(GpuArch::SM_75,
                                            1024, 32, 0);
    CHECK(info.warps_per_block == 32, "warps_per_block=32");
    CHECK(info.blocks_per_sm == 1, "blocks_per_sm=1");
    CHECK(info.achieved_warps_per_sm == 32, "achieved_warps=32");
    CHECK(info.theoretical_max_warps_per_sm == 32, "sm_75 max_warps=32");
}

// ── Test 8: zero block size ────────────────────────────────────────────────

static void test_occupancy_zero_block() {
    OccupancyInfo info = compute_occupancy(GpuArch::SM_80, 0, 32, 0);
    CHECK(info.occupancy_pct == 0.0, "zero block size → 0% occupancy");
    CHECK(info.blocks_per_sm == 0, "blocks_per_sm=0");
}

// ── Test 9: shared memory limit ────────────────────────────────────────────
//
// On sm_80 with 164KB shared mem per block:
//   blocks_by_smem = 167936 / 167936 = 1
//   → 1 block per SM
//
static void test_occupancy_smem_limited() {
    unsigned smem_164k = 164 * 1024;
    OccupancyInfo info = compute_occupancy(GpuArch::SM_80,
                                            128, 16, smem_164k);
    // blocks_by_smem = 167936 / 167936 = 1
    // blocks_by_warps = 64/4 = 16 (128 threads = 4 warps)
    // blocks_by_blocks = 32
    // blocks_by_regs = 65536 / (16*32=512 rounded) / 4 = 32 → not limit
    // blocks_by_threads = 2048/128 = 16
    // blocks_per_sm = min(16, 32, 32, 16, 1) = 1
    CHECK(info.blocks_per_sm == 1, "blocks_per_sm=1 (smem-limited)");
    CHECK((info.limited_by & 16u) != 0, "limited by shared_mem");
}

// ── Test 10: gpu_arch_name round-trips ─────────────────────────────────────

static void test_arch_name_roundtrip() {
    for (GpuArch arch : {GpuArch::SM_70, GpuArch::SM_75, GpuArch::SM_80,
                          GpuArch::SM_86, GpuArch::SM_89, GpuArch::SM_90}) {
        std::string name = gpu_arch_name(arch);
        CHECK(parse_gpu_arch(name) == arch,
              "round-trip " + name + " → enum → name");
    }
}

// ── Main ────────────────────────────────────────────────────────────────────

int main() {
    std::cout << "=== Clunk GPU Occupancy Tests ===" << std::endl;

    std::cout << "  ArchLimits known values..." << std::endl;
    test_arch_limits_known_values();

    std::cout << "  parse_gpu_arch..." << std::endl;
    test_parse_gpu_arch();

    std::cout << "  arch_from_cc..." << std::endl;
    test_arch_from_cc();

    std::cout << "  full occupancy sm_80..." << std::endl;
    test_occupancy_full_sm80();

    std::cout << "  reg-limited sm_80..." << std::endl;
    test_occupancy_reg_limited_sm80();

    std::cout << "  sm_89 block cap..." << std::endl;
    test_occupancy_sm89_block_cap();

    std::cout << "  sm_75 thread limit..." << std::endl;
    test_occupancy_sm75_thread_limit();

    std::cout << "  zero block..." << std::endl;
    test_occupancy_zero_block();

    std::cout << "  smem-limited..." << std::endl;
    test_occupancy_smem_limited();

    std::cout << "  arch name roundtrip..." << std::endl;
    test_arch_name_roundtrip();

    std::cout << "\n=== Results: " << g_pass << " passed, "
              << g_fail << " failed ===" << std::endl;
    return g_fail > 0 ? 1 : 0;
}
