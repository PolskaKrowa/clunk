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
 * Clunk GPU OccupancyModel — occupancy-cliff penalty calculation.
 *
 * Implements the occupancy-cliff penalty: a positive number representing
 * the occupancy loss when a candidate rewrite crosses a register-pressure
 * cliff on the target GPU arch.
 */
#include "clunk/GPU/OccupancyModel.h"

#include <algorithm>

namespace clunk::gpu {

namespace {

// Clamp ratio to [0, 1].
double clamp_ratio(double r) {
    if (r < 0.0) return 0.0;
    if (r > 1.0) return 1.0;
    return r;
}

} // namespace

double occupancy_penalty(GpuArch arch,
                          unsigned baseline_regs_per_thread,
                          unsigned candidate_regs_per_thread,
                          unsigned threads_per_block,
                          unsigned shared_mem_per_block)
{
    if (threads_per_block == 0) return 0.0;

    OccupancyInfo baseline = compute_occupancy(arch,
                                                threads_per_block,
                                                baseline_regs_per_thread,
                                                shared_mem_per_block);
    OccupancyInfo candidate = compute_occupancy(arch,
                                                 threads_per_block,
                                                 candidate_regs_per_thread,
                                                 shared_mem_per_block);

    double prev = clamp_ratio(baseline.occupancy_ratio);
    double curr = clamp_ratio(candidate.occupancy_ratio);

    if (curr >= prev) return 0.0;  // No cliff crossed (or improvement).
    return (prev - curr) * 100.0;
}

double occupancy_penalty(GpuArch arch,
                          unsigned regs_per_thread,
                          unsigned threads_per_block,
                          unsigned shared_mem_per_block)
{
    if (regs_per_thread == 0) return 0.0;
    // Default: compare against (regs_per_thread - 1) — i.e. "did adding
    // one more register cross a cliff?".
    unsigned prev = (regs_per_thread > 0) ? (regs_per_thread - 1) : 0;
    return occupancy_penalty(arch, prev, regs_per_thread,
                              threads_per_block, shared_mem_per_block);
}

} // namespace clunk::gpu
