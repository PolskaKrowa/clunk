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
 * Clunk GPU OccupancyModel — occupancy-cliff cost penalties.
 *
 * Builds on the basic compute_occupancy() in ArchLimits.h to expose a
 * cost penalty for "crossing an occupancy cliff". A rewrite that takes
 * register count from 32 to 33 on SM_80 (granularity round-up 33→36)
 * crosses a 100%→75% cliff and is therefore penalised; a rewrite that
 * takes it from 30 to 31 (no cliff crossed) is not.
 *
 * The penalty is a *positive* number representing the loss — i.e. the
 * amount by which the candidate is worse than the baseline due solely
 * to the occupancy drop. The integration agent's GPU cost model adds
 * this to the candidate's total cost.
 *
 * Thread-safety: all functions are pure / read-only.
 */
#include <cstdint>

#include "clunk/GPU/ArchLimits.h"

namespace clunk::gpu {

// ── occupancy_penalty ──────────────────────────────────────────────────────
//
// Compares the occupancy of (regs_per_thread, threads_per_block,
// shared_mem_per_block) against the occupancy at the *previous* register
// count (regs_per_thread - 1). Returns:
//
//     penalty = max(0.0,
//                   (prev_occupancy_ratio - curr_occupancy_ratio) * 100.0)
//
// The intuition: a rewrite that takes register count from 32 to 33
// (crossing the cliff) should be penalised; a rewrite that takes it
// from 30 to 31 (no cliff) should not. The 100.0 multiplier turns the
// 0.0–1.0 ratio into a 0–100 scale that's comparable to other cost
// terms (PTXCostModel costs are in the 1–100 range).
//
// If `regs_per_thread == 0` (caller doesn't know the reg count), the
// penalty is 0.
//
// The same function can be called with an arbitrary "baseline" register
// count via the 5-arg overload below — useful when the caller already
// has a candidate-baseline pair.
double occupancy_penalty(GpuArch arch,
                          unsigned regs_per_thread,
                          unsigned threads_per_block,
                          unsigned shared_mem_per_block);

// ── occupancy_penalty (explicit baseline) ──────────────────────────────────
//
// Compare occupancy at `baseline_regs_per_thread` vs `candidate_regs_per_thread`.
// Returns max(0, (baseline_ratio - candidate_ratio) * 100.0).
//
// This is the form the GPU cost model uses when scoring a candidate
// rewrite against the original kernel: baseline = original's regs/thread,
// candidate = rewrite's regs/thread.
double occupancy_penalty(GpuArch arch,
                          unsigned baseline_regs_per_thread,
                          unsigned candidate_regs_per_thread,
                          unsigned threads_per_block,
                          unsigned shared_mem_per_block);

} // namespace clunk::gpu
