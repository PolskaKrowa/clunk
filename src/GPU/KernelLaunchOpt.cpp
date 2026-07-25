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
 * Clunk GPU Kernel Launch Optimiser — tunes launch configuration, stream
 * scheduling, and memory transfer overlap for GPU kernels.
 */
#include "clunk/GPU/PTXOptimizer.h"

#include "clunk/GPU/ArchLimits.h"
#include "clunk/GPU/LivenessAnalysis.h"

#include <algorithm>
#include <cmath>
#include <unordered_set>

namespace clunk::gpu {

// ── Constructor ─────────────────────────────────────────────────────────────

KernelLaunchOptimizer::KernelLaunchOptimizer(const pattern::ArchDescriptor& target_arch)
    : target_arch_(target_arch)
{
    if (target_arch_.is_gpu && target_arch_.compute_capability > 0) {
        GpuArch derived = arch_from_compute_capability(
            target_arch_.compute_capability);
        if (derived != GpuArch::Unknown) {
            arch_ = derived;
        }
    }
}

// ── optimise_launch ─────────────────────────────────────────────────────────

LaunchOptimisation KernelLaunchOptimizer::optimise_launch(
    const ir::Function& kernel,
    const LaunchConfig& current_config)
{
    LaunchOptimisation result;
    result.original = current_config;

    // Cache register pressure estimation once — estimate_occupancy() is called
    // 5× below and each call would otherwise redo an O(N) walk of the kernel.
    const unsigned regs_per_thread = estimate_regs_per_thread(kernel);

    // Estimate occupancy with the current config.
    double current_occ = estimate_occupancy(kernel, current_config, regs_per_thread);

    // Try different block sizes and pick the best.
    LaunchConfig best_config = current_config;
    double best_occ = current_occ;

    static const unsigned block_sizes[] = {32, 64, 128, 256, 512, 1024};
    for (unsigned bs : block_sizes) {
        LaunchConfig trial = current_config;
        trial.block_x = bs;
        trial.block_y = 1;
        trial.block_z = 1;

        double trial_occ = estimate_occupancy(kernel, trial, regs_per_thread);
        if (trial_occ > best_occ) {
            best_occ = trial_occ;
            best_config = trial;
        }
    }

    result.optimised = best_config;
    result.estimated_speedup = (current_occ > 0.0) ? best_occ / current_occ : 1.0;

    // Build rationale string
    std::string rationale = "Block size tuned: ";
    rationale += std::to_string(current_config.block_x) + " -> " +
                 std::to_string(best_config.block_x);
    rationale += " (occupancy: " + std::to_string(current_occ * 100.0) +
                 "% -> " + std::to_string(best_occ * 100.0) + "%)";

    // If the kernel has high register pressure, mention it
    auto& attrs = kernel.attributes();
    if (attrs.count("gpu.register_pressure")) {
        rationale += "; register pressure: " + attrs.at("gpu.register_pressure");
    }

    // Mention the target arch
    ArchLimits limits = get_arch_limits(arch_);
    rationale += "; target: " + std::string(limits.name);

    result.rationale = rationale;

    return result;
}

// ── optimise_stream_schedule (stub) ────────────────────────────────────────
//
// Stub: a real implementation would build a DAG of independent kernels
// and assign them to different CUDA streams for overlap.

std::vector<LaunchConfig> KernelLaunchOptimizer::optimise_stream_schedule(
    const std::vector<std::pair<std::string, LaunchConfig>>& kernels)
{
    std::vector<LaunchConfig> result;
    result.reserve(kernels.size());
    for (auto& [name, config] : kernels) {
        (void)name;
        result.push_back(config);
    }
    return result;
}

// ── find_redundant_sync_points (stub) ──────────────────────────────────────
//
// Stub: a real implementation would scan the kernel sequence for
// cudaDeviceSynchronize() calls between independent launches and flag
// the ones that can be removed.

std::vector<size_t> KernelLaunchOptimizer::find_redundant_sync_points(
    const std::vector<std::string>& kernel_sequence)
{
    (void)kernel_sequence;
    return {};
}

// ── optimise_memory_transfers (stub) ────────────────────────────────────────
//
// Stub: a real implementation would pair each cudaMemcpyAsync with
// the first kernel that consumes it and ensure it's issued earliest on
// the same stream.

std::vector<KernelLaunchOptimizer::MemTransferOpt>
KernelLaunchOptimizer::optimise_memory_transfers(
    const std::vector<std::string>& operations)
{
    (void)operations;
    return {};
}

// ── estimate_regs_per_thread ───────────────────────────────────────────────
//
// O(N) walk over the kernel's instructions to estimate how many registers
// each thread will need. Extracted from estimate_occupancy() so callers can
// compute it once and reuse it across multiple occupancy probes.
//
// Priority:
//   1. gpu.register_pressure attribute (set by PTXOptimizer's
//      reduce_register_pressure pass, which uses LivenessAnalysis).
//   2. Real LivenessAnalysis pass (function_max_live).
//   3. Conservative default (32).
//
unsigned KernelLaunchOptimizer::estimate_regs_per_thread(
    const ir::Function& kernel) const
{
    // (1) Attribute set by PTXOptimizer
    auto& attrs = kernel.attributes();
    if (attrs.count("gpu.register_pressure")) {
        try {
            unsigned v = static_cast<unsigned>(
                std::stoul(attrs.at("gpu.register_pressure")));
            if (v > 0) return v;
        } catch (...) {
            // fall through
        }
    }

    // (2) Run LivenessAnalysis directly.
    LivenessAnalysis liveness;
    LiveSets live = liveness.compute(kernel);
    if (live.function_max_live > 0) {
        return static_cast<unsigned>(
            std::max(static_cast<size_t>(8), live.function_max_live));
    }

    // (3) Conservative default
    return 32;
}

// ── estimate_occupancy ─────────────────────────────────────────────────────
//
// Uses the ArchLimits table and compute_occupancy() helper from
// ArchLimits.h to model per-architecture occupancy correctly.
//
double KernelLaunchOptimizer::estimate_occupancy(
    const ir::Function& kernel,
    const LaunchConfig& config,
    unsigned regs_per_thread /* = 0 */)
{
    (void)kernel;  // not directly used here; regs come from caller or attrs

    unsigned block_size = config.block_x * config.block_y * config.block_z;
    if (block_size == 0) return 0.0;

    // Use caller-provided regs_per_thread if non-zero, else recompute.
    if (regs_per_thread == 0) {
        regs_per_thread = estimate_regs_per_thread(kernel);
    }

    // Use the ArchLimits-based occupancy calculator.
    OccupancyInfo info = compute_occupancy(arch_,
                                            block_size,
                                            regs_per_thread,
                                            config.shared_mem_bytes);

    // Return occupancy as a fraction in [0, 1].
    return std::min(info.occupancy_pct / 100.0, 1.0);
}

// ── tune_block_dims ─────────────────────────────────────────────────────────

LaunchConfig KernelLaunchOptimizer::tune_block_dims(
    const ir::Function& kernel,
    const LaunchConfig& config)
{
    LaunchConfig best = config;
    double best_occ = estimate_occupancy(kernel, config);

    // Try power-of-2 block sizes (matches the original implementation's
    // candidate set; optimise_launch above also tries the same set but
    // tune_block_dims is part of the public API so we keep it here too).
    static const unsigned sizes[] = {32, 64, 128, 256, 512, 1024};
    for (unsigned bs : sizes) {
        LaunchConfig trial = config;
        trial.block_x = bs;
        trial.block_y = 1;
        trial.block_z = 1;

        double occ = estimate_occupancy(kernel, trial);
        if (occ > best_occ) {
            best_occ = occ;
            best = trial;
        }
    }

    return best;
}

} // namespace clunk::gpu
