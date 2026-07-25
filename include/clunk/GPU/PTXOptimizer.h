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
 * Clunk GPU Optimisation — PTX/ISA level optimisation and kernel launch tuning.
 * Operates after LLVM IR level optimisation on GPU code paths.
 *
 * The class is arch-aware (see GpuArch enum in ArchLimits.h): set_arch()
 * overrides the architecture implied by the ArchDescriptor passed to the
 * constructor, and the PTX emitter honours the chosen arch when emitting
 * the .target directive.
 *
 * INVARIANTS:
 *   * optimise() does NOT mutate its input ir::Function. Every pass
 *     either deep-copies the IR or stores its results in a side table.
 *   * All analysis methods are const-safe to call concurrently (the
 *     class is thread-compatible).
 */
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "clunk/IR/Function.h"
#include "clunk/GPU/ArchLimits.h"
#include "clunk/Pattern/PatternLibrary.h"

namespace clunk::gpu {

// ── PTX instruction (simplified model) ──────────────────────────────────
//
// Retained for backwards-compatibility with the original public header.
// The PTXEmitter now uses a string-based representation internally; this
// struct is kept only so existing source that references it still compiles.
struct PTXInstruction {
    std::string opcode;          // e.g. "ld.global", "add.f32", "mad.lo.s32"
    std::vector<std::string> operands;
    std::string predicate;       // Conditional execution predicate
    std::map<std::string, std::string> modifiers;
};

// ── Warp state (for execution modelling) ────────────────────────────────
struct WarpState {
    uint32_t active_mask = 0xFFFFFFFF;  // Which lanes are active
    uint32_t diverged = 0;              // Which lanes have diverged
    size_t pc = 0;                       // Program counter
    std::vector<uint64_t> registers;    // Per-lane register state (indexed by reg id)
};

// ── Memory transaction ──────────────────────────────────────────────────
struct MemoryTransaction {
    enum class Type { Load, Store, Atomic };
    enum class AddressSpace : uint8_t { Global, Shared, Local, Const };

    Type type;
    AddressSpace address_space = AddressSpace::Global;
    size_t bytes = 0;
    bool coalesced = false;
    size_t warp_id = 0;
};

// Helper: human-readable name for an AddressSpace (for diagnostics/serialization).
static inline const char* address_space_name(MemoryTransaction::AddressSpace as) noexcept {
    switch (as) {
        case MemoryTransaction::AddressSpace::Global: return "global";
        case MemoryTransaction::AddressSpace::Shared: return "shared";
        case MemoryTransaction::AddressSpace::Local:  return "local";
        case MemoryTransaction::AddressSpace::Const:  return "const";
    }
    return "global";
}

// ── Kernel launch configuration ─────────────────────────────────────────
struct LaunchConfig {
    unsigned grid_x = 1, grid_y = 1, grid_z = 1;
    unsigned block_x = 256, block_y = 1, block_z = 1;
    unsigned shared_mem_bytes = 0;
    unsigned stream_id = 0;
};

// ── Kernel launch optimisation result ───────────────────────────────────
struct LaunchOptimisation {
    LaunchConfig original;
    LaunchConfig optimised;
    double estimated_speedup = 1.0;
    std::string rationale;
};

// ── PTX-level optimiser ─────────────────────────────────────────────────
class PTXOptimizer final {
public:
    explicit PTXOptimizer(const pattern::ArchDescriptor& target_arch);

    // Optimise a GPU kernel at the PTX/ISA level.
    //
    // Contract: returns a deep-copied, optimised ir::Function. The input
    // function is NEVER mutated. This is the entry point invoked by
    // Pipeline::gpu_optimise.
    std::shared_ptr<ir::Function> optimise(const ir::Function& kernel);

    // ── Arch targeting ──────────────────────────────────────────────────
    //
    // set_arch() overrides the architecture implied by the ArchDescriptor
    // passed to the constructor. The default is SM_80 (most widely
    // deployed compute capability today).
    void set_arch(GpuArch arch) { arch_ = arch; }
    GpuArch arch() const { return arch_; }

    // After optimise(), this returns the PTX text emitted for the result
    // function. Empty if optimise() has not been called or PTX emission
    // was skipped.
    const std::string& ptx_text() const { return ptx_text_; }

    // ── Analysis passes (const; safe for concurrent calls) ─────────────

    // Model warp execution (Nsight-inspired).
    std::vector<WarpState> simulate_execution(
        const ir::Function& kernel,
        size_t num_warps = 1) const;

    // Analyse memory access patterns.
    std::vector<MemoryTransaction> analyse_memory_accesses(
        const ir::Function& kernel) const;

    // Estimate warp divergence using the real DivergenceAnalysis pass.
    double estimate_divergence(const ir::Function& kernel) const;

    // ── Configuration ───────────────────────────────────────────────────
    void set_max_registers(unsigned max_regs) { max_registers_ = max_regs; }
    void set_target_occupancy(double occ) { target_occupancy_ = occ; }

    unsigned max_registers() const { return max_registers_; }
    double target_occupancy() const { return target_occupancy_; }

private:
    pattern::ArchDescriptor target_arch_;
    GpuArch arch_ = GpuArch::SM_80;
    unsigned max_registers_ = 255;
    double target_occupancy_ = 0.75;
    std::string ptx_text_;

    // ── IR-level passes ─────────────────────────────────────────────────
    //
    // All passes deep-copy their input; the returned ir::Function is a
    // fresh object whose instructions may carry additional metadata
    // strings (gpu hints).
    std::shared_ptr<ir::Function> schedule_instructions(
        const ir::Function& kernel) const;
    std::shared_ptr<ir::Function> optimise_memory_coalescing(
        const ir::Function& kernel) const;
    std::shared_ptr<ir::Function> reduce_register_pressure(
        const ir::Function& kernel) const;
    void analyse_fp_precision(const ir::Function& kernel) const;
};

// ── Kernel launch optimiser ─────────────────────────────────────────────
class KernelLaunchOptimizer final {
public:
    explicit KernelLaunchOptimizer(const pattern::ArchDescriptor& target_arch);

    // Optimise kernel launch configuration
    LaunchOptimisation optimise_launch(
        const ir::Function& kernel,
        const LaunchConfig& current_config);

    // Optimise stream scheduling and overlap (stub: passes input through).
    std::vector<LaunchConfig> optimise_stream_schedule(
        const std::vector<std::pair<std::string, LaunchConfig>>& kernels);

    // Minimise host/device synchronisation points (stub: returns empty).
    std::vector<size_t> find_redundant_sync_points(
        const std::vector<std::string>& kernel_sequence);

    // Optimise memory transfer timing (stub: returns empty).
    struct MemTransferOpt {
        size_t original_position;
        size_t optimised_position;
        std::string rationale;
    };
    std::vector<MemTransferOpt> optimise_memory_transfers(
        const std::vector<std::string>& operations);

    // ── Arch targeting ──────────────────────────────────────────────────
    void set_arch(GpuArch arch) { arch_ = arch; }
    GpuArch arch() const { return arch_; }

private:
    pattern::ArchDescriptor target_arch_;
    GpuArch arch_ = GpuArch::SM_80;

    // Estimate registers per thread for a kernel (O(N) walk).
    unsigned estimate_regs_per_thread(const ir::Function& kernel) const;

    // Estimate occupancy given launch config and kernel.
    // If regs_per_thread is provided, skips the O(N) re-estimation.
    double estimate_occupancy(const ir::Function& kernel,
                              const LaunchConfig& config,
                              unsigned regs_per_thread = 0);

    // Auto-tune block dimensions
    LaunchConfig tune_block_dims(const ir::Function& kernel,
                                  const LaunchConfig& config);
};

} // namespace clunk::gpu
