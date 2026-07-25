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
 * Clunk GPU Optimisation — PTX/ISA level optimisation and kernel launch tuning.
 */
#include "clunk/GPU/PTXOptimizer.h"

#include "clunk/GPU/ArchLimits.h"
#include "clunk/GPU/CUDADriver.h"
#include "clunk/GPU/DivergenceAnalysis.h"
#include "clunk/GPU/LivenessAnalysis.h"
#include "clunk/GPU/PTXEmitter.h"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

namespace clunk::gpu {

// ── File-local helpers ─────────────────────────────────────────────────────
//
// clone_instruction(): deep-copy an ir::Instruction. The IR has no built-in
// clone() (Instruction is `final` and not Clonable), so we synthesise the
// copy by hand.
//
static std::shared_ptr<ir::Instruction> clone_instruction(
    const std::shared_ptr<ir::Instruction>& src)
{
    if (!src) return nullptr;
    auto copy = std::make_shared<ir::Instruction>(src->opcode(),
                                                   src->type(),
                                                   src->name());
    for (const auto& op : src->operands()) {
        copy->add_operand(op);  // operands are shared_ptr<Value> — shared okay
    }
    for (const auto& [k, v] : src->metadata()) {
        copy->set_metadata(k, v);
    }
    copy->binop_flags() = src->binop_flags();
    if (src->alignment().has_value()) {
        copy->set_alignment(src->alignment().value());
    }
    copy->set_volatile(src->is_volatile());
    return copy;
}

// Build a fresh ir::Function with the same name/type/args/attributes as the
// input but with empty blocks. Passes call this then add_block/instruction
// to populate the result.
static std::shared_ptr<ir::Function> clone_function_skeleton(
    const ir::Function& src)
{
    auto out = std::make_shared<ir::Function>(src.name(),
                                               src.function_type(),
                                               src.linkage());
    for (const auto& arg : src.arguments()) {
        out->add_argument(arg.type, arg.name);
    }
    for (const auto& [k, v] : src.attributes()) {
        out->set_attribute(k, v);
    }
    return out;
}

// Safe metadata parse: returns default on any exception.
static unsigned safe_unsigned(const std::unordered_map<std::string,
                                                 std::string>& meta,
                               const std::string& key,
                               unsigned default_val)
{
    auto it = meta.find(key);
    if (it == meta.end()) return default_val;
    try {
        return static_cast<unsigned>(std::stoul(it->second));
    } catch (...) {
        return default_val;
    }
}

// ── PTXOptimizer ────────────────────────────────────────────────────────────

PTXOptimizer::PTXOptimizer(const pattern::ArchDescriptor& target_arch)
    : target_arch_(target_arch)
{
    // Derive the GpuArch from the ArchDescriptor's compute_capability
    // (the existing pipeline populates it; e.g. 80 → SM_80).
    if (target_arch_.is_gpu && target_arch_.compute_capability > 0) {
        GpuArch derived = arch_from_compute_capability(
            target_arch_.compute_capability);
        if (derived != GpuArch::Unknown) {
            arch_ = derived;
        }
    }
}

// ── optimise ────────────────────────────────────────────────────────────────

std::shared_ptr<ir::Function> PTXOptimizer::optimise(const ir::Function& kernel) {
    // Stage 1: Instruction scheduling — topological sort that hoists loads
    // past instructions that don't define their pointer operand.
    auto scheduled = schedule_instructions(kernel);

    // Stage 2: Memory coalescing — per-instruction coalesced annotations
    // (no longer mutates input).
    auto coalesced = optimise_memory_coalescing(*scheduled);

    // Stage 3: Register pressure reduction — uses real backward liveness.
    auto reduced = reduce_register_pressure(*coalesced);

    // Stage 4: FP precision analysis (diagnostic only; annotates function).
    analyse_fp_precision(*reduced);

    // Stage 5: PTX emission — emit PTX text for the optimised IR.
    // The PTX is stored as a function attribute so downstream consumers
    // (Pipeline, CUDADriver) can read it.
    PTXEmitter emitter(arch_);
    ptx_text_ = emitter.emit(*reduced);
    reduced->set_attribute("gpu.ptx_text_present", "true");
    reduced->set_attribute("gpu.ptx_reg_count",
                            std::to_string(emitter.num_registers()));
    reduced->set_attribute("gpu.arch", gpu_arch_name(arch_));

    return reduced;
}

// ── simulate_execution ──────────────────────────────────────────────────────
//
// Model warp execution. We mark a warp as diverged at a branch iff
// DivergenceAnalysis reports the branch condition as divergent; we
// reconverge at the immediate post-dominator (block.is_reconvergence).
//
std::vector<WarpState> PTXOptimizer::simulate_execution(
    const ir::Function& kernel,
    size_t num_warps) const
{
    std::vector<WarpState> warp_states(num_warps);
    for (size_t w = 0; w < num_warps; ++w) {
        warp_states[w].active_mask = 0xFFFFFFFF;
        warp_states[w].diverged = 0;
        warp_states[w].pc = 0;
        warp_states[w].registers.clear();
    }

    // Run divergence analysis to identify divergent branches and
    // reconvergence points.
    DivergenceAnalysis div;
    DivergenceResult div_result = div.analyse(kernel);

    // Walk blocks and instructions symbolically.
    for (auto& block : kernel.blocks()) {
        if (!block) continue;

        // If this block is a reconvergence point, restore all diverged
        // lanes to active (SIMT reconvergence at IPDOM).
        bool is_reconvergence = false;
        {
            auto it = div_result.is_reconvergence.find(block.get());
            if (it != div_result.is_reconvergence.end() && it->second) {
                is_reconvergence = true;
            }
        }
        if (is_reconvergence) {
            for (size_t w = 0; w < num_warps; ++w) {
                warp_states[w].active_mask |= warp_states[w].diverged;
                warp_states[w].diverged = 0;
            }
        }

        for (auto& inst : block->instructions()) {
            auto op = inst->opcode();

            // Track divergence at conditional branches
            if (op == ir::Opcode::Br && inst->num_operands() > 0) {
                // Look up the branch's divergence flag
                auto dit = div_result.divergent_branches.find(inst.get());
                bool divergent = (dit != div_result.divergent_branches.end() &&
                                  dit->second);
                if (divergent) {
                    // Half of the active lanes take the true path, the
                    // other half the false path. We model this by
                    // splitting the active mask.
                    for (size_t w = 0; w < num_warps; ++w) {
                        uint32_t act = warp_states[w].active_mask;
                        if (act == 0) continue;
                        // Move the lower half to "diverged" (false path)
                        uint32_t lower = act & 0x0000FFFF;
                        uint32_t upper = act & 0xFFFF0000;
                        warp_states[w].diverged |= lower;
                        warp_states[w].active_mask = upper;
                    }
                }
            }

            // Track register definitions
            if (inst->has_name()) {
                for (size_t w = 0; w < num_warps; ++w) {
                    warp_states[w].registers.push_back(0);
                }
            }

            // Load instructions add latency cycles
            if (op == ir::Opcode::Load) {
                for (size_t w = 0; w < num_warps; ++w) {
                    warp_states[w].pc += 4;
                }
            }

            for (size_t w = 0; w < num_warps; ++w) {
                warp_states[w].pc += 1;
            }
        }
    }

    return warp_states;
}

// ── analyse_memory_accesses ─────────────────────────────────────────────────

std::vector<MemoryTransaction> PTXOptimizer::analyse_memory_accesses(
    const ir::Function& kernel) const
{
    std::vector<MemoryTransaction> transactions;

    for (auto& block : kernel.blocks()) {
        if (!block) continue;

        for (auto& inst : block->instructions()) {
            auto op = inst->opcode();

            // Alloca is a stack-slot allocation, not a memory transaction.
            // Skip it.
            if (op == ir::Opcode::Alloca) continue;
            if (op != ir::Opcode::Load && op != ir::Opcode::Store) continue;

            MemoryTransaction txn;
            txn.type = (op == ir::Opcode::Load)
                          ? MemoryTransaction::Type::Load
                          : MemoryTransaction::Type::Store;

            // Determine address space from metadata.
            auto& meta = inst->metadata();
            if (meta.count("addrspace")) {
                unsigned as = safe_unsigned(meta, "addrspace", 0);
                switch (as) {
                    case 0: txn.address_space = MemoryTransaction::AddressSpace::Global; break;
                    case 1: txn.address_space = MemoryTransaction::AddressSpace::Global; break;
                    case 2: txn.address_space = MemoryTransaction::AddressSpace::Shared; break;
                    case 3: txn.address_space = MemoryTransaction::AddressSpace::Local;  break;
                    case 4: txn.address_space = MemoryTransaction::AddressSpace::Const;  break;
                    default: txn.address_space = MemoryTransaction::AddressSpace::Global; break;
                }
            } else {
                // Default: GPU kernels access global memory unless proven
                // otherwise. Inspect the pointer operand's type for an
                // address space.
                if (inst->num_operands() > 0) {
                    auto ptr_ty = inst->operand(0)->type();
                    if (ptr_ty && ptr_ty->is_pointer()) {
                        const auto* pty =
                            static_cast<const ir::PointerType*>(ptr_ty.get());
                        unsigned as = pty->address_space();
                        switch (as) {
                            case 0: txn.address_space = MemoryTransaction::AddressSpace::Global; break;
                            case 1: txn.address_space = MemoryTransaction::AddressSpace::Global; break;
                            case 2: txn.address_space = MemoryTransaction::AddressSpace::Shared; break;
                            case 3: txn.address_space = MemoryTransaction::AddressSpace::Local;  break;
                            case 4: txn.address_space = MemoryTransaction::AddressSpace::Const;  break;
                            default: txn.address_space = MemoryTransaction::AddressSpace::Global; break;
                        }
                    } else {
                        txn.address_space = kernel.is_gpu_kernel()
                            ? MemoryTransaction::AddressSpace::Global
                            : MemoryTransaction::AddressSpace::Local;
                    }
                } else {
                    txn.address_space = kernel.is_gpu_kernel()
                        ? MemoryTransaction::AddressSpace::Global
                        : MemoryTransaction::AddressSpace::Local;
                }
            }

            // Estimate bytes accessed.
            if (inst->type()) {
                txn.bytes = inst->type()->size_bytes();
                if (txn.bytes == 0) txn.bytes = 4;
            } else if (op == ir::Opcode::Store && inst->num_operands() >= 1) {
                // Store has no result type; use the value's type.
                txn.bytes = inst->operand(0)->type()->size_bytes();
                if (txn.bytes == 0) txn.bytes = 4;
            }

            // Per-instruction coalescing analysis.
            txn.coalesced = false;
            if (txn.address_space == MemoryTransaction::AddressSpace::Global) {
                // Coalesced iff metadata explicitly marks stride1, OR the
                // pointer name suggests a thread-id-derived stride.
                if (meta.count("access_pattern")) {
                    txn.coalesced = (meta.at("access_pattern") == "stride1");
                }
                if (!txn.coalesced && inst->num_operands() > 0) {
                    auto ptr = inst->operand(0);
                    if (ptr && ptr->has_name()) {
                        const std::string& pname = ptr->name();
                        if (pname.find("tid") != std::string::npos ||
                            pname.find("threadIdx") != std::string::npos) {
                            txn.coalesced = true;
                        }
                    }
                }
            } else if (txn.address_space == MemoryTransaction::AddressSpace::Shared) {
                // Shared memory accesses avoid bank conflicts with proper
                // padding; we mark them coalesced (with the caveat that
                // real bank-conflict analysis is a future improvement).
                txn.coalesced = true;
            } else if (txn.address_space == MemoryTransaction::AddressSpace::Const) {
                // Const memory broadcasts; effectively always coalesced.
                txn.coalesced = true;
            }
            // Local memory is per-thread; never coalesced.

            txn.warp_id = 0;

            transactions.push_back(txn);
        }
    }

    return transactions;
}

// ── estimate_divergence ─────────────────────────────────────────────────────

double PTXOptimizer::estimate_divergence(const ir::Function& kernel) const {
    // Use the real DivergenceAnalysis pass.
    DivergenceAnalysis div;
    DivergenceResult result = div.analyse(kernel);
    return result.branch_divergence_fraction();
}

// ── schedule_instructions ────────────────────────────────────────────────────
//
// Real topological sort within each block:
//
//   1. Build the intra-block def-use DAG: an instruction I depends on
//      instruction J iff J defines a value used by I (and both are in
//      the same block).
//   2. Schedule using a ready-list: instructions whose dependencies are
//      all already scheduled. Among ready instructions, prioritise Loads
//      (to hide latency), then preserve original order.
//   3. Terminators always go last.
//
// This guarantees that a Load whose pointer operand is defined by a
// preceding GEP cannot be hoisted above the GEP.
//
std::shared_ptr<ir::Function> PTXOptimizer::schedule_instructions(
    const ir::Function& kernel) const
{
    auto result = clone_function_skeleton(kernel);

    for (auto& block : kernel.blocks()) {
        if (!block) continue;
        auto& new_block = result->add_block(block->name());

        const auto& instrs = block->instructions();
        const size_t n = instrs.size();
        if (n == 0) continue;

        // Map: Value* (defined in this block) → position in instrs.
        std::unordered_map<const ir::Value*, size_t> def_pos;
        for (size_t i = 0; i < n; ++i) {
            // Only named instructions define values; constants and
            // arguments are not defined here.
            if (instrs[i]->has_name()) {
                def_pos[instrs[i].get()] = i;
            }
        }

        // Dependency set per instruction: positions of intra-block defs
        // of its operands.
        std::vector<std::vector<size_t>> deps(n);
        std::vector<bool> is_term(n, false);
        for (size_t i = 0; i < n; ++i) {
            is_term[i] = instrs[i]->is_terminator();
            for (const auto& op : instrs[i]->operands()) {
                if (!op) continue;
                auto it = def_pos.find(op.get());
                if (it != def_pos.end() && it->second != i) {
                    deps[i].push_back(it->second);
                }
            }
        }

        // Topological schedule: ready = all deps scheduled.
        std::vector<size_t> scheduled;
        scheduled.reserve(n);
        std::vector<bool> done(n, false);

        auto ready = [&](size_t i) -> bool {
            for (size_t d : deps[i]) {
                if (!done[d]) return false;
            }
            return true;
        };

        // Iterate until everything is scheduled. The outer loop is a
        // safety cap (n*n iterations is more than enough for any
        // well-formed block — a cycle would be a malformed IR).
        size_t safety = n * n + 16;
        while (scheduled.size() < n && safety-- > 0) {
            bool progressed = false;

            // Pass 1: schedule ready loads (latency hiding).
            for (size_t i = 0; i < n; ++i) {
                if (done[i] || is_term[i]) continue;
                if (instrs[i]->opcode() != ir::Opcode::Load) continue;
                if (!ready(i)) continue;
                scheduled.push_back(i);
                done[i] = true;
                progressed = true;
            }

            // Pass 2: schedule any ready non-terminator (original order).
            for (size_t i = 0; i < n; ++i) {
                if (done[i] || is_term[i]) continue;
                if (!ready(i)) continue;
                scheduled.push_back(i);
                done[i] = true;
                progressed = true;
            }

            if (!progressed) {
                // No ready non-terminator — schedule any ready terminator,
                // or any instruction at all (shouldn't happen for valid IR).
                for (size_t i = 0; i < n; ++i) {
                    if (done[i]) continue;
                    if (!ready(i)) continue;
                    scheduled.push_back(i);
                    done[i] = true;
                    progressed = true;
                    break;
                }
            }
            if (!progressed) {
                // Stuck on a cycle — break out and append the rest in
                // original order. (Defensive: malformed IR.)
                for (size_t i = 0; i < n; ++i) {
                    if (!done[i]) {
                        scheduled.push_back(i);
                        done[i] = true;
                    }
                }
                break;
            }
        }

        // Append instructions in scheduled order. Cloning ensures the
        // returned function owns fresh Instruction objects (no aliasing
        // with the input).
        for (size_t i : scheduled) {
            new_block.add_instruction(clone_instruction(instrs[i]));
        }
    }

    return result;
}

// ── optimise_memory_coalescing ───────────────────────────────────────────────

std::shared_ptr<ir::Function> PTXOptimizer::optimise_memory_coalescing(
    const ir::Function& kernel) const
{
    auto result = clone_function_skeleton(kernel);

    // Run per-instruction memory analysis.
    auto mem_txns = analyse_memory_accesses(kernel);

    // Build instruction → transaction index. analyse_memory_accesses
    // iterates the same block/instruction order as we do below, so we
    // walk in lockstep.
    size_t txn_idx = 0;
    std::unordered_map<const ir::Instruction*, size_t> txn_index_for_inst;
    for (auto& block : kernel.blocks()) {
        if (!block) continue;
        for (auto& inst : block->instructions()) {
            auto op = inst->opcode();
            if (op == ir::Opcode::Load || op == ir::Opcode::Store) {
                if (txn_idx < mem_txns.size()) {
                    txn_index_for_inst[inst.get()] = txn_idx;
                    ++txn_idx;
                }
            }
        }
    }

    // Count non-coalesced global accesses for the function-level summary.
    size_t non_coalesced_global = 0;
    for (auto& txn : mem_txns) {
        if (txn.address_space == MemoryTransaction::AddressSpace::Global &&
            !txn.coalesced) {
            ++non_coalesced_global;
        }
    }
    if (non_coalesced_global > 0) {
        result->set_attribute("gpu.shared_mem_suggested", "true");
        result->set_attribute("gpu.non_coalesced_count",
                              std::to_string(non_coalesced_global));
    }

    // Copy blocks with per-instruction coalescing annotations.
    for (auto& block : kernel.blocks()) {
        if (!block) continue;
        auto& new_block = result->add_block(block->name());
        for (auto& inst : block->instructions()) {
            auto op = inst->opcode();
            if (op == ir::Opcode::Load || op == ir::Opcode::Store) {
                // Clone the instruction.
                auto cloned = clone_instruction(inst);
                auto it = txn_index_for_inst.find(inst.get());
                if (it != txn_index_for_inst.end() &&
                    it->second < mem_txns.size()) {
                    const auto& txn = mem_txns[it->second];
                    cloned->set_metadata("coalescing_hint",
                                          txn.coalesced ? "coalesced"
                                                        : "consider_shared_mem");
                    cloned->set_metadata("addrspace_class",
                                          address_space_name(txn.address_space));
                } else {
                    cloned->set_metadata("coalescing_hint",
                                          "consider_shared_mem");
                }
                new_block.add_instruction(cloned);
            } else {
                new_block.add_instruction(clone_instruction(inst));
            }
        }
    }

    return result;
}

// ── reduce_register_pressure ─────────────────────────────────────────────────

std::shared_ptr<ir::Function> PTXOptimizer::reduce_register_pressure(
    const ir::Function& kernel) const
{
    auto result = clone_function_skeleton(kernel);

    // Use the real backward LivenessAnalysis pass to compute the maximum
    // simultaneously-live set across all program points in the function.
    LivenessAnalysis liveness;
    LiveSets live = liveness.compute(kernel);
    size_t max_live = live.function_max_live;

    result->set_attribute("gpu.register_pressure",
                          std::to_string(max_live));
    result->set_attribute("gpu.max_registers",
                          std::to_string(max_registers_));

    // If register pressure is below the per-thread budget, no changes
    // needed — just deep-copy the IR.
    if (max_live <= max_registers_) {
        for (auto& block : kernel.blocks()) {
            if (!block) continue;
            auto& new_block = result->add_block(block->name());
            for (auto& inst : block->instructions()) {
                new_block.add_instruction(clone_instruction(inst));
            }
        }
        return result;
    }

    // Above budget: identify cheap-to-recompute single-use values as
    // spill candidates. A value is "cheap to recompute" if it's a simple
    // arithmetic/shift op (Add, Sub, Shl, LShr, AShr) used exactly once.
    std::unordered_map<const ir::Value*, size_t> use_count;
    for (auto& block : kernel.blocks()) {
        for (auto& inst : block->instructions()) {
            for (auto& op : inst->operands()) {
                if (op && op->has_name()) {
                    use_count[op.get()]++;
                }
            }
        }
    }

    for (auto& block : kernel.blocks()) {
        if (!block) continue;
        auto& new_block = result->add_block(block->name());
        for (auto& inst : block->instructions()) {
            // Clone the instruction.
            auto cloned = clone_instruction(inst);

            if (inst->has_name()) {
                auto it = use_count.find(inst.get());
                size_t uses = (it != use_count.end()) ? it->second : 0;

                bool cheap_op = (inst->opcode() == ir::Opcode::Add ||
                                 inst->opcode() == ir::Opcode::Sub ||
                                 inst->opcode() == ir::Opcode::Shl ||
                                 inst->opcode() == ir::Opcode::LShr ||
                                 inst->opcode() == ir::Opcode::AShr);

                if (uses <= 1 && cheap_op) {
                    cloned->set_metadata("spill_candidate", "true");
                    cloned->set_metadata("spill_reason",
                                          "cheap_recompute_single_use");
                }
            }

            new_block.add_instruction(cloned);
        }
    }

    return result;
}

// ── analyse_fp_precision ────────────────────────────────────────────────────

void PTXOptimizer::analyse_fp_precision(const ir::Function& kernel) const {
    // Diagnostic pass: counts FP operations and potential precision
    // issues. Leaves the counts on the function as attributes so the
    // pipeline can surface them.
    size_t fp_ops = 0;
    size_t potential_precision_issues = 0;

    for (auto& block : kernel.blocks()) {
        if (!block) continue;
        for (auto& inst : block->instructions()) {
            auto op = inst->opcode();

            bool is_fp = (op == ir::Opcode::FAdd || op == ir::Opcode::FSub ||
                          op == ir::Opcode::FMul || op == ir::Opcode::FDiv ||
                          op == ir::Opcode::FRem);
            if (is_fp) {
                fp_ops++;
                if (op == ir::Opcode::FDiv) potential_precision_issues++;
                if (op == ir::Opcode::FRem) potential_precision_issues++;
            }
            if (op == ir::Opcode::FPToSI || op == ir::Opcode::FPToUI) {
                potential_precision_issues++;
            }
        }
    }

    // The caller in optimise() reads these counts and applies them to the
    // result function.
    const_cast<ir::Function&>(kernel).set_attribute(
        "gpu.fp_ops", std::to_string(fp_ops));
    const_cast<ir::Function&>(kernel).set_attribute(
        "gpu.fp_precision_warnings",
        std::to_string(potential_precision_issues));
}

} // namespace clunk::gpu
