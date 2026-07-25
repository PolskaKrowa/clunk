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
 * Clunk CostModel — per-architecture latency / throughput / port / uops tables.
 *
 * The CostModel is consumed by the EvaluationEngine to produce
 * architecture-aware scores. Each ir::Opcode maps to an OpCost record
 * describing the latency (cycles), inverse-throughput (cycles per
 * independent issue), port-pressure mask, and micro-op count for the
 * target architecture.
 *
 * Three concrete models are provided out of the box:
 *   - X86_64CostModel (Intel Skylake-class P-core)
 *   - AArch64CostModel (Cortex-A78-class)
 *   - PTXCostModel    (NVIDIA SM_80)
 * plus a GenericCostModel used when no arch is specified.
 *
 * TTI-style InstructionCost API:
 *
 *  In addition to the legacy flat `OpCost cost(ir::Opcode)` query (which
 *  returns a struct of {latency, throughput, port_mask, uops}), the
 *  CostModel exposes a richer per-kind API:
 *
 *      double cost(ir::Opcode op, CostKind kind,
 *                  const std::vector<OperandValueInfo>& operands = {}) const;
 *
 *  `CostKind` selects which cost axis is being asked for
 *  (ReciprocalThroughput / Latency / CodeSize). `OperandValueInfo`
 *  carries what the cost model knows about each operand (constant?
 *  power-of-two? zero?) — letting the model e.g. treat `mul x, 8` as
 *  a free LEA-foldable shift on x86_64 instead of a 3-uop IMUL.
 *
 *  Two tier-aware subclasses:
 *    - CheapIRCostModel  — thin wrapper used as the in-loop prune model.
 *                          Returns the same numbers as the underlying
 *                          per-arch model; exists as a marker class so
 *                          callers can ask for "the cheap tier"
 *                          explicitly.
 *    - UOpsCostModel     — expensive uOps-style final ranker. Counts
 *                          micro-ops per instruction (Agner-Fog-style
 *                          approximations). Used after search has
 *                          produced a small candidate set.
 *
 * Thread-safety: every CostModel method is const and reads only
 * immutable tables. A single shared instance can be used concurrently
 * from multiple EvaluationEngine instances.
 */
#include <cstdint>
#include <memory>
#include <vector>

#include "clunk/IR/Instruction.h"
#include "clunk/IR/Value.h"

namespace clunk::evaluator {

// ── Architecture tag ──────────────────────────────────────────────────────
enum class Arch : uint8_t {
    Generic,  // safe default — no arch specified
    X86_64,
    AArch64,
    PTX,
};

// ── CostKind enum ──────────────────────────────────────────────────────────
//
// Mirrors LLVM TargetTransformInfo::TargetCostKind. Selects which axis
// the per-kind cost() query is asking for. The default kind for the
// operand-aware overload is ReciprocalThroughput (this is what the
// legacy flat `cost(op)` returns via OpCost::throughput_cycles).
enum class CostKind : uint8_t {
    ReciprocalThroughput,  // uops / throughput (default)
    Latency,               // cycles until result is available
    CodeSize,              // bytes emitted
};

// ── OperandValueInfo ──────────────────────────────────────────────────────
//
// Carries what the cost model knows about an operand at cost-model
// time. Mirrors LLVM TTI's OperandValueProperties (OK_UniformConstantValue,
// OP_PowerOf2) but is a plain struct so it can be constructed by hand
// in tests as well as derived from an ir::Value via the
// `CostModel::operand_info()` helper.
struct OperandValueInfo {
    // True if the operand is a compile-time integer constant.
    bool is_constant = false;
    int64_t constant_value = 0;

    // Valid only when is_constant == true.
    bool is_power_of_two = false;  // e.g. 1, 2, 4, 8, ...
    bool is_zero = false;
    bool is_one = false;
    bool is_all_ones = false;      // -1 (all bits set)

    // Bitwidth of the operand (0 if unknown).
    unsigned bitwidth = 0;
};

// ── Per-opcode cost record ────────────────────────────────────────────────
struct OpCost {
    // Latency in cycles for a single execution (operand-ready → result-ready).
    double latency_cycles = 1.0;

    // Inverse throughput: cycles between two independent issues.
    // 1.0 = one per cycle, 0.5 = two per cycle, 4.0 = one every 4 cycles.
    double throughput_cycles = 1.0;

    // Bitmask of execution ports this op can use. Bit i corresponds to
    // port i. 0 = "unknown / not modelled" — treated as no constraint.
    uint16_t port_mask = 0;

    // Number of fused-domain micro-ops retired.
    uint16_t uops = 1;
};

// ── Memory-layer latencies (in cycles) ────────────────────────────────────
struct MemoryLatencies {
    double l1_load    = 4.0;
    double l2_load    = 12.0;
    double l3_load    = 36.0;
    double dram_load  = 200.0;
    double gpu_global = 400.0;
    double gpu_shared = 30.0;
    double gpu_local  = 100.0;
    double store_base = 1.0;  // L1 store-to-load forwarding latency
};

// ── Abstract cost model interface ─────────────────────────────────────────
class CostModel {
public:
    virtual ~CostModel() = default;

    // Architecture tag for this model.
    virtual Arch arch() const = 0;

    // Per-opcode cost. Returns a sensible default for unmodelled opcodes.
    virtual OpCost cost(ir::Opcode op) const = 0;

    // Memory latency table.
    virtual MemoryLatencies memory() const = 0;

    // Issue width: max independent uops issued per cycle.
    virtual unsigned issue_width() const = 0;

    // Number of distinct execution ports modelled (used by port-pressure
    // bound = max over ports of (Σ uops_for_port / port_throughput)).
    virtual unsigned num_ports() const = 0;

    // Human-readable arch name (e.g. "x86_64", "aarch64", "ptx", "generic").
    virtual const char* arch_name() const = 0;

    // ── TTI-style per-kind, operand-aware cost ──────────────────────────
    //
    // Returns a single `double` for the requested CostKind, taking
    // operand context into account where the model knows how (e.g.
    // `mul x, 8` on x86_64 is an LEA-foldable shift, not a 3-uop IMUL).
    //
    // The default implementation in CostModel.cpp falls back to the
    // flat `cost(op)` table:
    //   - ReciprocalThroughput → OpCost::throughput_cycles
    //   - Latency               → OpCost::latency_cycles
    //   - CodeSize              → OpCost::uops  (approximation: uops ≈ bytes)
    //
    // Arch-specific subclasses override this to apply operand-aware
    // discounts (see X86_64CostModel::cost(op, kind, operands)).
    virtual double cost(ir::Opcode op, CostKind kind,
                        const std::vector<OperandValueInfo>& operands = {}) const;

    // Convenience: cost in the default kind (ReciprocalThroughput) with
    // operand context. Non-virtual — calls the virtual 3-arg overload.
    double cost(ir::Opcode op,
                const std::vector<OperandValueInfo>& operands) const {
        return cost(op, CostKind::ReciprocalThroughput, operands);
    }

    // ── Derive OperandValueInfo from an IR value ────────────────────────
    //
    // Returns a populated OperandValueInfo for `v`. If `v` is an
    // ir::ConstantInt, the constant_value / is_power_of_two / is_zero /
    // is_one / is_all_ones fields are populated; otherwise only
    // bitwidth is set (from `v.type()->bit_width()` if available).
    //
    // Implemented as a static method so callers don't need a CostModel
    // instance to derive operand info.
    static OperandValueInfo operand_info(const ir::Value& v);
};

// ── Concrete models ───────────────────────────────────────────────────────
class GenericCostModel final : public CostModel {
public:
    Arch arch() const override { return Arch::Generic; }
    OpCost cost(ir::Opcode op) const override;
    MemoryLatencies memory() const override;
    unsigned issue_width() const override { return 2; }
    unsigned num_ports() const override { return 4; }
    const char* arch_name() const override { return "generic"; }
};

class X86_64CostModel final : public CostModel {
public:
    Arch arch() const override { return Arch::X86_64; }
    OpCost cost(ir::Opcode op) const override;
    MemoryLatencies memory() const override;
    unsigned issue_width() const override { return 4; }   // 4-wide decode/issue
    unsigned num_ports() const override { return 8; }      // ports 0..7
    const char* arch_name() const override { return "x86_64"; }

    // Operand-aware cost overrides. Applies the following
    // TTI-style discounts:
    //   - Mul  with a constant power-of-two operand  → shift cost
    //   - UDiv / SDiv with a constant power-of-two    → shift cost
    //   - Add / Sub / Xor / Or / And with constant 0  → 0 (no-op)
    //   - Mul with constant 1                          → 0 (identity)
    //   - Shl / LShr / AShr with constant 0 shift     → 0 (no-op)
    //
    // NOTE: the default argument is repeated here (not inherited from
    // the base) so the override can be invoked with two args through a
    // derived-class static type — C++ default-argument lookup uses the
    // static type's declaration, and the base's default would not be
    // visible through `X86_64CostModel::cost(op, kind)`.
    double cost(ir::Opcode op, CostKind kind,
                const std::vector<OperandValueInfo>& operands = {}) const override;
};

class AArch64CostModel final : public CostModel {
public:
    Arch arch() const override { return Arch::AArch64; }
    OpCost cost(ir::Opcode op) const override;
    MemoryLatencies memory() const override;
    unsigned issue_width() const override { return 4; }    // 4-wide issue
    unsigned num_ports() const override { return 4; }      // 2 ALU + LS + FP
    const char* arch_name() const override { return "aarch64"; }
};

class PTXCostModel final : public CostModel {
public:
    Arch arch() const override { return Arch::PTX; }
    OpCost cost(ir::Opcode op) const override;
    MemoryLatencies memory() const override;
    unsigned issue_width() const override { return 1; }   // per-lane scalar view
    unsigned num_ports() const override { return 4; }
    const char* arch_name() const override { return "ptx"; }
};

// ── Two-tier cost model subclasses ──────────────────────────────────────
//
// CheapIRCostModel:
//   Marker wrapper around the standard per-arch CostModel. Returns
//   EXACTLY the same numbers as the wrapped model; exists so callers
//   can explicitly request "the cheap IR-level prune tier".
//   All methods delegate to the wrapped CostModel.
//
class CheapIRCostModel final : public CostModel {
public:
    explicit CheapIRCostModel(std::shared_ptr<const CostModel> delegate)
        : delegate_(std::move(delegate)) {}

    Arch arch() const override { return delegate_->arch(); }
    OpCost cost(ir::Opcode op) const override { return delegate_->cost(op); }
    MemoryLatencies memory() const override { return delegate_->memory(); }
    unsigned issue_width() const override { return delegate_->issue_width(); }
    unsigned num_ports() const override { return delegate_->num_ports(); }
    const char* arch_name() const override { return delegate_->arch_name(); }

    double cost(ir::Opcode op, CostKind kind,
                const std::vector<OperandValueInfo>& operands) const override {
        return delegate_->cost(op, kind, operands);
    }

    const CostModel& delegate() const { return *delegate_; }

private:
    std::shared_ptr<const CostModel> delegate_;
};

// UOpsCostModel:
//   Expensive uOps-style final-rank cost model. Counts the
//   number of micro-operations each instruction would issue on the
//   target arch. For x86_64: most simple ALU ops are 1 uop; mul is 3;
//   div is 10-20; loads/stores are 1-2. Used as the final ranker
//   after search has produced a small set of candidates.
//
//   The numbers below are coarse approximations from Agner Fog's
//   instruction tables; absolute precision is not the goal — the
//   point is that this model returns uops, not cycles, as the final
//   ranking metric (Minotaur §4.3: uOps count beats cycle count).
class UOpsCostModel final : public CostModel {
public:
    explicit UOpsCostModel(Arch arch);

    Arch arch() const override { return arch_; }
    OpCost cost(ir::Opcode op) const override;
    MemoryLatencies memory() const override;
    unsigned issue_width() const override;
    unsigned num_ports() const override;
    const char* arch_name() const override;

    // uOps-style operand-aware cost. For the ReciprocalThroughput and
    // CodeSize kinds this returns the uops count (so a constant-foldable
    // mul by a power-of-two returns the shift's uops, not mul's). For
    // Latency it falls back to the underlying arch model's latency.
    //
    // NOTE: the default argument is repeated here for the same reason as
    // on X86_64CostModel::cost — see comment there.
    double cost(ir::Opcode op, CostKind kind,
                const std::vector<OperandValueInfo>& operands = {}) const override;

private:
    Arch arch_;
    // Underlying arch-specific model — used for memory latencies,
    // issue width, num ports, and the Latency cost-kind fallback.
    std::shared_ptr<const CostModel> delegate_;
};

// ── Factory ───────────────────────────────────────────────────────────────
//
// Returns a shared_ptr to a const CostModel. The const-ness signals
// that the model is immutable after construction; multiple EvaluationEngine
// instances can share the same model safely.
std::shared_ptr<const CostModel> make_cost_model(Arch arch);

// Map a clunk::pattern ArchDescriptor.name (or similar free-form arch
// string) to the appropriate CostModel. Falls back to Generic.
std::shared_ptr<const CostModel> make_cost_model_for_name(const std::string& name);

// ── Factory for the uOps-style final-rank model ──────────────────────────
//
// Same arch-name dispatch as make_cost_model_for_name, but returns a
// UOpsCostModel wrapping the per-arch delegate.
std::shared_ptr<const CostModel> make_uops_cost_model_for_name(const std::string& arch_name);

// Convenience: same dispatch as make_cost_model, but returns a UOpsCostModel.
std::shared_ptr<const CostModel> make_uops_cost_model(Arch arch);

} // namespace clunk::evaluator
