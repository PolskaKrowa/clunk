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
 * Clunk Peephole Miner — SMT-verified enumerative superoptimisation.
 * See PeepholeMiner.h for the rationale and scope.
 */
#include "clunk/Search/PeepholeMiner.h"
#include "clunk/Search/SMTVerifier.h"
#include "clunk/Search/StochasticSearch.h"  // structural_hash
#include "clunk/Evaluator/Interpreter.h"    // concrete-execution pre-filter
#include "clunk/IR/Clone.h"                 // deep_copy_function
#include "clunk/IR/BasicBlock.h"
#include "clunk/IR/Instruction.h"
#include "clunk/IR/Module.h"
#include "clunk/IR/Type.h"
#include "clunk/IR/Value.h"

#include <algorithm>
#include <functional>
#include <limits>
#include <random>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace clunk::search {

namespace {

// Integer opcodes the enumerator emits. All are SMT-modelable (bitvector
// theory) and never introduce host UB during cost analysis (no div/rem).
constexpr ir::Opcode kOps[] = {
    ir::Opcode::Add, ir::Opcode::Sub, ir::Opcode::Mul,
    ir::Opcode::And, ir::Opcode::Or,  ir::Opcode::Xor,
    ir::Opcode::Shl, ir::Opcode::LShr, ir::Opcode::AShr,
};

bool is_allowed_binop(ir::Opcode op) {
    for (auto o : kOps) if (o == op) return true;
    return false;
}

// Division / remainder ops. These may appear in a mined SOURCE — so the miner
// can prove e.g. `udiv x, 8` -> `lshr x, 3` — but the enumerator never EMITS
// them: a div/rem costs far more than any bit-op, so it could never be the
// cheaper replacement. SMT models them soundly (Z3's div is total, so a
// div-by-zero UB input can only make the prover MISS a rewrite, never bless a
// wrong one).
constexpr ir::Opcode kDivRem[] = {
    ir::Opcode::UDiv, ir::Opcode::SDiv, ir::Opcode::URem, ir::Opcode::SRem,
};

bool is_div_rem(ir::Opcode op) {
    for (auto o : kDivRem) if (o == op) return true;
    return false;
}

// icmp predicates the compare-select synthesiser tries. Signed/unsigned
// less-than plus (in)equality; greater-than variants are reachable by swapping
// the operands, so this minimal set still covers min/max/clamp/sign-select.
constexpr ir::CmpPredicate kPreds[] = {
    ir::CmpPredicate::EQ,  ir::CmpPredicate::NE,
    ir::CmpPredicate::SLT, ir::CmpPredicate::SLE,
    ir::CmpPredicate::ULT, ir::CmpPredicate::ULE,
};

ir::ConstantInt* as_ci(const std::shared_ptr<ir::Value>& v) {
    return v ? dynamic_cast<ir::ConstantInt*>(v.get()) : nullptr;
}

// What mine_function needs to know about an eligible source function.
struct Eligible {
    unsigned bw = 0;
    std::vector<std::shared_ptr<ir::Value>> live_ins;  // argument leaves (by first use)
    std::vector<unsigned> live_in_widths;              // parallel to live_ins
    std::vector<int64_t> src_constants;
};

// Integer cast opcodes (trunc/zext/sext). Allowed in mined SOURCES with
// per-value width tracking, and emitted by the enumerator as single casts
// of off-width live-ins — the shape -O3's cast-heavy IR actually needs.
bool is_int_cast(ir::Opcode op) {
    return op == ir::Opcode::Trunc || op == ir::Opcode::ZExt ||
           op == ir::Opcode::SExt;
}

// Decide whether `fn` is a single-block, integer straight-line function ending
// in `ret <intN result>` (with N == bw >= 2), and extract its leaves /
// constants. Instructions may be the arithmetic/bitwise binops the enumerator
// emits, the div/rem family, `icmp` (yielding i1), or `select` (consuming an
// i1 cond); every non-condition operand is the single main width `bw`. Returns
// nullopt otherwise — the miner never touches functions it cannot fully model.
std::optional<Eligible> analyze(const ir::Function& fn) {
    if (fn.blocks().size() != 1) return std::nullopt;
    auto& block = fn.blocks().front();
    if (!block || block->size() < 2) return std::nullopt;  // need >=1 op + ret

    // Argument names + widths. Live-ins must reference an argument; widths let
    // us type-check i1 (icmp result / select cond) against the main width.
    std::unordered_set<std::string> arg_names;
    std::unordered_map<std::string, unsigned> width_of;  // args + in-block results
    for (auto& a : fn.arguments()) {
        if (a.name.empty() || !a.type) continue;
        arg_names.insert(a.name);
        width_of[a.name] = a.type->bit_width();
    }

    // The terminator must be `ret <named value>`.
    auto term = block->instruction(block->size() - 1);
    if (!term || term->opcode() != ir::Opcode::Ret || term->num_operands() != 1)
        return std::nullopt;
    auto out = term->operand(0);
    if (!out || !out->has_name() || !out->type()) return std::nullopt;
    const unsigned bw = out->type()->bit_width();
    if (bw < 2) return std::nullopt;  // i1-returning sources are out of scope (v1)

    Eligible e;
    e.bw = bw;
    std::unordered_set<std::string> defined;       // in-block instruction results
    std::unordered_set<std::string> live_in_seen;  // dedup leaves
    std::unordered_set<int64_t> const_seen;

    // Resolve an operand's bit-width, or 0 if it cannot be resolved (an
    // undefined name). Constants of unspecified width take the main width.
    auto operand_width = [&](const std::shared_ptr<ir::Value>& op) -> unsigned {
        if (auto ci = as_ci(op)) return ci->bit_width() == 0 ? bw : ci->bit_width();
        if (op->has_name()) {
            auto it = width_of.find(op->name());
            if (it != width_of.end()) return it->second;
        }
        return 0;
    };

    const size_t n_ops = block->size() - 1;  // excluding the ret
    for (size_t i = 0; i < n_ops; ++i) {
        auto inst = block->instruction(i);
        if (!inst || !inst->has_name() || !inst->type()) return std::nullopt;
        const ir::Opcode op = inst->opcode();
        const unsigned rw = inst->type()->bit_width();

        // Per-opcode type check with PER-VALUE widths (multi-width slices —
        // casts introduce values narrower/wider than the return width).
        // Binops need both operands at the result width; icmp needs equal-
        // width operands and yields i1; select consumes an i1 condition and
        // width-rw arms; casts need the LLVM direction invariant.
        if (op == ir::Opcode::ICmp) {
            if (inst->num_operands() != 2 || rw != 1) return std::nullopt;
            const unsigned w0 = operand_width(inst->operand(0));
            if (w0 == 0 || w0 != operand_width(inst->operand(1)))
                return std::nullopt;
        } else if (op == ir::Opcode::Select) {
            if (inst->num_operands() != 3 || rw < 1) return std::nullopt;
            if (operand_width(inst->operand(0)) != 1) return std::nullopt;   // cond i1
            if (operand_width(inst->operand(1)) != rw ||
                operand_width(inst->operand(2)) != rw) return std::nullopt;
        } else if (is_allowed_binop(op) || is_div_rem(op)) {
            if (!inst->is_binary_op() || inst->num_operands() != 2 || rw < 1)
                return std::nullopt;
            if (operand_width(inst->operand(0)) != rw ||
                operand_width(inst->operand(1)) != rw) return std::nullopt;
        } else if (is_int_cast(op)) {
            if (inst->num_operands() != 1 || rw < 1) return std::nullopt;
            const unsigned ow = operand_width(inst->operand(0));
            if (ow == 0) return std::nullopt;
            if (op == ir::Opcode::Trunc ? (ow <= rw) : (ow >= rw))
                return std::nullopt;  // malformed cast direction
        } else {
            return std::nullopt;  // opcode outside the modelled set
        }

        // Register operands as constants / argument live-ins. A named operand
        // is either an in-block SSA result (already defined — e.g. an icmp cond)
        // or a function-argument live-in (any width — the width is recorded so
        // the enumerator and the test-vector canonicalisation line up).
        for (auto& opnd : inst->operands()) {
            if (!opnd) return std::nullopt;
            if (auto ci = as_ci(opnd)) {
                if (const_seen.insert(ci->value()).second)
                    e.src_constants.push_back(ci->value());
                continue;
            }
            if (!opnd->has_name()) return std::nullopt;
            if (defined.count(opnd->name())) continue;  // in-block SSA value
            if (!arg_names.count(opnd->name())) return std::nullopt;
            const unsigned lw = operand_width(opnd);
            if (lw == 0) return std::nullopt;
            if (live_in_seen.insert(opnd->name()).second) {
                e.live_ins.push_back(opnd);
                e.live_in_widths.push_back(lw);
            }
        }
        width_of[inst->name()] = rw;
        defined.insert(inst->name());
    }

    // The returned value must be an in-block result (otherwise the function is
    // already trivial — a bare `ret %arg` / `ret const` — nothing to mine).
    if (!defined.count(out->name())) return std::nullopt;
    if (n_ops > 0 && e.live_ins.empty()) return std::nullopt;  // constant function
    return e;
}

// Build the constant operand pool: source constants first (most likely
// useful), then {0,1,2,-1}, then small powers of two, then user extras;
// deduplicated and capped.
std::vector<std::shared_ptr<ir::Value>> constant_pool(
    const Eligible& e, const MinerConfig& cfg) {
    std::vector<int64_t> vals;
    std::unordered_set<int64_t> seen;
    auto push = [&](int64_t v) {
        if (seen.insert(v).second) vals.push_back(v);
    };
    for (int64_t c : e.src_constants) push(c);
    push(0); push(1); push(2); push(-1); push(-2);
    // Powers of two, their low-bit masks (2^k - 1), and negations — the
    // constants bit-twiddling wins most often need.
    for (int64_t p = 1; p <= (int64_t(1) << 30); p <<= 1) {
        push(p); push(p - 1); push(-p);
    }
    for (int64_t x : cfg.extra_constants) push(x);
    if (vals.size() > cfg.max_constants) vals.resize(cfg.max_constants);

    std::vector<std::shared_ptr<ir::Value>> pool;
    auto ty = std::make_shared<ir::IntegerType>(e.bw);
    for (int64_t v : vals)
        pool.push_back(std::make_shared<ir::ConstantInt>(ty, v));
    return pool;
}

// Assemble a candidate function: source signature/args + an entry block of the
// enumerated instructions + `ret output`. Instruction objects are shared with
// the enumerator (read-only after construction), so no cloning is needed.
std::shared_ptr<ir::Function> build_function(
    const ir::Function& src,
    const std::vector<std::shared_ptr<ir::Instruction>>& instrs,
    const std::shared_ptr<ir::Value>& output) {
    auto fn = std::make_shared<ir::Function>(
        src.name(), src.function_type(), src.linkage());
    for (auto& a : src.arguments()) fn->add_argument(a.type, a.name);
    auto& bb = fn->add_block("entry");
    for (auto& in : instrs) bb.add_instruction(in);
    bb.add_instruction(ir::inst::make_ret(output));
    return fn;
}

// Enumeration context threaded through the recursion.
struct EnumCtx {
    const ir::Function& src;
    unsigned bw;
    evaluator::EvaluationEngine* engine;
    const MinerConfig& cfg;
    double src_score;
    uint64_t src_hash;
    // Concrete-execution pre-filter (STOKE/Souper style): a candidate must
    // agree with the source on every test vector before we spend an SMT call
    // on it. This is what keeps the SMT budget from being burned on the many
    // cheap-but-wrong single-op candidates (add x,0; or x,0; ...).
    const std::vector<std::vector<int64_t>>* tests = nullptr;
    const std::vector<std::optional<int64_t>>* src_out = nullptr;
    size_t candidates = 0;  // total leaves built (against cfg.max_candidates)
    // Cheaper-than-source, structurally-novel, test-passing candidates.
    std::vector<std::pair<std::shared_ptr<ir::Function>, double>> cheaper;
};

// True iff `cand` produces the SAME result as the source on every test vector
// where the source produced a value. A disagreement proves non-equivalence
// (skip SMT); if the interpreter can't evaluate the candidate, keep it and let
// SMT decide (return true).
bool passes_concrete_tests(const EnumCtx& ctx, const ir::Function& cand) {
    if (!ctx.tests || !ctx.src_out) return true;
    // Compare outputs modulo the result width: the interpreter zero-extends
    // masked results but returns arguments sign-extended, so a pass-through
    // (`ret %x`) and the source can carry the same bwN bit pattern in
    // different int64 representations. Comparing the low bw bits makes the
    // filter representation-independent (and matches BV equivalence).
    const unsigned bw = ctx.bw;
    auto low = [bw](int64_t v) -> uint64_t {
        if (bw == 0 || bw >= 64) return static_cast<uint64_t>(v);
        return static_cast<uint64_t>(v) & ((uint64_t(1) << bw) - 1);
    };
    for (size_t k = 0; k < ctx.tests->size(); ++k) {
        const auto& expected = (*ctx.src_out)[k];
        if (!expected) continue;  // source undefined here — uninformative
        auto got = evaluator::Interpreter::interpret(cand, (*ctx.tests)[k]);
        if (!got) return true;    // can't evaluate — defer to SMT
        if (low(*got) != low(*expected)) return false;
    }
    return true;
}

// Keep `cand` iff it is strictly cheaper than the source, structurally novel,
// and behaves identically on every test vector — the only candidates worth an
// SMT call. Shared by the length-0 (bare leaf/constant) path and the
// enumerator.
void consider(EnumCtx& ctx, std::shared_ptr<ir::Function> cand) {
    if (ctx.candidates >= ctx.cfg.max_candidates) return;
    ++ctx.candidates;
    double sc = ctx.engine->analyse(*cand).score;
    if (sc > ctx.src_score + 1e-9 &&
        StochasticSearch::structural_hash(*cand) != ctx.src_hash &&
        passes_concrete_tests(ctx, *cand)) {
        ctx.cheaper.emplace_back(std::move(cand), sc);
    }
}

// Recursively build straight-line programs of exactly `target_len` instructions
// (output = the last one). `pool` is the operand pool (leaves + results built
// so far); `instrs` is the program prefix.
// ── Souper-style enumeration pruning ────────────────────────────────────────
// Every rule below skips only candidates for which an EQUAL-OR-SHORTER
// equivalent is also enumerated (identities reduce to the L0 pass-through
// leaf; constant results reduce to an L0 constant leaf — {0,1,2,-1} are
// always in the pool). Shortest-first search therefore loses no wins, and
// the candidate budget covers far more distinct space: the blind L2/L3
// cross-product used to burn its 200k cap mostly on x+0 / x^x / c1+c2
// noise.

bool is_commutative(ir::Opcode op) {
    return op == ir::Opcode::Add || op == ir::Opcode::Mul ||
           op == ir::Opcode::And || op == ir::Opcode::Or ||
           op == ir::Opcode::Xor;
}

// Skip (op, lhs, rhs) if it is an identity / constant / dead form. `bw` is
// the enumeration width (constants compare modulo bw).
bool prune_binop(ir::Opcode op,
                 const std::shared_ptr<ir::Value>& lhs,
                 const std::shared_ptr<ir::Value>& rhs,
                 unsigned bw) {
    auto cl = as_ci(lhs);
    auto cr = as_ci(rhs);
    if (cl && cr) return true;  // const-const folds to an L0 constant leaf
    auto canon = [bw](int64_t v) -> int64_t {
        if (bw == 0 || bw >= 64) return v;
        const uint64_t m = (uint64_t(1) << bw) - 1;
        uint64_t r = static_cast<uint64_t>(v) & m;
        if (r & (uint64_t(1) << (bw - 1))) r |= ~m;
        return static_cast<int64_t>(r);
    };
    if (lhs.get() == rhs.get()) {
        // x-x / x^x are constants; x&x / x|x are pass-throughs.
        if (op == ir::Opcode::Sub || op == ir::Opcode::Xor ||
            op == ir::Opcode::And || op == ir::Opcode::Or)
            return true;
    }
    if (cr) {
        const int64_t v = canon(cr->value());
        if (v == 0) return true;  // identity for add/sub/or/xor/shifts; const for mul/and
        if (op == ir::Opcode::Mul && v == 1) return true;   // identity
        if (op == ir::Opcode::And && v == -1) return true;  // identity
        if (op == ir::Opcode::Or  && v == -1) return true;  // const -1
        // Shift by >= bw: poison in LLVM (and useless) — never emit.
        if ((op == ir::Opcode::Shl || op == ir::Opcode::LShr ||
             op == ir::Opcode::AShr) &&
            static_cast<uint64_t>(cr->value()) >= bw)
            return true;
    }
    if (cl) {
        const int64_t v = canon(cl->value());
        // 0 op x: identity/const for everything except Sub (0-x = negation).
        if (v == 0 && op != ir::Opcode::Sub) return true;
        if (op == ir::Opcode::Mul && v == 1) return true;
        if (op == ir::Opcode::And && v == -1) return true;
        if (op == ir::Opcode::Or  && v == -1) return true;
    }
    return false;
}

void enumerate(EnumCtx& ctx,
               const std::vector<std::shared_ptr<ir::Value>>& pool,
               std::vector<std::shared_ptr<ir::Instruction>>& instrs,
               size_t target_len) {
    if (instrs.size() == target_len) {
        // Connectivity pruning: every intermediate instruction must feed the
        // output transitively, or the candidate carries dead code — the same
        // program minus the dead instruction is enumerated at a shorter
        // length, so this one can never be the minimal win.
        if (instrs.size() > 1) {
            std::unordered_set<const ir::Value*> live{instrs.back().get()};
            for (size_t k = instrs.size(); k-- > 0;) {
                if (!live.count(instrs[k].get())) return;
                for (auto& op : instrs[k]->operands()) live.insert(op.get());
            }
        }
        consider(ctx, build_function(ctx.src, instrs, instrs.back()));
        return;
    }

    auto ty = std::make_shared<ir::IntegerType>(ctx.bw);
    const std::string name = "t" + std::to_string(instrs.size());
    for (auto op : kOps) {
        const bool comm = is_commutative(op);
        for (size_t i = 0; i < pool.size(); ++i) {
            for (size_t j = comm ? i : 0; j < pool.size(); ++j) {
                if (ctx.candidates >= ctx.cfg.max_candidates) return;
                if (prune_binop(op, pool[i], pool[j], ctx.bw)) continue;
                auto inst = std::make_shared<ir::Instruction>(op, ty, name);
                inst->add_operand(pool[i]);
                inst->add_operand(pool[j]);
                std::vector<std::shared_ptr<ir::Value>> pool2 = pool;
                pool2.push_back(inst);
                instrs.push_back(inst);
                enumerate(ctx, pool2, instrs, target_len);
                instrs.pop_back();
            }
        }
    }
}

// Targeted synthesis of the `select(icmp pred a, b, t, f)` shape (signed/
// unsigned min/max, clamp, sign-select). Blind enumeration can't reach these
// affordably: it would need length 2 AND an icmp step, but the binop
// cross-product at length 2 exhausts the candidate budget long before any
// comparison is built. This pass constructs exactly the compare-select shape
// from a small operand set, so those wins are reachable. It reuses `consider`,
// so only strictly-cheaper, novel, test-passing candidates reach SMT.
void synth_compare_select(EnumCtx& ctx,
                          const std::vector<std::shared_ptr<ir::Value>>& cmp_pool,
                          const std::vector<std::shared_ptr<ir::Value>>& val_pool) {
    auto tyW = std::make_shared<ir::IntegerType>(ctx.bw);
    auto tyB = std::make_shared<ir::IntegerType>(1);
    for (size_t a = 0; a < cmp_pool.size(); ++a) {
        for (size_t b = 0; b < cmp_pool.size(); ++b) {
            if (a == b) continue;  // icmp x,x is a constant — nothing to select on
            for (auto pred : kPreds) {
                if (ctx.candidates >= ctx.cfg.max_candidates) return;
                auto cmp = std::make_shared<ir::Instruction>(ir::Opcode::ICmp, tyB, "t0");
                cmp->add_operand(cmp_pool[a]);
                cmp->add_operand(cmp_pool[b]);
                cmp->set_metadata("pred", std::to_string(static_cast<unsigned>(pred)));
                for (auto& t : val_pool) {
                    for (auto& f : val_pool) {
                        if (ctx.candidates >= ctx.cfg.max_candidates) return;
                        auto sel = std::make_shared<ir::Instruction>(
                            ir::Opcode::Select, tyW, "t1");
                        sel->add_operand(cmp);
                        sel->add_operand(t);
                        sel->add_operand(f);
                        std::vector<std::shared_ptr<ir::Instruction>> instrs{cmp, sel};
                        consider(ctx, build_function(ctx.src, instrs, sel));
                    }
                }
            }
        }
    }
}

// Small deduplicated operand set for compare-select synthesis: argument leaves
// first, then a few pivotal constants (0, -1) and the source's own constants.
// Kept small (blow-up is quadratic in this set) — the high-value wins compare
// inputs against each other or against 0.
std::vector<std::shared_ptr<ir::Value>> compare_pool(
    const Eligible& e, size_t cap) {
    // Only main-width live-ins can appear as icmp/select operands at bw.
    std::vector<std::shared_ptr<ir::Value>> pool;
    for (size_t i = 0; i < e.live_ins.size(); ++i)
        if (e.live_in_widths.size() <= i || e.live_in_widths[i] == e.bw)
            pool.push_back(e.live_ins[i]);
    auto ty = std::make_shared<ir::IntegerType>(e.bw);
    std::unordered_set<int64_t> seen;
    auto push_c = [&](int64_t v) {
        if (pool.size() >= cap) return;
        if (seen.insert(v).second)
            pool.push_back(std::make_shared<ir::ConstantInt>(ty, v));
    };
    push_c(0); push_c(-1);
    for (int64_t c : e.src_constants) push_c(c);
    if (pool.size() > cap) pool.resize(cap);
    return pool;
}

// ── Multi-block slice mining ─────────────────────────────────────────────────
//
// mine_and_rewrite lifts a pure-integer, single-use, straight-line slice out of
// ONE block of an arbitrary function, mines it as a mini-function (reusing the
// whole single-block machinery above), and splices the proven-cheaper
// equivalent back — leaving the surrounding blocks, memory ops and loops
// untouched. Soundness rests on each slice's own SMT proof plus a whole-
// function SMT re-verification of the result against the original.

// A slice lifted out of one block.
struct Window {
    std::string block_name;
    std::string root_name;
    std::vector<std::shared_ptr<ir::Instruction>> instrs;  // block order, root last
    std::unordered_set<std::string> names;                 // all instr names
    std::vector<std::shared_ptr<ir::Value>> live_ins;      // ordered, deduped, named
};

// Integer op the miner can model inside a slice.
bool is_sliceable_op(const ir::Instruction& in) {
    return is_allowed_binop(in.opcode()) || is_div_rem(in.opcode()) ||
           in.opcode() == ir::Opcode::ICmp || in.opcode() == ir::Opcode::Select ||
           is_int_cast(in.opcode());
}

// Clone an instruction, remapping each operand through `remap` and assigning a
// new name. Preserves metadata / flags / alignment / volatility.
std::shared_ptr<ir::Instruction> clone_remap(
    const ir::Instruction& oi, const std::string& new_name,
    const std::function<std::shared_ptr<ir::Value>(const std::shared_ptr<ir::Value>&)>& remap) {
    auto ci = std::make_shared<ir::Instruction>(oi.opcode(), oi.type(), new_name);
    for (auto& op : oi.operands()) ci->add_operand(remap(op));
    for (auto& [k, v] : oi.metadata()) ci->set_metadata(k, v);
    ci->binop_flags() = oi.binop_flags();
    if (oi.alignment()) ci->set_alignment(oi.alignment().value());
    ci->set_volatile(oi.is_volatile());
    return ci;
}

// Extract the maximal single-use integer slice rooted at `root`. Folds in an
// operand-defining instruction iff it is in the same block, before the root, a
// sliceable op, and used EXACTLY once in the whole function (so removing it
// orphans nothing). Everything else becomes a named live-in. Returns nullopt if
// the slice is trivial, too large, or references an unnamed non-constant.
std::optional<Window> extract_window(
    const ir::BasicBlock& block,
    const std::shared_ptr<ir::Instruction>& root,
    const std::unordered_map<std::string, size_t>& use_count,
    const std::unordered_map<std::string, size_t>& pos_in_block,
    size_t max_instrs) {
    if (!root->has_name() || !root->type() || root->type()->bit_width() < 2)
        return std::nullopt;
    if (!is_sliceable_op(*root)) return std::nullopt;
    const size_t root_pos = pos_in_block.at(root->name());

    std::unordered_set<std::string> in_window{root->name()};
    std::vector<std::shared_ptr<ir::Instruction>> stack{root};
    while (!stack.empty()) {
        auto cur = stack.back();
        stack.pop_back();
        for (auto& op : cur->operands()) {
            auto oi = std::dynamic_pointer_cast<ir::Instruction>(op);
            if (!oi || !oi->has_name() || in_window.count(oi->name())) continue;
            auto pit = pos_in_block.find(oi->name());
            if (pit == pos_in_block.end() || pit->second >= root_pos) continue;
            auto uit = use_count.find(oi->name());
            if (uit == use_count.end() || uit->second != 1) continue;  // multi-use
            if (!is_sliceable_op(*oi)) continue;
            if (in_window.size() >= max_instrs) continue;  // cap: leave as live-in
            in_window.insert(oi->name());
            stack.push_back(oi);
        }
    }

    Window w;
    w.block_name = block.name();
    w.root_name = root->name();
    for (size_t i = 0; i < block.size(); ++i) {
        auto inst = block.instruction(i);
        if (inst && inst->has_name() && in_window.count(inst->name())) {
            w.instrs.push_back(inst);
            w.names.insert(inst->name());
        }
    }
    if (w.instrs.size() < 2) return std::nullopt;

    std::unordered_set<const ir::Value*> li_seen;
    for (auto& in : w.instrs) {
        for (auto& op : in->operands()) {
            if (!op) return std::nullopt;
            auto oi = std::dynamic_pointer_cast<ir::Instruction>(op);
            if (oi && w.names.count(oi->name())) continue;         // internal edge
            if (dynamic_cast<ir::ConstantInt*>(op.get())) continue; // embedded const
            if (!op->has_name()) return std::nullopt;               // undef/poison — bail
            if (li_seen.insert(op.get()).second) w.live_ins.push_back(op);
        }
    }
    return w;
}

// Build a standalone single-block mini-function computing the slice's root from
// its live-ins (turned into arguments arg0..argN). Internal operand edges keep
// pointer identity; live-in edges are rewired to the fresh argument values.
std::shared_ptr<ir::Function> build_mini_function(const Window& w) {
    std::unordered_map<const ir::Value*, std::shared_ptr<ir::Value>> li_to_arg;
    std::vector<std::shared_ptr<ir::Type>> arg_types;
    for (size_t i = 0; i < w.live_ins.size(); ++i) {
        auto av = std::make_shared<ir::Value>(w.live_ins[i]->type(),
                                              "arg" + std::to_string(i));
        li_to_arg[w.live_ins[i].get()] = av;
        arg_types.push_back(w.live_ins[i]->type());
    }
    auto root_ty = w.instrs.back()->type();
    auto fnty = std::make_shared<ir::FunctionType>(root_ty, arg_types);
    auto mini = std::make_shared<ir::Function>("slice", fnty, ir::Linkage::Internal);
    for (size_t i = 0; i < w.live_ins.size(); ++i)
        mini->add_argument(arg_types[i], "arg" + std::to_string(i));
    auto& bb = mini->add_block("entry");

    std::unordered_map<const ir::Value*, std::shared_ptr<ir::Value>> clone_of;
    for (auto& oi : w.instrs) {
        auto ci = clone_remap(*oi, oi->name(),
            [&](const std::shared_ptr<ir::Value>& op) -> std::shared_ptr<ir::Value> {
                auto c = clone_of.find(op.get());
                if (c != clone_of.end()) return c->second;
                auto a = li_to_arg.find(op.get());
                if (a != li_to_arg.end()) return a->second;
                return op;  // constant
            });
        clone_of[oi.get()] = ci;
        bb.add_instruction(ci);
    }
    bb.add_instruction(ir::inst::make_ret(clone_of[w.instrs.back().get()]));
    return mini;
}

// Rewrite every operand referencing `old_name` in `fn` to `repl` (by name-based
// SSA — matches how the analysis/SMT resolve values). Used for length-0
// (pass-through) splices where the root collapses to a live-in / constant.
void rewrite_uses(ir::Function& fn, const std::string& old_name,
                  const std::shared_ptr<ir::Value>& repl) {
    for (auto& block : fn.blocks()) {
        if (!block) continue;
        for (auto& inst : block->instructions()) {
            if (!inst) continue;
            for (size_t i = 0; i < inst->num_operands(); ++i) {
                auto op = inst->operand(i);
                if (op && op->has_name() && op->name() == old_name)
                    inst->set_operand(i, repl);
            }
        }
    }
}

// Splice a mined replacement into `out` in place of window `w`. `out` must be a
// copy of the function `w` was extracted from (names line up). Returns false if
// the block/replacement shape is unexpected.
bool splice_inplace(ir::Function& out, const Window& w,
                    const ir::Function& replacement, size_t& fresh_counter) {
    auto B = out.block(w.block_name);
    if (!B) return false;
    auto& rbb = replacement.blocks().front();
    if (!rbb || rbb->empty()) return false;
    auto rret = rbb->instruction(rbb->size() - 1);
    if (!rret || rret->opcode() != ir::Opcode::Ret || rret->num_operands() != 1)
        return false;

    // argK name -> the live-in it maps to (a by-name ref; constants pass through
    // the replacement unchanged and never appear as live-ins).
    std::unordered_map<std::string, std::shared_ptr<ir::Value>> arg_val;
    for (size_t i = 0; i < w.live_ins.size(); ++i) {
        arg_val["arg" + std::to_string(i)] =
            std::make_shared<ir::Value>(w.live_ins[i]->type(), w.live_ins[i]->name());
    }
    auto map_operand = [&](const std::shared_ptr<ir::Value>& op,
                           const std::unordered_map<const ir::Value*,
                               std::shared_ptr<ir::Value>>& clones)
        -> std::shared_ptr<ir::Value> {
        auto c = clones.find(op.get());
        if (c != clones.end()) return c->second;                 // replacement-internal
        if (op->has_name()) {
            auto a = arg_val.find(op->name());
            if (a != arg_val.end()) return a->second;            // argK -> live-in
        }
        return op;  // constant
    };

    // Clone the replacement body (all but the ret). The final replacement value
    // takes over the root's name so existing uses of %root resolve unchanged.
    const size_t n_body = rbb->size() - 1;
    auto routput = rret->operand(0);

    std::vector<std::shared_ptr<ir::Instruction>> repl_seq;
    std::unordered_map<const ir::Value*, std::shared_ptr<ir::Value>> clones;
    for (size_t i = 0; i < n_body; ++i) {
        auto ro = rbb->instruction(i);
        // The body instruction that produces the returned value takes over the
        // root's name so external uses of %root resolve to it; the rest get
        // fresh, collision-free names.
        const bool produces_output = (routput.get() == ro.get());
        std::string nm = produces_output
                             ? w.root_name
                             : "_sl" + std::to_string(fresh_counter++) + "_" + ro->name();
        auto ci = clone_remap(*ro, nm, [&](const std::shared_ptr<ir::Value>& op) {
            return map_operand(op, clones);
        });
        clones[ro.get()] = ci;
        repl_seq.push_back(ci);
    }

    // If the replacement returns a live-in / constant directly (length-0 or the
    // output isn't one of the body instructions), collapse the root to it.
    const bool output_is_body =
        clones.find(routput.get()) != clones.end();
    std::shared_ptr<ir::Value> collapsed_to;
    if (!output_is_body) {
        // routput is an argK ref or a constant.
        if (routput->has_name()) {
            auto a = arg_val.find(routput->name());
            collapsed_to = (a != arg_val.end()) ? a->second : routput;
        } else {
            collapsed_to = routput;  // constant
        }
    }

    // Window instructions that still have users OUTSIDE the window must keep
    // their definitions (multi-use harvesting pulls shared sub-expressions
    // into the slice for SMT precision; the shared value itself stays live
    // for its other users). Compute the kept set as a fixed point: an
    // externally-used window instruction is kept, and anything a kept
    // instruction references transitively must be kept too. The root is
    // never in the set — its name is taken over by the replacement.
    std::unordered_set<std::string> kept;
    for (auto& blk : out.blocks()) {
        if (!blk) continue;
        for (auto& inst : blk->instructions()) {
            if (!inst) continue;
            if (inst->has_name() && w.names.count(inst->name())) continue;  // inside
            for (auto& op : inst->operands())
                if (op && op->has_name() && w.names.count(op->name()) &&
                    op->name() != w.root_name)
                    kept.insert(op->name());
        }
    }
    bool grew = true;
    while (grew) {
        grew = false;
        for (auto& wi : w.instrs) {
            if (!wi || !kept.count(wi->name())) continue;
            for (auto& op : wi->operands())
                if (op && op->has_name() && w.names.count(op->name()) &&
                    op->name() != w.root_name && kept.insert(op->name()).second)
                    grew = true;
        }
    }

    // Rebuild the block: drop the window's dead non-root instructions (keep
    // the externally-used ones); at the root, insert the replacement body
    // (or nothing, collapsing uses).
    std::vector<std::shared_ptr<ir::Instruction>> rebuilt;
    for (size_t i = 0; i < B->size(); ++i) {
        auto inst = B->instruction(i);
        if (!inst) { rebuilt.push_back(inst); continue; }
        const std::string& nm = inst->has_name() ? inst->name() : std::string();
        if (!nm.empty() && nm == w.root_name) {
            if (output_is_body) {
                for (auto& r : repl_seq) rebuilt.push_back(r);
            } else {
                // Collapse: rewrite external uses of %root to the live-in/const.
                rewrite_uses(out, w.root_name, collapsed_to);
            }
            continue;  // original root dropped
        }
        if (!nm.empty() && w.names.count(nm)) {
            if (kept.count(nm)) rebuilt.push_back(inst);  // still used elsewhere
            continue;
        }
        rebuilt.push_back(inst);
    }
    B->instructions() = std::move(rebuilt);
    return true;
}

// ── Souper/Minotaur-style harvesting ──────────────────────────────────────────
//
// harvest_slice lifts a single integer instruction (the "seed") out of an
// arbitrary function into a standalone mini-function, walking back through
// operands up to `max_depth` instructions deep. Every operand that is NOT a
// single-use sliceable integer op in the same block (i.e. memory ops, calls,
// phis, terminators, cross-block references, multi-use instructions,
// depth-exceeded instructions, function arguments, constants) is replaced by
// an opaque `argN` parameter — exactly Souper's `makeArrayRead` model. The
// resulting mini-function is single-block, integer-only, ends in
// `ret <intN>`, and is directly mineable by mine_function().
//
// The internal struct carries the public HarvestedSlice plus the live-in
// `ir::Value` shared_ptrs (parallel to opaque_var_names) and the ordered
// slice instructions, which harvest_and_rewrite needs to splice a proven
// rewrite back via the existing splice_inplace helper (which takes a Window).

struct HarvestedSliceInternal {
    HarvestedSlice public_part;
    std::vector<std::shared_ptr<ir::Value>> live_ins;       // parallel to opaque_var_names
    std::vector<std::shared_ptr<ir::Instruction>> slice_instrs;  // block order, seed last
    std::string root_name;                                   // seed's SSA name
};

std::optional<HarvestedSliceInternal> harvest_slice_impl(
    const ir::Function& fn,
    const std::string& block_name,
    size_t inst_index,
    size_t max_depth,
    bool harvest_across_memory,
    const std::unordered_map<std::string, size_t>* pos_hint = nullptr) {

    auto block = fn.block(block_name);
    if (!block || inst_index >= block->size()) return std::nullopt;
    auto seed = block->instruction(inst_index);
    if (!seed || !seed->has_name() || !seed->type()) return std::nullopt;
    // Only integer-typed seeds with bw >= 2 are harvestable (mine_function's
    // contract: i1-returning sources are out of scope).
    if (seed->type()->type_id() != ir::TypeID::Integer) return std::nullopt;
    if (seed->type()->bit_width() < 2) return std::nullopt;
    if (!is_sliceable_op(*seed)) return std::nullopt;
    if (max_depth == 0) return std::nullopt;  // need at least the seed

    // Position-in-block lookup for the seed's block. Callers that harvest
    // many seeds from the same block pass a precomputed map (`pos_hint`) so
    // the per-seed cost stays O(depth), not O(block).
    std::unordered_map<std::string, size_t> pos_local;
    if (!pos_hint) {
        for (size_t i = 0; i < block->size(); ++i) {
            auto in = block->instruction(i);
            if (in && in->has_name()) pos_local[in->name()] = i;
        }
    }
    const auto& pos = pos_hint ? *pos_hint : pos_local;
    auto pit_seed = pos.find(seed->name());
    const size_t seed_pos = (pit_seed != pos.end()) ? pit_seed->second : inst_index;

    // Backward DFS to find slice instructions. The seed is always in; each
    // operand is folded in iff it is a sliceable integer op in the same
    // block before the seed AND the slice size is still under `max_depth`.
    // Everything else becomes an opaque var.
    //
    // Multi-use operands ARE followed (Souper's model): including a
    // multi-use instruction in the slice gives the SMT query its real
    // definition instead of an opaque var, which is what makes rewrites of
    // expression DAGs (ubiquitous in -O3 output, where CSE reuses values)
    // provable. Splicing stays sound because splice_inplace only drops
    // window instructions with no remaining users outside the window.
    std::unordered_set<std::string> in_slice{seed->name()};
    std::vector<std::shared_ptr<ir::Instruction>> stack{seed};
    while (!stack.empty()) {
        auto cur = stack.back();
        stack.pop_back();
        for (auto& op : cur->operands()) {
            if (!op) continue;
            if (dynamic_cast<ir::ConstantInt*>(op.get())) continue;  // inline const
            auto oi = std::dynamic_pointer_cast<ir::Instruction>(op);
            if (!oi || !oi->has_name()) continue;  // non-instruction → opaque var
            if (in_slice.count(oi->name())) continue;
            auto pit = pos.find(oi->name());
            if (pit == pos.end() || pit->second >= seed_pos) continue;  // cross-block / after seed
            if (!is_sliceable_op(*oi)) continue;  // unsupported op → opaque var
            if (in_slice.size() >= max_depth) continue;  // depth bound → opaque var
            in_slice.insert(oi->name());
            stack.push_back(oi);
        }
    }

    // Collect slice instructions in block order (seed is last by construction).
    std::vector<std::shared_ptr<ir::Instruction>> slice_instrs;
    for (size_t i = 0; i < block->size(); ++i) {
        auto in = block->instruction(i);
        if (in && in->has_name() && in_slice.count(in->name()))
            slice_instrs.push_back(in);
    }
    if (slice_instrs.empty()) return std::nullopt;

    // Collect live-ins (operands of slice instructions that are not in the
    // slice and not constants). Dedup by name. Each becomes an opaque arg.
    std::vector<std::shared_ptr<ir::Value>> live_ins;
    std::unordered_set<std::string> li_seen;
    std::vector<std::string> opaque_var_names;
    std::vector<std::pair<std::string, std::string>> var_origins;
    for (auto& in : slice_instrs) {
        for (auto& op : in->operands()) {
            if (!op) continue;
            if (dynamic_cast<ir::ConstantInt*>(op.get())) continue;  // inline const
            auto oi = std::dynamic_pointer_cast<ir::Instruction>(op);
            if (oi && in_slice.count(oi->name())) continue;  // internal edge
            // This operand becomes an opaque var.
            const std::string name = op->has_name() ? op->name() : std::string();
            if (!li_seen.insert(name).second) continue;  // already a live-in
            // If harvest_across_memory is false, bail on memory ops (matching
            // the old extract_window conservative behaviour).
            if (!harvest_across_memory && oi && oi->is_memory_op())
                return std::nullopt;
            live_ins.push_back(op);
            const std::string var_name = "arg" + std::to_string(opaque_var_names.size());
            opaque_var_names.push_back(var_name);
            var_origins.emplace_back(var_name, name);
        }
    }

    // Build the mini-function. Args are the opaque vars (typed by their
    // original values); the body is the cloned slice instructions (internal
    // operand edges preserved, live-in edges rewired to args, constants
    // inline); the terminator is `ret <seed_clone>`.
    std::unordered_map<const ir::Value*, std::shared_ptr<ir::Value>> li_to_arg;
    std::vector<std::shared_ptr<ir::Type>> arg_types;
    for (size_t i = 0; i < live_ins.size(); ++i) {
        auto av = std::make_shared<ir::Value>(live_ins[i]->type(),
                                              "arg" + std::to_string(i));
        li_to_arg[live_ins[i].get()] = av;
        arg_types.push_back(live_ins[i]->type());
    }
    auto root_ty = slice_instrs.back()->type();
    auto fnty = std::make_shared<ir::FunctionType>(root_ty, arg_types);
    auto mini = std::make_shared<ir::Function>("slice", fnty, ir::Linkage::Internal);
    for (size_t i = 0; i < live_ins.size(); ++i)
        mini->add_argument(arg_types[i], "arg" + std::to_string(i));
    auto& bb = mini->add_block("entry");

    std::unordered_map<const ir::Value*, std::shared_ptr<ir::Value>> clone_of;
    for (auto& oi : slice_instrs) {
        auto ci = clone_remap(*oi, oi->name(),
            [&](const std::shared_ptr<ir::Value>& op) -> std::shared_ptr<ir::Value> {
                auto c = clone_of.find(op.get());
                if (c != clone_of.end()) return c->second;  // internal edge
                auto a = li_to_arg.find(op.get());
                if (a != li_to_arg.end()) return a->second;  // live-in → arg
                return op;  // constant (inline)
            });
        clone_of[oi.get()] = ci;
        bb.add_instruction(ci);
    }
    bb.add_instruction(ir::inst::make_ret(clone_of[slice_instrs.back().get()]));

    HarvestedSliceInternal result;
    result.public_part.mini_function = mini;
    result.public_part.source_block = block_name;
    result.public_part.source_inst_index = inst_index;
    result.public_part.opaque_var_names = std::move(opaque_var_names);
    result.public_part.var_origins = std::move(var_origins);
    result.live_ins = std::move(live_ins);
    result.slice_instrs = std::move(slice_instrs);
    result.root_name = seed->name();
    return result;
}

// ── Dominating branch conditions (PC-aware mining) ────────────────────────────────
//
// One dominating condition for a block: every execution reaching the block
// necessarily came through this conditional edge, so the condition may be
// CONJOINED into a slice's equivalence query. Collected by walking the
// block's single-predecessor chain upward; a join point (>1 predecessor)
// stops the walk because its incoming conditions are disjunctive, and a
// cycle guard stops it on loop back-edges. This is Souper's sound
// direct-dominator PC model, deliberately without full path enumeration.

struct DomCondition {
    ir::CmpPredicate pred = ir::CmpPredicate::EQ;
    std::shared_ptr<ir::Value> lhs;   // icmp lhs, or the raw i1 condition
    std::shared_ptr<ir::Value> rhs;   // icmp rhs (null for a raw i1 condition)
    bool negated = false;             // reached via the false edge
    bool is_raw_i1 = false;           // constraint is lhs == (negated ? 0 : 1)
};

std::vector<DomCondition> dominating_conditions(const ir::Function& fn,
                                                const std::string& block_name) {
    std::vector<DomCondition> out;
    const_cast<ir::Function&>(fn).compute_predecessors();
    std::unordered_set<std::string> visited{block_name};
    std::string cur = block_name;
    for (;;) {
        auto bb = fn.block(cur);
        if (!bb) break;
        auto preds = bb->predecessors();
        if (preds.size() != 1) break;                  // join point or entry
        const std::string pred_name = preds.front();
        if (!visited.insert(pred_name).second) break;  // cycle guard
        auto pbb = fn.block(pred_name);
        if (!pbb) break;
        auto term = pbb->terminator();
        if (term && term->opcode() == ir::Opcode::Br && term->num_operands() >= 1) {
            const auto& md = term->metadata();
            auto t = md.find("true_bb");
            auto f = md.find("false_bb");
            if (t != md.end() && f != md.end() && t->second != f->second &&
                (t->second == cur || f->second == cur)) {
                DomCondition dc;
                dc.negated = (f->second == cur);
                auto cond = term->operand(0);
                auto ci = std::dynamic_pointer_cast<ir::Instruction>(cond);
                if (ci && ci->opcode() == ir::Opcode::ICmp &&
                    ci->num_operands() >= 2) {
                    dc.pred = ir::CmpPredicate::EQ;
                    auto pit = ci->metadata().find("pred");
                    if (pit != ci->metadata().end()) {
                        try {
                            dc.pred = static_cast<ir::CmpPredicate>(
                                std::stoul(pit->second));
                        } catch (...) { dc.pred = ir::CmpPredicate::EQ; }
                    }
                    dc.lhs = ci->operand(0);
                    dc.rhs = ci->operand(1);
                    if (dc.lhs && dc.rhs) out.push_back(dc);
                } else if (cond) {
                    dc.is_raw_i1 = true;
                    dc.lhs = cond;
                    out.push_back(dc);
                }
            }
        }
        cur = pred_name;
    }
    return out;
}

// Map dominating conditions onto a harvested slice's argument positions.
// A condition is expressible iff each non-constant operand is one of the
// slice's live-in ORIGINS (var_origins: argK ↔ original value name — SSA
// name equality is value identity, so the constraint transfers exactly).
// Inexpressible conditions are dropped, which only widens the proof's
// input space (sound).
std::vector<ArgAssumption> map_conditions_to_args(
    const std::vector<DomCondition>& dcs,
    const std::vector<std::pair<std::string, std::string>>& var_origins) {
    std::unordered_map<std::string, int> origin_to_arg;
    for (size_t i = 0; i < var_origins.size(); ++i)
        if (!var_origins[i].second.empty())
            origin_to_arg[var_origins[i].second] = static_cast<int>(i);

    std::vector<ArgAssumption> out;
    for (const auto& dc : dcs) {
        ArgAssumption a;
        auto map_side = [&](const std::shared_ptr<ir::Value>& v,
                            int& argi, int64_t& cval) -> bool {
            if (!v) return false;
            if (auto ci = dynamic_cast<ir::ConstantInt*>(v.get())) {
                cval = ci->value();
                return true;
            }
            if (v->has_name()) {
                auto it = origin_to_arg.find(v->name());
                if (it != origin_to_arg.end()) { argi = it->second; return true; }
            }
            return false;
        };
        if (dc.is_raw_i1) {
            // Plain i1 branch condition: cond == 1 (true edge) / 0 (false).
            a.predicate = ir::CmpPredicate::EQ;
            if (!map_side(dc.lhs, a.lhs_arg, a.lhs_const) || a.lhs_arg < 0)
                continue;
            a.rhs_const = dc.negated ? 0 : 1;
            out.push_back(a);
        } else {
            a.predicate = dc.pred;
            a.negated = dc.negated;
            if (!map_side(dc.lhs, a.lhs_arg, a.lhs_const)) continue;
            if (!map_side(dc.rhs, a.rhs_arg, a.rhs_const)) continue;
            if (a.lhs_arg < 0 && a.rhs_arg < 0) continue;  // const vs const
            out.push_back(a);
        }
    }
    return out;
}

// ── Select-arm conditional harvesting (select form) ───────────────────────────────
//
// -O3 turns most branches into selects, so the dominating-branch PC walk
// rarely fires on real optimised IR. The same conditional-equivalence idea
// applies inside a block, though: for `%s = select %c, %a, %b`, the value
// %a is observable ONLY when %c is true — so if %a's defining instruction
// has no user other than the select, it may be replaced by anything
// equivalent UNDER %c (and %b under NOT %c). The replacement still
// executes unconditionally, which is fine: the enumerator's op set is
// trap-free and flag-free, so the off-condition value is dead and harmless.
//
// The job carries the arm slice (as a mini-function whose args cover the
// arm's AND the condition's live-ins — one shared arg space) plus an
// i1-returning condition mini-function over those same args, which
// mine_function conjoins into the SMT query via verify_with_assumptions.

struct SelectArmJob {
    HarvestedSliceInternal h;                 // arm slice, combined args
    std::shared_ptr<ir::Function> cond_mini;  // i1 mini-fn over the same args
    bool cond_negated = false;                // false arm → NOT(cond)
};

std::optional<SelectArmJob> harvest_select_arm(
    const ir::Function& fn,
    const std::shared_ptr<ir::BasicBlock>& block,
    const std::unordered_map<std::string, size_t>& pos,
    const std::unordered_map<std::string, size_t>& user_count,
    const std::shared_ptr<ir::Instruction>& sel,
    bool true_arm,
    size_t max_depth,
    size_t cond_depth,
    bool harvest_across_memory) {
    if (!sel || sel->opcode() != ir::Opcode::Select || sel->num_operands() < 3)
        return std::nullopt;
    auto cond = sel->operand(0);
    auto arm = std::dynamic_pointer_cast<ir::Instruction>(
        sel->operand(true_arm ? 1 : 2));
    if (!cond || !arm || !arm->has_name()) return std::nullopt;

    // The arm's value must be observable ONLY through this select.
    auto uit = user_count.find(arm->name());
    if (uit == user_count.end() || uit->second != 1) return std::nullopt;

    auto ait = pos.find(arm->name());
    if (ait == pos.end()) return std::nullopt;  // arm not in this block

    // Harvest the arm slice (single-instruction slices allowed here: under
    // a condition even one op can fold to an identity or constant).
    auto h = harvest_slice_impl(fn, block->name(), ait->second, max_depth,
                                harvest_across_memory, &pos);
    if (!h) return std::nullopt;

    // ── Build the condition chain over the SAME argument space ─────────
    // Backward closure from the condition value: same-block, before-the-
    // condition's-own-position, sliceable, depth-bounded. The chain is only
    // a specification for the SMT query — it is never spliced — so
    // multi-use members are fine.
    std::unordered_map<std::string, int> origin_to_arg;
    for (size_t i = 0; i < h->public_part.var_origins.size(); ++i)
        if (!h->public_part.var_origins[i].second.empty())
            origin_to_arg[h->public_part.var_origins[i].second] =
                static_cast<int>(i);

    std::vector<std::shared_ptr<ir::Instruction>> cond_chain;  // block order
    std::unordered_set<std::string> in_chain;
    auto cond_inst = std::dynamic_pointer_cast<ir::Instruction>(cond);
    if (cond_inst && cond_inst->has_name() &&
        !origin_to_arg.count(cond_inst->name())) {
        auto cit = pos.find(cond_inst->name());
        if (cit != pos.end() && is_sliceable_op(*cond_inst)) {
            in_chain.insert(cond_inst->name());
            std::vector<std::shared_ptr<ir::Instruction>> stack{cond_inst};
            while (!stack.empty()) {
                auto cur = stack.back();
                stack.pop_back();
                for (auto& op : cur->operands()) {
                    if (!op) continue;
                    if (dynamic_cast<ir::ConstantInt*>(op.get())) continue;
                    auto oi = std::dynamic_pointer_cast<ir::Instruction>(op);
                    if (!oi || !oi->has_name()) continue;
                    if (in_chain.count(oi->name())) continue;
                    // CRUCIAL: a value that is already a live-in of the arm
                    // slice must stay a live-in (argK) here too — that
                    // shared argument is what links the condition to the
                    // arm in the SMT query. Recomputing it inside the
                    // condition chain would leave the arm's argK
                    // unconstrained and the proof would never close.
                    if (origin_to_arg.count(oi->name())) continue;
                    auto pit = pos.find(oi->name());
                    if (pit == pos.end()) continue;      // cross-block → live-in
                    if (!is_sliceable_op(*oi)) continue;  // unsupported → live-in
                    // NOTE: bounded by cond_depth, NOT the arm slice depth —
                    // the chain is a spec for the SMT query, never spliced,
                    // so a shallow arm job must still see the full condition
                    // (a truncated chain constrains an opaque var instead of
                    // the arm's inputs and the proof never closes).
                    if (in_chain.size() >= cond_depth) continue;
                    in_chain.insert(oi->name());
                    stack.push_back(oi);
                }
            }
            for (size_t i = 0; i < block->size(); ++i) {
                auto in = block->instruction(i);
                if (in && in->has_name() && in_chain.count(in->name()))
                    cond_chain.push_back(in);
            }
        }
    }

    // Live-ins of the condition (chain operands outside the chain, or the
    // condition itself when it has no includable chain). New ones extend
    // the combined argument space; the arm mini-function gains them as
    // (unused) trailing arguments so both minis share one signature.
    const size_t arm_own_args = h->live_ins.size();
    auto intern_live_in = [&](const std::shared_ptr<ir::Value>& v) -> int {
        if (!v || !v->has_name()) return -1;
        auto it = origin_to_arg.find(v->name());
        if (it != origin_to_arg.end()) return it->second;
        const int idx = static_cast<int>(h->live_ins.size());
        const std::string arg_name = "arg" + std::to_string(idx);
        h->live_ins.push_back(v);
        h->public_part.opaque_var_names.push_back(arg_name);
        h->public_part.var_origins.emplace_back(arg_name, v->name());
        origin_to_arg[v->name()] = idx;
        return idx;
    };

    // Map from original cond-chain values to their clones in the cond mini.
    // Live-in edges become argK references (fresh Values named argK).
    std::vector<std::pair<std::shared_ptr<ir::Value>, int>> chain_live_ins;
    if (cond_chain.empty()) {
        // The condition itself is opaque (an argument, cross-block value,
        // or unsupported op): the cond mini is just `ret i1 argK`.
        const int idx = intern_live_in(cond);
        if (idx < 0) return std::nullopt;
    } else {
        for (auto& in : cond_chain) {
            for (auto& op : in->operands()) {
                if (!op) continue;
                if (dynamic_cast<ir::ConstantInt*>(op.get())) continue;
                auto oi = std::dynamic_pointer_cast<ir::Instruction>(op);
                if (oi && in_chain.count(oi->name())) continue;  // internal
                const int idx = intern_live_in(op);
                if (idx < 0) return std::nullopt;  // unnamed non-constant
                chain_live_ins.emplace_back(op, idx);
            }
        }
    }

    // If the condition introduced new live-ins, rebuild the arm mini over
    // the combined argument list (the two minis must share one signature —
    // verify_with_z3 builds the shared Z3 argument constants from the arm
    // mini's arguments). The extra args are unused by the arm body, which
    // is harmless: equivalence over more inputs is strictly stronger.
    std::vector<std::shared_ptr<ir::Type>> arg_types;
    for (auto& li : h->live_ins) arg_types.push_back(li->type());
    if (h->live_ins.size() != arm_own_args) {
        std::unordered_map<const ir::Value*, std::shared_ptr<ir::Value>> li_to_arg;
        for (size_t i = 0; i < h->live_ins.size(); ++i)
            li_to_arg[h->live_ins[i].get()] = std::make_shared<ir::Value>(
                h->live_ins[i]->type(), "arg" + std::to_string(i));
        auto root_ty = h->slice_instrs.back()->type();
        auto fnty = std::make_shared<ir::FunctionType>(root_ty, arg_types);
        auto mini = std::make_shared<ir::Function>("slice", fnty,
                                                   ir::Linkage::Internal);
        for (size_t i = 0; i < h->live_ins.size(); ++i)
            mini->add_argument(arg_types[i], "arg" + std::to_string(i));
        auto& mbb = mini->add_block("entry");
        std::unordered_map<const ir::Value*, std::shared_ptr<ir::Value>> clone_of;
        for (auto& oi : h->slice_instrs) {
            auto ci = clone_remap(*oi, oi->name(),
                [&](const std::shared_ptr<ir::Value>& op) -> std::shared_ptr<ir::Value> {
                    auto c = clone_of.find(op.get());
                    if (c != clone_of.end()) return c->second;   // internal
                    auto a = li_to_arg.find(op.get());
                    if (a != li_to_arg.end()) return a->second;  // live-in → arg
                    return op;  // constant
                });
            clone_of[oi.get()] = ci;
            mbb.add_instruction(ci);
        }
        mbb.add_instruction(
            ir::inst::make_ret(clone_of[h->slice_instrs.back().get()]));
        h->public_part.mini_function = mini;
    }
    auto cond_fnty = std::make_shared<ir::FunctionType>(cond->type(), arg_types);
    auto cond_mini = std::make_shared<ir::Function>("selcond", cond_fnty,
                                                    ir::Linkage::Internal);
    std::vector<std::shared_ptr<ir::Value>> arg_vals;
    for (size_t i = 0; i < h->live_ins.size(); ++i) {
        cond_mini->add_argument(arg_types[i], "arg" + std::to_string(i));
        arg_vals.push_back(std::make_shared<ir::Value>(
            arg_types[i], "arg" + std::to_string(i)));
    }
    auto& cbb = cond_mini->add_block("entry");
    if (cond_chain.empty()) {
        auto it = origin_to_arg.find(cond->name());
        if (it == origin_to_arg.end()) return std::nullopt;
        cbb.add_instruction(ir::inst::make_ret(arg_vals[it->second]));
    } else {
        std::unordered_map<const ir::Value*, std::shared_ptr<ir::Value>> clone_of;
        for (auto& oi : cond_chain) {
            auto ci = clone_remap(*oi, oi->name(),
                [&](const std::shared_ptr<ir::Value>& op) -> std::shared_ptr<ir::Value> {
                    auto c = clone_of.find(op.get());
                    if (c != clone_of.end()) return c->second;   // internal
                    if (op && op->has_name()) {
                        auto a = origin_to_arg.find(op->name());
                        if (a != origin_to_arg.end()) return arg_vals[a->second];
                    }
                    return op;  // constant
                });
            clone_of[oi.get()] = ci;
            cbb.add_instruction(ci);
        }
        cbb.add_instruction(ir::inst::make_ret(clone_of[cond_chain.back().get()]));
    }

    SelectArmJob job;
    job.h = std::move(*h);
    job.cond_mini = cond_mini;
    job.cond_negated = !true_arm;
    return job;
}

}  // namespace

PeepholeMiner::PeepholeMiner(evaluator::EvaluationEngine* engine,
                             const MinerConfig& config)
    : engine_(engine), config_(config) {}

std::optional<MinedPattern> PeepholeMiner::mine_function(
    const ir::Function& fn,
    const MineAssumptions* assumptions) {
    stats_.functions_seen++;
    if (!engine_) return std::nullopt;
    if (assumptions && assumptions->empty()) assumptions = nullptr;

    auto elig = analyze(fn);
    if (!elig) return std::nullopt;
    if (elig->live_ins.empty()) return std::nullopt;
    stats_.eligible++;

    const double src_score = engine_->analyse(fn).score;
    const uint64_t src_hash = StochasticSearch::structural_hash(fn);

    // Operand pool = MAIN-WIDTH argument leaves + constant pool. Off-width
    // live-ins (introduced by casts in the source) can't be bw-typed leaves;
    // they participate via the cast seeds below.
    const unsigned bw = elig->bw;
    std::vector<std::shared_ptr<ir::Value>> base_pool;
    for (size_t i = 0; i < elig->live_ins.size(); ++i)
        if (elig->live_in_widths.size() <= i || elig->live_in_widths[i] == bw)
            base_pool.push_back(elig->live_ins[i]);
    for (auto& c : constant_pool(*elig, config_)) base_pool.push_back(c);

    // ── Cast seeds: single casts of off-width live-ins to the main width ─
    // Each seed is one instruction (`zext/sext/trunc argK to bw`) prepended
    // to a separate enumeration run, with its result joining the pool. This
    // reaches candidates like `ret (zext arg0)` — the redundant-mask-around-
    // zext shape (-O3 IR is full of casts) — at exactly one instruction of
    // length budget. Connectivity pruning discards the duplicates where the
    // seed goes unused.
    struct CastSeed {
        std::shared_ptr<ir::Instruction> inst;
    };
    std::vector<CastSeed> cast_seeds;
    {
        auto tyW = std::make_shared<ir::IntegerType>(bw);
        size_t ci = 0;
        for (size_t i = 0; i < elig->live_ins.size() &&
                           i < elig->live_in_widths.size(); ++i) {
            const unsigned w = elig->live_in_widths[i];
            if (w == bw || w == 0) continue;
            auto mk = [&](ir::Opcode op) {
                auto inst = std::make_shared<ir::Instruction>(
                    op, tyW, "c" + std::to_string(ci++));
                inst->add_operand(elig->live_ins[i]);
                cast_seeds.push_back({inst});
            };
            if (w < bw) { mk(ir::Opcode::ZExt); mk(ir::Opcode::SExt); }
            else        { mk(ir::Opcode::Trunc); }
        }
    }

    // Canonicalise a value to a PER-ARGUMENT bit-width (low bits, sign-
    // extended) so it is a well-formed value of that argument's type. With
    // casts in slices the arguments no longer share one width.
    std::vector<unsigned> arg_widths;
    for (auto& a : fn.arguments())
        arg_widths.push_back(a.type ? static_cast<unsigned>(a.type->bit_width())
                                    : bw);
    auto canon_w = [](int64_t v, unsigned w) -> int64_t {
        if (w == 0 || w >= 64) return v;
        const uint64_t m = (uint64_t(1) << w) - 1;
        uint64_t r = static_cast<uint64_t>(v) & m;
        if (r & (uint64_t(1) << (w - 1))) r |= ~m;  // sign-extend
        return static_cast<int64_t>(r);
    };
    auto canon_vec = [&](std::vector<int64_t>& v) {
        for (size_t i = 0; i < v.size(); ++i)
            v[i] = canon_w(v[i], i < arg_widths.size() ? arg_widths[i] : bw);
    };

    // Concrete test vectors for the pre-filter. Mix small/edge values with
    // random ones so unrelated candidates disagree quickly. Deterministic
    // seed for reproducibility.
    const size_t n_args = fn.argument_count();

    // Evaluate one assumption conjunct on a concrete vector of canonical
    // (sign-extended) argument values. Mirrors the SMT encoding: signed
    // predicates compare the sign-extended reps directly; unsigned ones
    // compare the width-masked reps. Assumptions referencing out-of-range
    // args are treated as vacuously true (the SMT side drops them too).
    auto assumptions_hold = [&](const std::vector<int64_t>& v) -> bool {
        if (!assumptions) return true;
        // Arbitrary i1 condition (select-arm mining): interpret it on the
        // same vector and require the requested truth value.
        if (assumptions->condition_fn) {
            auto cv = evaluator::Interpreter::interpret(
                *assumptions->condition_fn, v);
            if (!cv) return false;  // uninterpretable — don't trust the vector
            const bool truth = ((*cv) & 1) != 0;
            if (truth == assumptions->condition_negated) return false;
        }
        for (const auto& a : assumptions->arg_assumptions) {
            unsigned w = 64;
            auto side = [&](int argi, int64_t cval) -> int64_t {
                if (argi >= 0 && static_cast<size_t>(argi) < v.size()) {
                    if (static_cast<size_t>(argi) < fn.argument_count() &&
                        fn.arguments()[argi].type)
                        w = std::min<unsigned>(
                            w, static_cast<unsigned>(
                                   fn.arguments()[argi].type->bit_width()));
                    return v[argi];
                }
                return cval;
            };
            const int64_t l = side(a.lhs_arg, a.lhs_const);
            const int64_t r = side(a.rhs_arg, a.rhs_const);
            if (a.lhs_arg < 0 && a.rhs_arg < 0) continue;  // const vs const
            if (w == 0 || w > 64) continue;
            const uint64_t mask =
                (w >= 64) ? ~uint64_t(0) : ((uint64_t(1) << w) - 1);
            const uint64_t lu = static_cast<uint64_t>(l) & mask;
            const uint64_t ru = static_cast<uint64_t>(r) & mask;
            // Sign-extend the masked values for the signed predicates so
            // constants wider than w compare like the SMT numerals do.
            auto sext = [&](uint64_t x) -> int64_t {
                if (w >= 64) return static_cast<int64_t>(x);
                if (x & (uint64_t(1) << (w - 1))) x |= ~mask;
                return static_cast<int64_t>(x);
            };
            const int64_t ls = sext(lu), rs = sext(ru);
            bool ok = true;
            switch (a.predicate) {
                case ir::CmpPredicate::EQ:  ok = lu == ru; break;
                case ir::CmpPredicate::NE:  ok = lu != ru; break;
                case ir::CmpPredicate::UGT: ok = lu >  ru; break;
                case ir::CmpPredicate::UGE: ok = lu >= ru; break;
                case ir::CmpPredicate::ULT: ok = lu <  ru; break;
                case ir::CmpPredicate::ULE: ok = lu <= ru; break;
                case ir::CmpPredicate::SGT: ok = ls >  rs; break;
                case ir::CmpPredicate::SGE: ok = ls >= rs; break;
                case ir::CmpPredicate::SLT: ok = ls <  rs; break;
                case ir::CmpPredicate::SLE: ok = ls <= rs; break;
                default: break;
            }
            if (a.negated) ok = !ok;
            if (!ok) return false;
        }
        return true;
    };

    std::vector<std::vector<int64_t>> tests;
    {
        std::mt19937_64 rng(0x5EED5EED5EEDULL);
        std::uniform_int_distribution<int64_t> dist(
            std::numeric_limits<int64_t>::min(), std::numeric_limits<int64_t>::max());
        static const int64_t seeds[] = {0, 1, 2, 3, -1, -2, 7, 255, 256, 65535,
                                        123456789, -987654321};
        for (int64_t s : seeds) {
            std::vector<int64_t> v(n_args, s);
            canon_vec(v);
            if (assumptions_hold(v)) tests.push_back(std::move(v));
        }
        // Random vectors. Under assumptions this is rejection sampling with
        // a generous attempt budget — a weak pre-filter only costs extra SMT
        // calls (capped by max_smt_checks), never soundness.
        const int want = 24;
        const int max_attempts = assumptions ? 2000 : want;
        int kept = 0;
        for (int i = 0; i < max_attempts && kept < want; ++i) {
            std::vector<int64_t> v(n_args);
            for (auto& x : v) x = dist(rng);
            // Repair pass for simple (arg == const) conjuncts — the common
            // dominating-branch shape random sampling would never hit.
            if (assumptions) {
                for (const auto& a : assumptions->arg_assumptions) {
                    if (a.negated || a.predicate != ir::CmpPredicate::EQ) continue;
                    if (a.lhs_arg >= 0 && a.rhs_arg < 0 &&
                        static_cast<size_t>(a.lhs_arg) < v.size())
                        v[a.lhs_arg] = a.rhs_const;
                    else if (a.rhs_arg >= 0 && a.lhs_arg < 0 &&
                             static_cast<size_t>(a.rhs_arg) < v.size())
                        v[a.rhs_arg] = a.lhs_const;
                }
            }
            canon_vec(v);
            if (!assumptions_hold(v)) continue;
            tests.push_back(std::move(v));
            ++kept;
        }
    }
    std::vector<std::optional<int64_t>> src_out;
    src_out.reserve(tests.size());
    for (auto& v : tests)
        src_out.push_back(evaluator::Interpreter::interpret(fn, v));

    SMTConfig sc;
    sc.timeout_ms = config_.smt_timeout_ms;
    SMTVerifier verifier(sc);

    size_t smt_used = 0;

    // Enumerate shortest-first; return the cheapest proven win at the first
    // length that yields one (minimal-length superoptimisation). Length 0
    // means the result is a bare leaf or constant (`ret %x` / `ret C`) — the
    // true optimum for identities and constant functions, e.g. (x^C)^C -> x.
    // Small operand set for the compare-select synthesiser (built once).
    const std::vector<std::shared_ptr<ir::Value>> cs_pool =
        compare_pool(*elig, /*cap=*/8);

    for (size_t L = 0; L <= config_.max_length; ++L) {
        EnumCtx ctx{fn, elig->bw, engine_, config_, src_score, src_hash,
                    &tests, &src_out, 0, {}};
        if (L == 0) {
            for (auto& leaf : base_pool) {
                if (ctx.candidates >= config_.max_candidates) break;
                consider(ctx, build_function(fn, {}, leaf));
            }
        } else {
            // Compare-select wins are length 2. Run the targeted synthesiser
            // FIRST so its candidates aren't starved by the binop enumeration
            // hitting the candidate budget.
            if (L == 2) synth_compare_select(ctx, cs_pool, cs_pool);
            std::vector<std::shared_ptr<ir::Instruction>> instrs;
            enumerate(ctx, base_pool, instrs, L);
            // Cast-seeded runs: one enumeration per off-width live-in cast,
            // with the cast as the first instruction (one unit of the length
            // budget) and its bw-typed result joining the pool. At L == 1
            // this yields the bare `ret (cast argK)` candidates.
            for (auto& seed : cast_seeds) {
                if (ctx.candidates >= config_.max_candidates) break;
                std::vector<std::shared_ptr<ir::Instruction>> instrs2{seed.inst};
                if (L == 1) {
                    consider(ctx, build_function(fn, instrs2, seed.inst));
                    continue;
                }
                std::vector<std::shared_ptr<ir::Value>> pool2 = base_pool;
                pool2.push_back(seed.inst);
                enumerate(ctx, pool2, instrs2, L);
            }
        }
        stats_.candidates_tried += ctx.candidates;
        if (ctx.cheaper.empty()) continue;

        // Verify the cheapest candidates first.
        std::sort(ctx.cheaper.begin(), ctx.cheaper.end(),
                  [](const auto& a, const auto& b) { return a.second > b.second; });
        for (auto& [cand, score] : ctx.cheaper) {
            if (smt_used >= config_.max_smt_checks) break;
            ++smt_used;
            ++stats_.smt_checks;
            // Under a path/select condition the equivalence only needs to
            // hold on the assumption-constrained input space (the splice
            // target only feeds observable state under those conditions).
            auto res = assumptions
                           ? verifier.verify_with_assumptions(
                                 fn, *cand, assumptions->arg_assumptions,
                                 assumptions->condition_fn.get(),
                                 assumptions->condition_negated)
                           : verifier.verify(fn, *cand);
            if (res.status == VerificationResult::Equivalent) {
                stats_.mined++;
                MinedPattern mp;
                // Own a stable snapshot of the source so the pattern outlives
                // the caller's function.
                mp.source = ir::deep_copy_function(fn);
                mp.replacement = cand;
                mp.source_score = src_score;
                mp.replacement_score = score;
                mp.smt_checks = smt_used;
                return mp;
            }
        }
        if (smt_used >= config_.max_smt_checks) break;
    }

    // ── CEGIS constant-synthesis fallback ────────────────────────────────────
    // If straight-line enumeration found no proven win, try CEGIS: build
    // template candidates with a placeholder constant and ask Z3 to
    // synthesise the constant value. This lets the miner invent constants
    // outside its 16-element pool — e.g. `shl x, 3` -> `mul x, 8` (CEGIS
    // finds the multiplier 8).
    if (config_.enable_cegis_synthesis && SMTVerifier::is_z3_available()) {
        auto cegis_win = mine_with_cegis(fn);
        if (cegis_win) {
            return cegis_win;
        }
    }

    return std::nullopt;
}

// ═══════════════════════════════════════════════════════════════════════════
// CEGIS constant synthesis
// ═══════════════════════════════════════════════════════════════════════════
//
// For each (live-in arg, binop opcode) pair, build a template candidate
// `op arg, ?` (where `?` is a placeholder ConstantInt with a known name),
// then ask SMTVerifier::synthesize_with_cegis to find the constant value
// that makes the template equivalent to `fn`. If CEGIS succeeds and the
// instantiated candidate is cheaper than `fn`, return it as a MinedPattern.
//
// This is the lever that lets the miner invent constants outside its
// 16-element pool — e.g.:
//   - `x * 8`  -> `shl x, 3`  (CEGIS finds the shift amount 3)
//   - `x / 5`  -> `mul x, 0xCCCCCCCD` (magic-number division — CEGIS finds
//                  the multiplier; this requires the miner to emit mul+shift,
//                  which the current template set doesn't cover, but the
//                  single-mul template can still find `x * 7` -> `x * 7`
//                  equivalences and similar).
//
// The template set is intentionally small (8 opcodes × 2 operand orders =
// 16 templates max) to keep the CEGIS budget bounded. Each CEGIS call
// takes ~50-200ms, so the worst case is ~3s for 16 templates — well within
// the per-function time budget.
//
std::optional<MinedPattern> PeepholeMiner::mine_with_cegis(const ir::Function& fn) {
    if (!engine_) return std::nullopt;
    if (!config_.enable_cegis_synthesis) return std::nullopt;
    // CEGIS needs Z3 (it calls synthesize_with_cegis which needs Z3).
    if (!SMTVerifier::is_z3_available()) return std::nullopt;

    auto elig = analyze(fn);
    if (!elig) return std::nullopt;
    if (elig->live_ins.empty()) return std::nullopt;

    const double src_score = engine_->analyse(fn).score;
    const unsigned bw = elig->bw;

    // Template opcodes: the same set the enumerator emits, minus div/rem
    // (too expensive to be a cheaper replacement) and minus shifts with
    // a variable shift amount (the placeholder goes in the constant slot).
    // We try each opcode with the placeholder as the SECOND operand
    // (op arg, ?) and, for non-commutative ops, also as the FIRST
    // (op ?, arg).
    struct TemplateShape {
        ir::Opcode op;
        bool placeholder_first;  // false = `op arg, ?`; true = `op ?, arg`
        const char* name;
    };
    const TemplateShape shapes[] = {
        {ir::Opcode::Add,  false, "add"},
        {ir::Opcode::Sub,  false, "sub"},
        {ir::Opcode::Sub,  true,  "sub_rev"},
        {ir::Opcode::Mul,  false, "mul"},
        {ir::Opcode::And,  false, "and"},
        {ir::Opcode::Or,   false, "or"},
        {ir::Opcode::Xor,  false, "xor"},
        {ir::Opcode::Shl,  false, "shl"},
        {ir::Opcode::LShr, false, "lshr"},
        {ir::Opcode::AShr, false, "ashr"},
    };

    SMTConfig sc;
    sc.timeout_ms = config_.smt_timeout_ms;
    SMTVerifier verifier(sc);

    size_t attempts = 0;
    size_t best_smt_checks = 0;

    for (const auto& shape : shapes) {
        if (attempts >= config_.max_cegis_attempts) break;
        ++attempts;
        ++stats_.cegis_attempts;

        // Build the template candidate: `op arg0, ?` (or `op ?, arg0`).
        // Use the first MAIN-WIDTH live-in as the non-placeholder operand
        // (off-width live-ins from cast slices can't type a bw template).
        std::shared_ptr<ir::Value> live_in;
        for (size_t li = 0; li < elig->live_ins.size(); ++li) {
            if (elig->live_in_widths.size() > li &&
                elig->live_in_widths[li] != bw) continue;
            live_in = elig->live_ins[li];
            break;
        }
        if (!live_in) break;  // no bw-typed live-in — no templates to try
        // Create a placeholder ConstantInt with value 0 and a known name.
        // The name is what synthesize_with_cegis will look up.
        auto int_ty = std::dynamic_pointer_cast<ir::IntegerType>(live_in->type());
        if (!int_ty) continue;
        auto placeholder = std::make_shared<ir::ConstantInt>(int_ty, 0);
        std::string ph_name = "ph_" + std::string(shape.name);
        placeholder->set_name(ph_name);

        // Build the template function.
        auto tmpl = std::make_shared<ir::Function>(
            fn.name() + "_cegis_" + shape.name, fn.function_type(), fn.linkage());
        for (auto& a : fn.arguments()) tmpl->add_argument(a.type, a.name);
        auto& bb = tmpl->add_block("entry");

        // Build the binop instruction.
        auto lhs = shape.placeholder_first ? placeholder : live_in;
        auto rhs = shape.placeholder_first ? live_in : placeholder;
        // Use inst::make_* if available, else build manually.
        std::shared_ptr<ir::Instruction> binop;
        switch (shape.op) {
            case ir::Opcode::Add:  binop = ir::inst::make_add(lhs, rhs, "r"); break;
            case ir::Opcode::Sub:  binop = ir::inst::make_sub(lhs, rhs, "r"); break;
            case ir::Opcode::Mul:  binop = ir::inst::make_mul(lhs, rhs, "r"); break;
            default: {
                // For ops without a make_* helper, build manually.
                binop = std::make_shared<ir::Instruction>(shape.op, int_ty, "r");
                binop->add_operand(lhs);
                binop->add_operand(rhs);
                break;
            }
        }
        bb.add_instruction(binop);
        bb.add_instruction(ir::inst::make_ret(binop));

        // Call CEGIS.
        auto sr = verifier.synthesize_with_cegis(
            fn, *tmpl, {ph_name}, {static_cast<int64_t>(bw)});
        if (!sr.success) continue;

        // CEGIS found a constant — instantiate the candidate.
        if (sr.model.empty()) continue;
        int64_t synth_val = sr.model[0].second;

        // Build the instantiated candidate (with the synthesised constant).
        auto instantiated = std::make_shared<ir::Function>(
            fn.name() + "_cegis_" + shape.name + "_inst", fn.function_type(), fn.linkage());
        for (auto& a : fn.arguments()) instantiated->add_argument(a.type, a.name);
        auto& bb2 = instantiated->add_block("entry");
        auto synth_const = std::make_shared<ir::ConstantInt>(int_ty, synth_val);
        auto lhs2 = shape.placeholder_first ? synth_const : live_in;
        auto rhs2 = shape.placeholder_first ? live_in : synth_const;
        std::shared_ptr<ir::Instruction> binop2;
        switch (shape.op) {
            case ir::Opcode::Add:  binop2 = ir::inst::make_add(lhs2, rhs2, "r"); break;
            case ir::Opcode::Sub:  binop2 = ir::inst::make_sub(lhs2, rhs2, "r"); break;
            case ir::Opcode::Mul:  binop2 = ir::inst::make_mul(lhs2, rhs2, "r"); break;
            default: {
                binop2 = std::make_shared<ir::Instruction>(shape.op, int_ty, "r");
                binop2->add_operand(lhs2);
                binop2->add_operand(rhs2);
                break;
            }
        }
        bb2.add_instruction(binop2);
        bb2.add_instruction(ir::inst::make_ret(binop2));

        // Score the instantiated candidate.
        double cand_score = engine_->analyse(*instantiated).score;
        // Convention: higher score = better (less negative).
        if (cand_score <= src_score) continue;

        // The candidate is cheaper AND CEGIS-verified. Return it.
        ++stats_.cegis_wins;
        MinedPattern mp;
        mp.source = ir::deep_copy_function(fn);
        mp.replacement = instantiated;
        mp.source_score = src_score;
        mp.replacement_score = cand_score;
        mp.smt_checks = attempts;  // approximate — each CEGIS iteration does ~2 SMT calls
        best_smt_checks = attempts;
        return mp;
    }

    (void)best_smt_checks;
    return std::nullopt;
}

std::vector<MinedPattern> PeepholeMiner::mine_module(const ir::Module& module) {
    std::vector<MinedPattern> wins;
    for (auto& fn : module.functions()) {
        if (!fn) continue;
        auto w = mine_function(*fn);
        if (w) wins.push_back(std::move(*w));
    }
    return wins;
}

std::optional<std::shared_ptr<ir::Function>> PeepholeMiner::mine_and_rewrite(
    const ir::Function& fn, bool* whole_fn_proven) {
    if (whole_fn_proven) *whole_fn_proven = false;
    if (!engine_) return std::nullopt;

    // Function-wide use counts (for the single-use slice constraint).
    std::unordered_map<std::string, size_t> use_count;
    for (auto& block : fn.blocks()) {
        if (!block) continue;
        for (auto& inst : block->instructions()) {
            if (!inst) continue;
            for (auto& op : inst->operands())
                if (op && op->has_name()) use_count[op->name()]++;
        }
    }

    // Extract one maximal slice per eligible root, across every block.
    std::vector<Window> windows;
    for (auto& block : fn.blocks()) {
        if (!block) continue;
        std::unordered_map<std::string, size_t> pos;
        for (size_t i = 0; i < block->size(); ++i) {
            auto in = block->instruction(i);
            if (in && in->has_name()) pos[in->name()] = i;
        }
        for (size_t i = 0; i < block->size(); ++i) {
            auto root = block->instruction(i);
            if (!root) continue;
            auto w = extract_window(*block, root, use_count, pos,
                                    config_.max_window_instructions);
            if (w && w->instrs.size() >= config_.min_window_instructions)
                windows.push_back(std::move(*w));
        }
    }
    stats_.windows_seen += windows.size();
    if (windows.empty()) return std::nullopt;

    // Largest slices first — biggest collapse potential, fewest of them.
    std::sort(windows.begin(), windows.end(),
              [](const Window& a, const Window& b) {
                  return a.instrs.size() > b.instrs.size();
              });

    // Greedily mine non-overlapping slices, splicing wins into one copy.
    const auto mine_start = std::chrono::steady_clock::now();
    auto out = ir::deep_copy_function(fn);
    std::unordered_set<std::string> consumed;
    size_t budget = 0, applied = 0, fresh_counter = 0;
    for (auto& w : windows) {
        if (budget >= config_.max_windows_per_function) break;
        if (config_.time_budget_seconds > 0.0 &&
            std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                          mine_start).count() >
                config_.time_budget_seconds)
            break;
        bool overlap = false;
        for (auto& n : w.names)
            if (consumed.count(n)) { overlap = true; break; }
        if (overlap) continue;
        ++budget;

        auto mini = build_mini_function(w);
        auto mp = mine_function(*mini);
        if (!mp || !mp->replacement) continue;
        stats_.windows_mined++;

        if (!splice_inplace(*out, w, *mp->replacement, fresh_counter)) continue;
        for (auto& n : w.names) consumed.insert(n);
        ++applied;
        stats_.rewrites_applied++;
    }
    if (applied == 0) return std::nullopt;

    // Gate 1: the splice must produce well-formed SSA (no dangling refs).
    if (!ir::validate_function(*out)) {
        stats_.reverify_rejections++;
        return std::nullopt;
    }
    // Gate 2 (default): re-prove the WHOLE rewrite equivalent to the original.
    // Each slice was SMT-proved in isolation; this catches any splice bug and
    // refuses functions the verifier can't model as a whole (memory/loops/too
    // large) unless the caller opts into trusting the by-construction argument.
    if (config_.require_smt_reverify) {
        SMTConfig sc;
        sc.timeout_ms = config_.smt_timeout_ms;
        SMTVerifier verifier(sc);
        auto res = verifier.verify(fn, *out);
        if (res.status == VerificationResult::Equivalent) {
            if (whole_fn_proven) *whole_fn_proven = true;
            return out;
        }
        if (res.status == VerificationResult::Unknown &&
            config_.trust_unverified_slices)
            return out;
        stats_.reverify_rejections++;
        return std::nullopt;
    }
    return out;
}

// ── Souper/Minotaur-style harvesting ──────────────────────────────────────────

std::optional<HarvestedSlice> PeepholeMiner::harvest_slice(
    const ir::Function& fn,
    const std::string& block_name,
    size_t inst_index,
    size_t max_depth) const {
    auto result = harvest_slice_impl(fn, block_name, inst_index, max_depth,
                                     config_.harvest_across_memory);
    if (!result) return std::nullopt;
    return std::move(result->public_part);
}

std::optional<std::shared_ptr<ir::Function>> PeepholeMiner::harvest_and_rewrite(
    const ir::Function& fn, bool* whole_fn_proven) {
    return harvest_and_rewrite_impl(fn, /*use_pcs=*/false, whole_fn_proven);
}

std::optional<std::shared_ptr<ir::Function>> PeepholeMiner::harvest_and_rewrite_impl(
    const ir::Function& fn, bool use_pcs, bool* whole_fn_proven) {
    if (whole_fn_proven) *whole_fn_proven = false;
    if (!engine_) return std::nullopt;

    const auto deadline_start = std::chrono::steady_clock::now();
    auto out_of_time = [&]() -> bool {
        if (config_.time_budget_seconds <= 0.0) return false;
        return std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                             deadline_start).count() >
               config_.time_budget_seconds;
    };

    // Harvest one slice per eligible integer instruction, across every block.
    // (Souper harvests every integer instruction with >=1 use; we harvest
    // every integer-typed sliceable op — the use-count filter is implicit:
    // a zero-use instruction's slice would have no observable effect when
    // spliced, and the miner wouldn't find a cheaper equivalent anyway.)
    std::vector<HarvestedSliceInternal> slices;
    for (auto& block : fn.blocks()) {
        if (!block) continue;
        // Precompute the name→position map once per block: harvesting every
        // seed would otherwise be quadratic in block size, which matters now
        // that oversized functions get the miner-only pipeline path.
        std::unordered_map<std::string, size_t> pos;
        for (size_t i = 0; i < block->size(); ++i) {
            auto in = block->instruction(i);
            if (in && in->has_name()) pos[in->name()] = i;
        }
        for (size_t i = 0; i < block->size(); ++i) {
            auto inst = block->instruction(i);
            if (!inst) continue;
            auto h = harvest_slice_impl(fn, block->name(), i,
                                        config_.harvest_max_depth,
                                        config_.harvest_across_memory, &pos);
            if (!h) continue;
            // Skip trivial single-instruction slices — the miner can't beat a
            // single op (it would need length 0 = a bare leaf/constant, which
            // is only a win for identities like `and x, -1`).
            if (h->slice_instrs.size() < config_.min_window_instructions) continue;
            slices.push_back(std::move(*h));
            if (slices.size() >= config_.max_windows_per_function) goto harvest_done;
        }
    }
harvest_done:
    stats_.slices_harvested += slices.size();
    if (slices.empty()) return std::nullopt;

    // Largest slices first — biggest collapse potential, fewest of them.
    std::sort(slices.begin(), slices.end(),
              [](const HarvestedSliceInternal& a, const HarvestedSliceInternal& b) {
                  return a.slice_instrs.size() > b.slice_instrs.size();
              });

    // Dominating branch conditions per block, computed once per distinct
    // block (the walk is per-block; the mapping onto args is per-slice).
    std::unordered_map<std::string, std::vector<DomCondition>> dom_conds;

    // Greedily mine non-overlapping slices, splicing wins into one copy.
    auto out = ir::deep_copy_function(fn);
    double out_score = engine_->analyse(*out).score;
    std::unordered_set<std::string> consumed;
    size_t applied = 0, fresh_counter = 0;

    auto overlaps_consumed = [&](const HarvestedSliceInternal& h) {
        for (auto& in : h.slice_instrs)
            if (consumed.count(in->name())) return true;
        return false;
    };

    // Splice a mined replacement into a trial copy and keep it only if the
    // WHOLE function gets cheaper. With multi-use harvesting the slice's
    // local cost delta overstates the win (shared sub-expressions stay live
    // for their other users), so mine_function's local comparison is no
    // longer a sufficient adoption test.
    auto try_adopt = [&](const HarvestedSliceInternal& h,
                         const MinedPattern& mp) -> bool {
        Window w;
        w.block_name = h.public_part.source_block;
        w.root_name = h.root_name;
        w.instrs = h.slice_instrs;
        for (auto& in : h.slice_instrs) w.names.insert(in->name());
        w.live_ins = h.live_ins;

        auto trial = ir::deep_copy_function(*out);
        if (!splice_inplace(*trial, w, *mp.replacement, fresh_counter))
            return false;
        if (!ir::validate_function(*trial)) return false;
        const double trial_score = engine_->analyse(*trial).score;
        if (trial_score <= out_score) {
            stats_.splices_rejected_by_score++;
            return false;
        }
        out = std::move(trial);
        out_score = trial_score;
        for (auto& in : h.slice_instrs) consumed.insert(in->name());
        ++applied;
        stats_.harvest_rewrites_applied++;
        return true;
    };

    // Mine one slice with the candidate length capped at the slice's own
    // size: a replacement LONGER than the slice is (essentially) never
    // cheaper, and for 1–2-instruction slices the uncapped L2/L3
    // enumeration burns hundreds of thousands of candidates per fruitless
    // slice — which is what used to exhaust the time budget on large
    // functions before any select job ran.
    auto mine_capped = [&](const HarvestedSliceInternal& h,
                           const MineAssumptions* ma)
        -> std::optional<MinedPattern> {
        const size_t saved = config_.max_length;
        config_.max_length = std::min(saved, h.slice_instrs.size());
        auto mp = mine_function(*h.public_part.mini_function, ma);
        config_.max_length = saved;
        return mp;
    };

    // ── Select-form: mine select arms under the select condition ──────
    // -O3 output is select-heavy, so this is where conditional equivalence
    // actually fires on real optimised IR: for `select %c, %a, %b`, arm %a
    // may be replaced by anything equivalent under %c (and %b under NOT %c)
    // as long as the select is the arm's only user. Runs BEFORE the generic
    // slice loop: on large functions the shared time budget runs out, and
    // these jobs have the highest hit-rate on -O3 code.
    if (use_pcs && !out_of_time()) {
        // Distinct-instruction user counts (an arm observable only through
        // its select is replaceable under the select's condition).
        std::unordered_map<std::string, size_t> user_count;
        for (auto& block : fn.blocks()) {
            if (!block) continue;
            for (auto& inst : block->instructions()) {
                if (!inst) continue;
                std::unordered_set<std::string> seen;
                for (auto& op : inst->operands())
                    if (op && op->has_name() && seen.insert(op->name()).second)
                        user_count[op->name()]++;
            }
        }
        std::vector<SelectArmJob> sel_jobs;
        for (auto& block : fn.blocks()) {
            if (!block) continue;
            std::unordered_map<std::string, size_t> pos;
            for (size_t i = 0; i < block->size(); ++i) {
                auto in = block->instruction(i);
                if (in && in->has_name()) pos[in->name()] = i;
            }
            for (size_t i = 0; i < block->size(); ++i) {
                auto inst = block->instruction(i);
                if (!inst || inst->opcode() != ir::Opcode::Select) continue;
                for (bool arm : {true, false}) {
                    // Depth-1 job first: the arm's operands stay opaque
                    // live-ins, so identity/constant folds under the
                    // condition (`and t,-2` -> `t`) surface as length-0
                    // leaf candidates — found immediately. A full-depth
                    // job follows for arms whose whole chain simplifies;
                    // it is far more expensive (large enumeration pools),
                    // so it must not be the only attempt.
                    for (size_t depth : {size_t(1), config_.harvest_max_depth}) {
                        auto job = harvest_select_arm(
                            fn, block, pos, user_count, inst, arm, depth,
                            config_.harvest_max_depth,
                            config_.harvest_across_memory);
                        // The full-depth job duplicates the depth-1 one when
                        // the arm has no in-block operand chain to follow.
                        if (job && (depth == 1 || job->h.slice_instrs.size() > 1))
                            sel_jobs.push_back(std::move(*job));
                        if (sel_jobs.size() >= config_.max_windows_per_function)
                            goto select_done;
                        if (depth >= config_.harvest_max_depth) break;
                    }
                }
            }
        }
select_done:
        // Cheapest jobs first: depth-1 arm slices mine in milliseconds (the
        // identity/constant folds live at L0–L1), while full-depth jobs can
        // burn seconds of enumeration each. Under a shared time budget the
        // fast, high-yield jobs must all run before any expensive one.
        std::stable_sort(sel_jobs.begin(), sel_jobs.end(),
                         [](const SelectArmJob& a, const SelectArmJob& b) {
                             return a.h.slice_instrs.size() <
                                    b.h.slice_instrs.size();
                         });
        stats_.slices_harvested += sel_jobs.size();
        for (auto& j : sel_jobs) {
            if (out_of_time()) break;
            if (overlaps_consumed(j.h)) continue;
            stats_.pc_slices_assumed++;
            MineAssumptions ma;
            ma.condition_fn = j.cond_mini;
            ma.condition_negated = j.cond_negated;
            auto mp = mine_capped(j.h, &ma);
            if (!mp || !mp->replacement) continue;
            stats_.slices_mined++;
            stats_.pc_slices_mined++;
            try_adopt(j.h, *mp);
        }
    }

    for (auto& h : slices) {
        if (out_of_time()) break;
        if (overlaps_consumed(h)) continue;

        // ── Path-condition assumptions for this slice's seed block ─────
        MineAssumptions ma;
        if (use_pcs) {
            auto dit = dom_conds.find(h.public_part.source_block);
            if (dit == dom_conds.end())
                dit = dom_conds.emplace(
                    h.public_part.source_block,
                    dominating_conditions(fn, h.public_part.source_block)).first;
            ma.arg_assumptions =
                map_conditions_to_args(dit->second, h.public_part.var_origins);
            if (!ma.empty()) stats_.pc_slices_assumed++;
        }

        auto mp = mine_capped(h, ma.empty() ? nullptr : &ma);
        if (!mp || !mp->replacement) continue;
        stats_.slices_mined++;
        if (!ma.empty()) stats_.pc_slices_mined++;

        try_adopt(h, *mp);
    }
    if (applied == 0) return std::nullopt;

    // Gate 1: the splice must produce well-formed SSA (checked per-splice
    // above; kept as a belt-and-braces final check).
    if (!ir::validate_function(*out)) {
        stats_.reverify_rejections++;
        return std::nullopt;
    }
    // Gate 2 (default): re-prove the WHOLE rewrite equivalent to the original.
    // Each harvested slice was SMT-proved in isolation over its opaque-var
    // inputs; PC-mined slices were proved under branch conditions that the
    // CFG establishes at their splice point, so the rewritten function is
    // unconditionally equivalent as a whole and this gate can still prove
    // it. It catches any splice bug and refuses functions the verifier
    // can't model as a whole (memory/loops/too-large) unless the caller
    // opts into trusting the by-construction argument.
    if (config_.require_smt_reverify) {
        SMTConfig sc;
        sc.timeout_ms = config_.smt_timeout_ms;
        SMTVerifier verifier(sc);
        auto res = verifier.verify(fn, *out);
        if (res.status == VerificationResult::Equivalent) {
            if (whole_fn_proven) *whole_fn_proven = true;
            return out;
        }
        if (res.status == VerificationResult::Unknown &&
            config_.trust_unverified_slices)
            return out;
        stats_.reverify_rejections++;
        return std::nullopt;
    }
    return out;
}

// ── Path-condition-aware mining ──────────────────────────────────────────────────

std::vector<PathCondition> PeepholeMiner::compute_path_conditions(
    const ir::Function& fn, const std::string& target_block) const {
    std::vector<PathCondition> pcs;
    auto target = fn.block(target_block);
    if (!target) return pcs;

    // Ensure predecessors are populated (compute_predecessors is idempotent
    // and const-correct via const_cast — mirrors SMTVerifier's pattern).
    const_cast<ir::Function&>(fn).compute_predecessors();

    // Backward walk from target_block through predecessors, collecting one
    // PathCondition per conditional branch on any ACYCLIC CFG path from entry
    // to target. The `path` set (per worklist item) detects back-edges: a
    // predecessor already on the current path is a back-edge and is NOT
    // recursed through (loop back-edge conditions are only meaningful for
    // the loop header itself, not for blocks reached after the loop).
    //
    // Exception: a back-edge that is a DIRECT predecessor of the original
    // target (i.e. the target is a loop header and the back-edge is the
    // loop-continue edge) IS recorded — it is the loop-continue condition,
    // which is a genuine path condition for re-entering the loop header.
    // Back-edges encountered during recursion (not direct predecessors of
    // the target) are neither recorded nor followed.
    //
    // This matches Souper's addPathConditions (direct-predecessor PCs) plus
    // a one-level BlockPC-style recursion through acyclic diamond arms,
    // without the exponential blowup of full path enumeration.

    // Helper: build a PathCondition from a conditional branch's terminator.
    auto build_pc = [&](const std::shared_ptr<ir::BasicBlock>& pred_bb,
                        const std::shared_ptr<ir::Instruction>& term,
                        const std::string& pred_name,
                        bool negated) {
        PathCondition pc;
        pc.block_name = pred_name;
        pc.branch_inst_index = pred_bb->size() > 0 ? pred_bb->size() - 1 : 0;
        pc.negated = negated;
        if (term->num_operands() >= 1 && term->operand(0)) {
            auto cond = term->operand(0);
            auto cond_inst = std::dynamic_pointer_cast<ir::Instruction>(cond);
            if (cond_inst && cond_inst->opcode() == ir::Opcode::ICmp) {
                auto pred_it = cond_inst->metadata().find("pred");
                if (pred_it != cond_inst->metadata().end()) {
                    pc.predicate = static_cast<ir::CmpPredicate>(
                        std::stoul(pred_it->second));
                } else {
                    pc.predicate = ir::CmpPredicate::EQ;
                }
                if (cond_inst->num_operands() >= 2) {
                    auto lhs = cond_inst->operand(0);
                    auto rhs = cond_inst->operand(1);
                    pc.lhs_name = (lhs && lhs->has_name()) ? lhs->name() : "";
                    pc.rhs_name = (rhs && rhs->has_name()) ? rhs->name() : "";
                }
            } else {
                // Plain i1 condition: PC is (cond == 1) on the true edge,
                // (cond == 0) on the false edge.
                pc.predicate = ir::CmpPredicate::EQ;
                pc.lhs_name = cond->has_name() ? cond->name() : "";
                pc.rhs_name = negated ? "0" : "1";
            }
        } else {
            pc.predicate = ir::CmpPredicate::EQ;
            pc.lhs_name = "";
            pc.rhs_name = "";
        }
        return pc;
    };

    struct WorkItem {
        std::string block;
        std::unordered_set<std::string> path;  // blocks on the current backward walk
    };
    std::vector<WorkItem> worklist;
    worklist.push_back({target_block, {target_block}});

    while (!worklist.empty()) {
        WorkItem item = std::move(worklist.back());
        worklist.pop_back();
        const std::string& bb_name = item.block;
        auto bb = fn.block(bb_name);
        if (!bb) continue;
        const bool is_original_target = (bb_name == target_block);

        for (const auto& pred_name : bb->predecessors()) {
            auto pred_bb = fn.block(pred_name);
            if (!pred_bb) continue;
            auto term = pred_bb->terminator();
            if (!term) continue;
            if (term->opcode() != ir::Opcode::Br) continue;

            const bool is_back_edge = (item.path.count(pred_name) > 0);

            // Record the edge condition iff it is conditional AND either
            // (a) not a back-edge, or (b) a direct predecessor of the
            // original target (loop-continue condition for a loop header).
            const bool record = !is_back_edge || is_original_target;

            if (record) {
                const auto& md = term->metadata();
                auto true_it = md.find("true_bb");
                auto false_it = md.find("false_bb");
                if (true_it != md.end() && false_it != md.end()) {
                    const bool true_matches  = (true_it->second  == bb_name);
                    const bool false_matches = (false_it->second == bb_name);
                    if (true_matches && false_matches) {
                        // Both edges to same block — no constraint. Skip.
                    } else if (true_matches) {
                        pcs.push_back(build_pc(pred_bb, term, pred_name, false));
                    } else if (false_matches) {
                        pcs.push_back(build_pc(pred_bb, term, pred_name, true));
                    }
                    // else: neither edge matches — shouldn't happen, skip.
                }
                // Unconditional branch (dest_bb only) — no PC to record.
            }

            // Recurse through non-back-edges only (back-edges would cycle).
            if (!is_back_edge) {
                WorkItem next{pred_name, item.path};
                next.path.insert(pred_name);
                worklist.push_back(std::move(next));
            }
        }
    }
    return pcs;
}

std::optional<std::shared_ptr<ir::Function>> PeepholeMiner::mine_with_path_conditions(
    const ir::Function& fn, bool* whole_fn_proven) {
    // Full PC-aware implementation (task I1 landed): harvest slices as
    // harvest_and_rewrite does, but mine each one under the dominating
    // branch conditions of its seed block, proved via
    // SMTVerifier::verify_with_assumptions. Falls back to plain harvesting
    // per-slice when no condition is expressible over the slice's live-ins.
    return harvest_and_rewrite_impl(fn, config_.use_path_conditions,
                                    whole_fn_proven);
}

pattern::OptimisationPattern PeepholeMiner::to_pattern(
    const MinedPattern& mp, const pattern::ArchDescriptor& arch) {
    pattern::OptimisationPattern p;
    const std::string fname = mp.source ? mp.source->name() : "fn";
    p.id = "mined_" + fname + "_" +
           std::to_string(StochasticSearch::structural_hash(*mp.replacement));
    p.name = "mined:" + fname;
    p.description = "SMT-verified enumerative rewrite of " + fname +
                    " (score " + std::to_string(mp.source_score) + " -> " +
                    std::to_string(mp.replacement_score) + ")";
    p.source_function = mp.source;
    p.replacement_function = mp.replacement;
    if (mp.source) p.source_ir = mp.source->to_string();
    if (mp.replacement) p.replacement_ir = mp.replacement->to_string();
    p.discovered_arch = arch;
    // avg_speedup as a cost ratio (scores are negative: -cost). Guard /0.
    if (mp.replacement_score != 0.0)
        p.avg_speedup = mp.source_score / mp.replacement_score;
    p.verification_count = 1;
    p.scope = pattern::OptimisationPattern::Scope::InstructionLevel;
    p.tags = {"mined", "integer", "smt-verified"};
    return p;
}

}  // namespace clunk::search
