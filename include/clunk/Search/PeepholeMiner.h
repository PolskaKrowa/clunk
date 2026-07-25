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
 * Clunk Peephole Miner — SMT-verified enumerative superoptimisation.
 *
 * The stochastic / evolutionary search only applies mutations that are a
 * SUBSET of what clang -O3 already does, so on -O3 output it finds nothing
 * (see scripts/beat_o3.sh). This miner is the Souper-style lever that can
 * actually beat -O3 on integer code: for a small, single-block, integer-only
 * function it ENUMERATES short straight-line instruction sequences, keeps the
 * ones the cost model ranks strictly cheaper, and PROVES equivalence to the
 * source with the SMT verifier. Winners are emitted as reusable
 * OptimisationPatterns (persisted via PatternLibrary), so they then apply to
 * future inputs through the normal pipeline's PatternGuided path.
 *
 * Scope: source must be one basic block ending in `ret <intN>` with N >= 2.
 * Eligible instructions are the arithmetic/bitwise binops the enumerator emits
 * (add/sub/mul/and/or/xor/shl/lshr/ashr), the div/rem family (udiv/sdiv/urem/
 * srem — ingested so the miner can prove e.g. `udiv x, 8` -> `lshr x, 3`, but
 * never emitted, being far too costly), `icmp` (yields i1), and `select`
 * (consumes an i1 cond). Every non-condition value shares one main bit-width.
 *
 * Two synthesis strategies feed the SMT gate: blind shortest-first enumeration
 * of the binop set, and a targeted compare-select pass that builds the
 * `select(icmp pred a, b, t, f)` shape (min/max/clamp/sign-select) which blind
 * enumeration cannot reach affordably. Anything outside this model is skipped
 * (mine_function returns nullopt) — sound: the miner never emits an unproven
 * pattern.
 */
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>
#include "clunk/IR/Function.h"
#include "clunk/IR/Instruction.h"  // ir::CmpPredicate (used by PathCondition)
#include "clunk/Evaluator/EvaluationEngine.h"
#include "clunk/Pattern/PatternLibrary.h"
#include "clunk/Search/SMTVerifier.h"  // ArgAssumption (PC-aware mining)

namespace clunk::search {

// ── Miner configuration ─────────────────────────────────────────────────────
struct MinerConfig {
    size_t max_length = 3;              // max instructions in a synthesised replacement
                                        // (length 0 = a bare leaf/constant result)
    size_t max_candidates = 200000;     // cap on enumerated candidates per function
    size_t max_smt_checks = 256;        // cap on SMT calls per function
    size_t max_source_instructions = 12;// skip larger source functions (excl. ret)
    size_t max_constants = 16;          // cap on the constant operand pool
    unsigned smt_timeout_ms = 5000;     // per-check Z3 timeout
    // Extra constants to seed the operand pool (on top of {0,1,2,-1}, the
    // source's own constants, and small powers of two).
    std::vector<int64_t> extra_constants;

    // ── Multi-block slice mining (mine_and_rewrite) ─────────────────────
    // A "window" is a pure-integer, single-use, straight-line sub-expression
    // extracted from ONE block of an arbitrary (possibly multi-block, memory/
    // loop-containing) function. Each window is mined as its own mini-function
    // and proven wins are spliced back.
    size_t min_window_instructions = 2;  // ignore trivial single-op windows
    size_t max_window_instructions = 8;  // largest slice to lift into a mini-fn
    size_t max_windows_per_function = 64;// cap slices mined per function
    // Wall-clock budget for one mine_and_rewrite / harvest_and_rewrite call
    // (0 = unbounded). Checked between slices, not inside a slice's own
    // enumeration/SMT work, so a call can overrun by roughly one slice's
    // worth of SMT. Without this, a large function's slice loop could run
    // far past the pipeline's deadline (each slice may spend many SMT
    // calls) — the pipeline sets it from its remaining budget.
    double time_budget_seconds = 0.0;
    // Require a whole-function SMT re-verification (rewritten vs original)
    // before accepting a rewrite. Sound: catches any splice bug and refuses
    // functions SMT cannot model (memory/loops/too-large) unless the escape
    // hatch below is set. Default true.
    bool require_smt_reverify = true;
    // Escape hatch: when the whole-function re-verify is `Unknown` (e.g. the
    // function has memory or loops OUTSIDE the mined slice), accept anyway on
    // the strength of each slice's own SMT proof + validate_function. Off by
    // default (keeps mine_and_rewrite sound by construction only).
    bool trust_unverified_slices = false;

    // ── Souper/Minotaur-style harvesting (harvest_slice / harvest_and_rewrite)
    // Harvest EVERY integer instruction, turning unsupported
    // operands (memory/call/phi/terminator) into opaque `Var`s (mini-function
    // arguments). The depth bound keeps the SMT query tractable — Minotaur's
    // empirical sweet spot is B=4.
    size_t harvest_max_depth = 4;        // max instructions in a harvested slice
    // When true, harvesting stops at the innermost loop boundary.
    // Minotaur-style loop-aware harvesting requires a LoopInfo analysis that
    // clunk does not yet have; until it lands, the harvester treats loop
    // back-edges conservatively (cross-block operands become opaque vars
    // rather than being followed), which is sound but less aggressive.
    bool harvest_across_loops = false;
    // When true (default), Load/Store/GEP/Alloca operands become opaque
    // vars. When false, the harvester refuses to harvest any slice whose
    // operand chain contains a memory op (matching the old extract_window
    // bail behaviour — use only when you want to guarantee a pure-integer
    // slice).
    bool harvest_across_memory = true;

    // ── Path-condition-aware mining (mine_with_path_conditions) ─────
    // When true, mine_with_path_conditions computes the branch conditions
    // dominating each harvested slice's seed block (single-predecessor
    // chain — every execution reaching the block came through those edges),
    // maps them onto the slice's live-in arguments, and conjoins them into
    // the SMT equivalence query via SMTVerifier::verify_with_assumptions.
    // This unlocks "simplify under branch" wins (e.g. `x & 3` -> `x` inside
    // an `if (x < 4)` block) that unconditional equivalence cannot reach —
    // the class of rewrite -O3 systematically misses. Default ON.
    bool use_path_conditions = true;

    // ── CEGIS CEGIS constant synthesis ─────────────────
    // When true, mine_function tries CEGIS (counterexample-guided inductive
    // synthesis) as a fallback when straight-line enumeration finds no win.
    // For each "template shape" (mul x,? / add x,? / xor x,? / shl x,? /
    // lshr x,? / and x,? / or x,? / sub x,?), it builds a candidate with a
    // placeholder constant and calls SMTVerifier::synthesize_with_cegis to
    // find the constant value that makes the candidate equivalent to the
    // source. This lets the miner invent constants outside its 16-element
    // pool — e.g. magic-number multipliers for division-by-constant. Default
    // ON at opt_level >= 2.
    bool enable_cegis_synthesis = true;
    // Cap on CEGIS attempts per function (each attempt is one template shape).
    // 8 templates × ~50ms per CEGIS = ~400ms worst case.
    size_t max_cegis_attempts = 8;
};

// ── Conditional-mining context ───────────────────────────────────────
// Everything mine_function may assume about its inputs when proving a
// replacement. `arg_assumptions` are flat icmp-style conjuncts over the
// mini-function's positional args (dominating branch conditions);
// `condition_fn` is an arbitrary i1-returning mini-function over the SAME
// argument space (a select's condition — the shape that dominates -O3
// output, where branches become selects). Both are conjoined into the SMT
// equivalence query and used to filter the concrete pre-filter's vectors.
struct MineAssumptions {
    std::vector<ArgAssumption> arg_assumptions;
    std::shared_ptr<ir::Function> condition_fn;  // i1 mini-fn, same args
    bool condition_negated = false;              // assume NOT(condition_fn)
    bool empty() const { return arg_assumptions.empty() && !condition_fn; }
};

// ── A proven, strictly-cheaper rewrite ───────────────────────────────────────
struct MinedPattern {
    std::shared_ptr<ir::Function> source;       // eligible source mini-function
    std::shared_ptr<ir::Function> replacement;  // cheaper, SMT-equivalent
    double source_score = 0.0;                  // higher = better (see EvaluationEngine)
    double replacement_score = 0.0;
    size_t smt_checks = 0;                       // SMT calls spent finding it
};

// ── A Souper-style harvested slice ─────────────────────────────────────
// One integer instruction lifted out of an arbitrary function into a
// standalone mini-function, with every unsupported operand (memory op, call,
// phi, cross-block reference, depth-exceeded instruction) replaced by an
// opaque `argN` parameter. The mini-function is single-block, integer-only,
// ends in `ret <intN>`, and is directly mineable by mine_function().
struct HarvestedSlice {
    std::shared_ptr<ir::Function> mini_function;  // the harvested mini-fn
    std::string source_block;                      // block the slice was lifted from
    size_t source_inst_index;                      // index of the seed instruction
    std::vector<std::string> opaque_var_names;     // names of the opaque args (arg0..)
    // Mapping from opaque var name -> the original value's name it replaced
    // (a function-argument name, an instruction-result name, or "" for an
    // unnamed/constant operand — though constants are kept inline, not
    // promoted to opaque vars). Used to splice a proven rewrite back into the
    // original function.
    std::vector<std::pair<std::string, std::string>> var_origins;
};

// ── A path condition (PC) on a CFG edge ───────────────────────────────
// Represents one conjunct of the path condition holding at the entry of a
// target block: "the conditional branch at `branch_inst_index` in
// `block_name` was taken such that (lhs `predicate` rhs) is `!negated`".
// For a plain-i1 condition (not an icmp), `predicate` is EQ, `lhs_name` is
// the condition's name, and `rhs_name` is "1" (true edge) or "0" (false edge).
struct PathCondition {
    std::string block_name;            // predecessor block carrying the branch
    size_t branch_inst_index;          // index of the terminator (br) in that block
    ir::CmpPredicate predicate;        // icmp predicate (or EQ for plain i1)
    std::string lhs_name;              // icmp lhs name (or the condition name)
    std::string rhs_name;              // icmp rhs name (or "1"/"0" for plain i1)
    bool negated;                      // true if the path takes the false branch
};

// ── Peephole miner ───────────────────────────────────────────────────────────
class PeepholeMiner {
public:
    explicit PeepholeMiner(evaluator::EvaluationEngine* engine,
                           const MinerConfig& config = {});

    // Try to find a cheaper, SMT-equivalent replacement for one function.
    // Returns nullopt if the function is ineligible or no proven win exists.
    //
    // `assumptions` (optional): conditions the caller has established at
    // the splice site (dominating branches / a select's condition). When
    // given, the concrete pre-filter only uses test vectors satisfying
    // them, and equivalence is proved via
    // SMTVerifier::verify_with_assumptions — i.e. the returned replacement
    // is equivalent ONLY on the constrained input space, so it may only be
    // spliced into a program point where the assumptions are established.
    std::optional<MinedPattern> mine_function(
        const ir::Function& fn,
        const MineAssumptions* assumptions = nullptr);

    // Mine every function in a module, returning all proven wins.
    std::vector<MinedPattern> mine_module(const ir::Module& module);

    // Optimise an ARBITRARY function — multi-block, and even containing memory
    // ops or loops — by extracting pure-integer straight-line slices from its
    // blocks, mining each as a mini-function, and splicing the proven-cheaper
    // equivalents back. This is how the superoptimiser reaches beyond the
    // single-block scope of mine_function. Returns the rewritten function (with
    // one or more applied wins) or nullopt if nothing was proven/applied.
    //
    // Soundness: every applied slice rewrite is SMT-verified in isolation, and
    // (by default) the whole rewritten function is SMT re-verified against the
    // original — so a splice bug can never leak an incorrect rewrite. Functions
    // the verifier cannot model as a whole are conservatively left unchanged
    // unless config.trust_unverified_slices is set.
    // `whole_fn_proven` (optional out): set true iff the final whole-function
    // SMT re-verification returned Equivalent — i.e. the rewrite carries an
    // end-to-end proof, not just per-slice proofs. False when the rewrite was
    // accepted through the trust_unverified_slices escape hatch or when
    // require_smt_reverify is off. Callers should propagate this into
    // Candidate::sound instead of assuming every miner rewrite is proven.
    std::optional<std::shared_ptr<ir::Function>> mine_and_rewrite(
        const ir::Function& fn, bool* whole_fn_proven = nullptr);

    // ── Souper/Minotaur-style harvesting ──────────────────
    // Harvest a single integer instruction (`inst_index` in `block_name` of
    // `fn`) into a standalone mini-function, walking back through operands
    // up to `max_depth` instructions deep. Unsupported operands (memory ops,
    // calls, phis, terminators, cross-block references, multi-use
    // instructions, depth-exceeded instructions) become opaque `argN`
    // parameters. Returns nullopt if the seed is not an integer-typed
    // sliceable op (binop/icmp/select) with bit-width >= 2.
    //
    // The returned mini-function is single-block, integer-only, ends in
    // `ret <intN>`, and is directly mineable by mine_function(). The
    // `var_origins` field records which original value each opaque var
    // replaced, so a proven rewrite can be spliced back (see
    // harvest_and_rewrite).
    std::optional<HarvestedSlice> harvest_slice(
        const ir::Function& fn,
        const std::string& block_name,
        size_t inst_index,
        size_t max_depth) const;

    // Harvest ALL integer slices from `fn` (one per eligible integer
    // instruction), mine each as a mini-function via mine_function(), and
    // splice any proven-cheaper rewrite back into the original. This is the
    // generalisation of mine_and_rewrite that uses Souper-style harvesting
    // (opaque vars for unsupported operands) instead of the pure-integer
    // straight-line window extraction. Returns the rewritten function or
    // nullopt if no wins were proven/applied.
    //
    // Soundness: identical to mine_and_rewrite — every applied slice rewrite
    // is SMT-verified in isolation, and (by default) the whole rewritten
    // function is SMT re-verified against the original. The harvest itself
    // is sound by construction: opaque vars are unconstrained inputs, so a
    // proven equivalence over all opaque-var values is a fortiori valid for
    // the specific original operands they stand in for.
    std::optional<std::shared_ptr<ir::Function>> harvest_and_rewrite(
        const ir::Function& fn, bool* whole_fn_proven = nullptr);

    // ── Path-condition-aware mining ───────────────────────
    // Compute the path condition (PC) for `target_block` in `fn`: the set of
    // branch conditions on every CFG path from the entry block to
    // `target_block`. Returns one PathCondition per conditional branch on
    // any such path. For loops, the back-edge condition is included (the
    // loop-continue condition); k-bounded unrolling is NOT performed here.
    //
    // The PCs are returned as a vector of (block, branch_inst_index,
    // predicate, lhs, rhs, negated) tuples. Callers can inspect them
    // directly, or (once SMTVerifier gains an assumptions API — task I1)
    // conjoin them into the equivalence query to unlock "fold to constant
    // under branch" wins.
    std::vector<PathCondition> compute_path_conditions(
        const ir::Function& fn,
        const std::string& target_block) const;

    // Mine `fn` with path conditions: identical to harvest_and_rewrite,
    // except each harvested slice is mined under the branch conditions that
    // dominate its seed block. The dominating conditions are collected by
    // walking the block's SINGLE-predecessor chain upward (every execution
    // reaching the block necessarily came through those conditional edges,
    // so conjoining them is sound; blocks with >1 predecessor stop the walk
    // — a join point's incoming conditions are disjunctive, not
    // conjunctive). Conditions whose operands are not visible to the slice
    // (not a live-in / constant) are dropped, which only weakens the
    // assumption set and is therefore also sound.
    //
    // This unlocks rewrites that are only valid on a specific branch path
    // (e.g. `%t = and %x, 3` -> `%x` inside an `if (x <u 4)` block) — the
    // Souper-class wins that unconditional equivalence, and -O3, miss.
    // Returns the rewritten function or nullopt.
    std::optional<std::shared_ptr<ir::Function>> mine_with_path_conditions(
        const ir::Function& fn, bool* whole_fn_proven = nullptr);

    // ── CEGIS CEGIS constant synthesis ─────────────────
    // Try to find a cheaper equivalent for `fn` by building template
    // candidates with a placeholder constant and asking
    // SMTVerifier::synthesize_with_cegis to find the constant value.
    // Templates: for each live-in arg `a` and each binop opcode `op`,
    // build `op a, ?` (and `op ?, a` for non-commutative ops). If CEGIS
    // finds a constant that makes the template equivalent to `fn`, the
    // instantiated candidate is scored; if cheaper, it's returned as a
    // MinedPattern.
    //
    // Soundness: identical to mine_function — every returned rewrite is
    // SMT-verified (CEGIS runs verify() as its second query).
    std::optional<MinedPattern> mine_with_cegis(const ir::Function& fn);

    // Convert a mined win into a library pattern (source_function /
    // replacement_function populated; source_ir / replacement_ir rendered).
    static pattern::OptimisationPattern to_pattern(
        const MinedPattern& mp, const pattern::ArchDescriptor& arch);

    struct Stats {
        size_t functions_seen = 0;
        size_t eligible = 0;
        size_t mined = 0;          // functions for which a win was proven
        size_t candidates_tried = 0;
        size_t smt_checks = 0;
        // Multi-block slice mining (mine_and_rewrite).
        size_t windows_seen = 0;       // slices extracted
        size_t windows_mined = 0;      // slices with a proven-cheaper rewrite
        size_t rewrites_applied = 0;   // slices actually spliced in
        size_t reverify_rejections = 0;// rewrites dropped by the whole-fn gate
        // Souper/Minotaur-style harvesting (harvest_and_rewrite).
        size_t slices_harvested = 0;       // harvested slices extracted
        size_t slices_mined = 0;           // harvested slices with a proven win
        size_t harvest_rewrites_applied = 0;  // harvested slices spliced in
        size_t splices_rejected_by_score = 0; // wins that didn't improve the whole fn
        // Path-condition-aware mining (mine_with_path_conditions).
        size_t pc_slices_assumed = 0;      // slices mined under >=1 assumption
        size_t pc_slices_mined = 0;        // ... that yielded a proven-under-PC win
        // CEGIS constant synthesis.
        size_t cegis_attempts = 0;         // template shapes tried
        size_t cegis_wins = 0;             // templates where CEGIS found a constant
    };
    const Stats& stats() const { return stats_; }

    MinerConfig& config() { return config_; }
    const MinerConfig& config() const { return config_; }

private:
    // Shared implementation of harvest_and_rewrite / mine_with_path_conditions.
    // When `use_pcs` is true, each slice is mined under the dominating branch
    // conditions of its seed block (mapped onto the slice's live-in args).
    std::optional<std::shared_ptr<ir::Function>> harvest_and_rewrite_impl(
        const ir::Function& fn, bool use_pcs, bool* whole_fn_proven);

    evaluator::EvaluationEngine* engine_;
    MinerConfig config_;
    Stats stats_{};
};

} // namespace clunk::search
