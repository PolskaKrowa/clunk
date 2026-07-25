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
 * Clunk Pipeline — orchestrates the full optimisation pipeline:
 * Pattern Library → Stochastic Search → Evolutionary Search → SMT Verify
 */
#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include <unordered_set>
#include <functional>
#include "clunk/IR/Module.h"
#include "clunk/IR/Function.h"
#include "clunk/Evaluator/EvaluationEngine.h"
#include "clunk/Evaluator/MCACostModel.h"
#include "clunk/Search/StochasticSearch.h"
#include "clunk/Search/EvolutionarySearch.h"
#include "clunk/Search/SMTVerifier.h"
#include "clunk/Search/PeepholeMiner.h"
#include "clunk/Search/EgraphRewriter.h"
#include "clunk/Search/RewriteCache.h"
#include "clunk/Pattern/PatternLibrary.h"

namespace clunk {

// ── Pipeline configuration ──────────────────────────────────────────────
struct PipelineConfig {
    // Optimisation level:
    //   0 = none (pass-through)
    //   1 = pattern library only (cheap, no search)
    //   2 = continuous search (patterns + stochastic + evolutionary + SMT)
    //   3 = continuous search with larger per-round budgets
    // At level >= 2 the pipeline runs souper/minotaur-style refinement
    // ROUNDS: each round searches from the current best, verifies the top
    // candidates against the ORIGINAL function with SMT, and adopts the
    // best verified improvement as the new baseline. Rounds continue until
    // `convergence_rounds` consecutive rounds find no improvement, until
    // `max_rounds` is reached, or until the time budget is exhausted —
    // there is no fixed cap on how far a function can be optimised.
    unsigned opt_level = 2;

    // Time budget in seconds (0 = no limit)
    double time_budget = 0.0;

    // Worker threads for MODULE-level parallelism: functions are independent,
    // so they are optimised concurrently. 0 = auto (hardware_concurrency),
    // 1 = sequential (deterministic — search RNG isn't shared across threads,
    // so results can differ by thread count). The shared evaluator (thread-safe
    // cache) and read-only pattern library are shared; each worker gets its own
    // search RNG / stats / Z3 context.
    size_t num_threads = 0;

    // ── Continuous-refinement controls ─────────────────────────────────
    // Maximum refinement rounds per function. 0 = unlimited (bounded only
    // by convergence detection and the time budget).
    size_t max_rounds = 0;

    // Stop refining a function after this many consecutive rounds without
    // an adopted improvement. Minimum 1.
    size_t convergence_rounds = 3;

    // ── Per-function time cap ───────────────────────────────────
    // Maximum wall-clock seconds to spend on ANY ONE function. 0 = no cap
    // (bounded only by the module-level time_budget). When set, forces the
    // search to move on from a stuck function so other functions get their
    // fair share of the budget. Default 0 (off) for backward compat; the
    // CLI sets a sensible default.
    double max_time_per_function = 0.0;

    // Souper-style soundness (default): when a working SMT prover is
    // available, only adopt candidates it PROVES equivalent to the
    // original. Set true to also adopt the best-scoring candidate the
    // prover returned Unknown for (functions with memory ops, floats, or
    // loops cannot currently be proved) — best-effort, like the legacy
    // behaviour, except candidates proved NotEquivalent are still always
    // rejected. Ignored when no prover is available (the evaluator is
    // trusted in that case).
    bool trust_unverified = false;

    // Target architecture
    pattern::ArchDescriptor target_arch;

    // Search configurations
    search::StochasticConfig stochastic_config;
    search::EvolutionaryConfig evolutionary_config;
    search::SMTConfig smt_config;

    // SMT-verified peephole miner (Souper-style enumerative superoptimisation).
    // At opt_level >= 2 the miner runs as one more candidate source per round:
    // it extracts pure-integer straight-line slices from the current baseline,
    // proves cheaper equivalents with SMT, and splices them back — the lever
    // that finds wins the mutation search cannot. Disabled
    // automatically when SMT is unavailable/skipped (it is unsound without a
    // real prover).
    bool enable_peephole_miner = true;
    search::MinerConfig peephole_config;

    // ── Souper/Minotaur-style harvesting ──────────────────────────────
    // When true, the mining_phase additionally calls
    // PeepholeMiner::harvest_and_rewrite() on the current baseline, which
    // harvests every integer instruction (replacing unsupported operands
    // with opaque vars) and SMT-proves cheaper rewrites. This is the
    // generalisation of mine_and_rewrite that can find wins on functions
    // with memory ops, calls, and (eventually) loops. Default ON at
    // opt_level >= 2.
    bool enable_harvest_miner = true;

    // ── llvm-mca final ranking (Minotaur-style) ──────────────────────────
    // When true and llc + llvm-mca are on PATH, verify_and_select re-ranks
    // improving candidates by MEASURED cycle count (llc → llvm-mca) before
    // adoption, and drops "improvements" the machine model says are not
    // faster. Refinement only — soundness never depends on it, and any
    // candidate the toolchain cannot lower falls back to the built-in
    // cost-model ranking. Off by default (each measurement costs two
    // subprocess launches). Enable with --mca.
    bool use_mca_ranker = false;

    // ── Loop optimisation (LICM + constant-trip full unrolling) ─────────
    // Sound-by-construction loop rewrites (the SMT verifier refuses
    // back-edges, so these carry no proof but are exact by construction):
    // loop-invariant code motion into the preheader, and full unrolling of
    // single-block loops whose trip count folds to a constant — which
    // converts an unverifiable loop function into straight-line code the
    // SMT-verified phases can then attack. The cost model arbitrates the
    // unrolling tradeoff like any other candidate. Default ON.
    bool enable_loop_opt = true;

    // ── Alias-aware memory optimisation ──────────────────────────────────
    // Block-local store-to-load forwarding, redundant-load elimination and
    // dead-store elimination over a conservative alias oracle (distinct
    // allocas, noalias arguments, disjoint constant GEPs). Sound by
    // construction. Default ON.
    bool enable_mem_opt = true;

    // ── Dataflow pruning (dead-code + unreachable-block elimination +
    //     known-bits-driven constant/branch folding) ─────────────────────
    // Runs ir::prune_dataflow() on the current baseline every round,
    // before the mutation searches see it: dead-code elimination (def-use
    // closure — no CFG traversal needed, correct across loops), removal
    // of blocks unreachable from entry (with Phi fixup), and a lightweight
    // Souper-style "infer known bits" pass (see clunk/Analysis/
    // KnownBits.h) that folds provably-constant values, drops no-op AND
    // masks, resolves provably-true/false comparisons, and turns a
    // conditional Br with a now-constant condition into an unconditional
    // one (which is what feeds the unreachable-block pass its work).
    // Sound by construction — same trust tier as LoopOpt/MemOpt. This
    // doesn't just shrink the FINAL output: a smaller, cleaner baseline is
    // cheaper for every later phase this round and every round after
    // (fewer instructions to mutate, lower to the e-graph, or measure).
    // Default ON.
    bool enable_dataflow_prune = true;

    // ── Interprocedural inlining ─────────────────────────────────────────
    // Inline small single-block module-internal callees before optimising
    // each function. Removes opaque calls — which also makes the caller
    // SMT-verifiable — and is only kept when the cost model scores the
    // inlined body cheaper. Default ON.
    bool enable_inliner = true;

    // ── Vector-intrinsic synthesis (Minotaur-style) ──────────────────────
    // When true, run_on_function runs a vector_phase() each round on
    // functions containing vector operations: it recognises scalar idioms
    // over vector values (lane-wise op groups, horizontal-reduction trees,
    // shuffle chains) and re-expresses them as vector instructions and
    // clunk.vector.reduce.* intrinsics. Every rewrite is gated by the cost
    // model AND an SMT equivalence proof (the verifier lane-blasts vector
    // functions via ir::Scalarizer). Default ON at opt_level >= 2; inert
    // on vector-free functions.
    bool enable_vector_synth = true;

    // ── Equality-saturation candidate generation ─────────────────────
    // When true, the pipeline runs a new egraph_phase() before the mining
    // phase each round: it lowers the current baseline to an e-graph,
    // applies pattern-library rewrites as e-graph rules, saturates, and
    // extracts the cheapest representative as a Candidate (sound=false:
    // pattern-library rewrites need SMT verification). Solves the
    // pattern-ordering problem for free. Default OFF (the pattern library
    // must be non-empty for this to be useful).
    bool enable_egraph_phase = false;

    // ── STOKE-style stochastic search moves ────────────────────────────
    // When true, the stochastic search may use the unsound STOKE-style
    // mutation kinds (OpcodeReplace, OperandReplace, OperandSwap,
    // InstructionInsert, InstructionReplace). Candidates produced via
    // these moves are NOT sound-by-construction and MUST be SMT-verified
    // before adoption. Default OFF (preserves today's sound-by-construction
    // search behaviour).
    bool allow_unsound_mutations = false;

    // Number of random test vectors the stochastic search uses as a
    // pre-filter before sending candidates to SMT. 0 = disabled (today's
    // behaviour). 32 = Massalin's recommended value. Default 0 — enable
    // explicitly with --test-vectors N when the search is producing many
    // non-equivalent candidates that SMT is rejecting.
    size_t test_vector_count = 0;

    // ── Persistent SMT rewrite cache ──────────────────────────────────
    // When non-empty, the SMT verifier consults this file-backed cache
    // before calling Z3 and writes the result back after. The cache is
    // human-readable (one entry per line, tab-separated). Default empty
    // (no caching).
    std::string smt_cache_path;

    // Pattern library path (empty = in-memory only)
    std::string pattern_library_path;

    // GPU optimisation
    bool enable_gpu_opt = true;
    bool enable_launch_opt = true;

    // Reporting
    bool verbose = false;
    bool dump_candidates = false;
    std::string dump_dir;

    // ── Scale-control fields ──────────────────────────────────────────
    // Maximum function size (instruction count) to attempt superoptimisation.
    // Functions larger than this are passed through unchanged. Default is
    // 512 — site-driven mutation keeps the per-iteration cost low on big
    // functions (it was 200 back when every iteration deep-copied the
    // whole function on a blind mutation guess). Tunable via
    // --max-function-size.
    size_t max_function_size = 512;

    // ── Oversized-function mining ───────────────────────────────────────
    // Functions above max_function_size are passed through for the full\n    // search stack but still get a miner-only refinement path instead of\n    // a skip, up to this second cap (0 = disabled). Because the
    // whole-function SMT re-verification cannot model functions this large,
    // miner wins on them are only adoptable in trust_unverified mode (each
    // slice still carries its own SMT proof).
    size_t max_mining_function_size = 8192;

    // If true, skip SMT verification entirely (always return Unknown).
    // Useful for very large inputs where SMT would never terminate.
    // Tunable via --skip-smt.
    bool skip_smt = false;
};

// ── Pipeline result ─────────────────────────────────────────────────────
struct PipelineResult {
    std::shared_ptr<ir::Module> optimised_module;

    // Per-function results
    struct FunctionResult {
        std::string function_name;
        std::shared_ptr<ir::Function> original;
        std::shared_ptr<ir::Function> optimised;
        double score_original = 0.0;
        double score_optimised = 0.0;
        double improvement_ratio = 1.0;
        bool verified = false;        // SMT verified
        size_t patterns_applied = 0;  // Patterns from library
        size_t rounds_run = 0;        // Refinement rounds executed
        size_t improvements_adopted = 0;  // Rounds that improved the baseline
        double time_spent_ms = 0.0;
    };

    std::map<std::string, FunctionResult> function_results;

    // Aggregate statistics
    double total_time_ms = 0.0;
    size_t total_functions_processed = 0;
    size_t total_optimised = 0;
    double avg_improvement = 1.0;
};

// ── Progress callback ───────────────────────────────────────────────────
using ProgressCallback = std::function<void(const std::string& stage,
                                             const std::string& function_name,
                                             double progress)>;

// ── The Pipeline ────────────────────────────────────────────────────────
class Pipeline final {
public:
    explicit Pipeline(const PipelineConfig& config = {});

    // Run the full optimisation pipeline on a module
    PipelineResult run(const ir::Module& module);

    // Run on a single function
    PipelineResult::FunctionResult run_on_function(const ir::Function& fn);

    // Set progress callback
    void set_progress_callback(ProgressCallback cb) { progress_cb_ = std::move(cb); }

    // Access sub-components
    evaluator::EvaluationEngine& evaluation_engine() { return eval_engine_; }
    search::StochasticSearch& stochastic_search() { return searchers_.stochastic; }
    search::EvolutionarySearch& evolutionary_search() { return searchers_.evolutionary; }
    search::SMTVerifier& smt_verifier() { return searchers_.smt; }
    pattern::PatternLibrary& pattern_library() { return pattern_lib_; }

    // Configuration
    PipelineConfig& config() { return config_; }
    const PipelineConfig& config() const { return config_; }

private:
    // Per-thread search state. The evaluator (thread-safe cache) and pattern
    // library (read-only match/apply) are shared across workers, but each
    // worker needs its own RNG, statistics, and Z3 context — those live here.
    // One Searchers is used for sequential runs; the parallel path gives each
    // worker thread its own.
    struct Searchers {
        search::StochasticSearch stochastic;
        search::EvolutionarySearch evolutionary;
        search::SMTVerifier smt;
        uint64_t last_mined_hash = 0;   // peephole-mining guard, reset per fn
        uint64_t last_egraph_hash = 0;  // e-graph guard, reset per fn
        uint64_t last_vector_hash = 0;  // vector-synthesis guard, reset per fn
        uint64_t last_loop_hash = 0;    // loop-opt guard, reset per fn
        uint64_t last_mem_hash = 0;     // mem-opt guard, reset per fn
        uint64_t last_prune_hash = 0;   // dataflow-prune guard, reset per fn
        // Rejected-candidate cache: structural hashes of candidates
        // that were SMT-rejected this round-chain. Prevents re-proposing
        // and re-rejecting the same candidate on unchanged baselines.
        std::unordered_set<uint64_t> rejected_hashes;
        Searchers(const PipelineConfig& cfg, evaluator::EvaluationEngine* eng)
            : stochastic(cfg.stochastic_config, eng),
              evolutionary(cfg.evolutionary_config, eng),
              smt(cfg.smt_config) {}
    };

    // Optimise one function using the given per-thread search state. The public
    // run_on_function() delegates to this with the shared sequential Searchers.
    PipelineResult::FunctionResult run_on_function(const ir::Function& fn,
                                                   Searchers& s);

    // Pipeline stages
    std::shared_ptr<ir::Function> apply_patterns(
        const ir::Function& fn,
        PipelineResult::FunctionResult& result);

    std::vector<search::Candidate> stochastic_phase(
        const ir::Function& fn, double remaining_seconds, Searchers& s);

    std::vector<search::Candidate> evolutionary_phase(
        const ir::Function& fn,
        const std::vector<search::Candidate>& stochastic_candidates,
        double remaining_seconds, Searchers& s);

    // SMT-verified peephole mining of `fn` (the current baseline). Returns the
    // miner's whole-function rewrite as a single sound candidate, or empty if
    // nothing was proven. Each distinct baseline is mined at most once
    // (structural-hash guard in `s`) since the miner is deterministic.
    std::vector<search::Candidate> mining_phase(const ir::Function& fn,
                                                Searchers& s);

    // ── Equality-saturation candidate generation ────────────────────
    // Lower `fn` (the current baseline) to an e-graph, apply pattern-library
    // rewrites as e-graph rules, saturate, and extract the cheapest
    // representative as a Candidate (sound=false: pattern-library rewrites
    // need SMT verification). Returns empty if disabled or no improvement.
    // Guarded by last_egraph_hash — skips re-running on an unchanged
    // baseline (eliminates the repeated-identical-candidate pathology).
    std::vector<search::Candidate> egraph_phase(const ir::Function& fn,
                                                 Searchers& s);

    // ── Vector-intrinsic synthesis phase ─────────────────────────────────
    // Runs VectorSynthesizer on the current baseline when it contains
    // vector operations. The synthesiser SMT-proves its rewrite internally
    // (Candidate::sound reflects the proof), so verify_and_select can adopt
    // it without re-verification. Guarded by last_vector_hash — a given
    // baseline is synthesised at most once (the pass is deterministic).
    std::vector<search::Candidate> vector_phase(const ir::Function& fn,
                                                 Searchers& s);

    // ── Loop-optimisation phase ──────────────────────────────────────────
    // LICM + constant-trip full unrolling of the current baseline. Both
    // rewrites are exact by construction (Candidate::sound = true); the
    // cost model decides adoption. Guarded by last_loop_hash.
    std::vector<search::Candidate> loop_phase(const ir::Function& fn,
                                               Searchers& s);

    // ── Memory-optimisation phase ────────────────────────────────────────
    // Alias-aware store-to-load forwarding / RLE / DSE on the current
    // baseline. Exact by construction. Guarded by last_mem_hash.
    std::vector<search::Candidate> mem_phase(const ir::Function& fn,
                                              Searchers& s);

    // ── Dataflow-pruning phase ───────────────────────────────────────────
    // ir::prune_dataflow(): DCE + unreachable-block elimination + known-
    // bits-driven constant/branch folding on the current baseline. Exact
    // by construction. Guarded by last_prune_hash.
    std::vector<search::Candidate> dataflow_prune_phase(const ir::Function& fn,
                                                         Searchers& s);

    // Select the best candidate that beats `current`, verifying against
    // `original` (the soundness anchor for the whole refinement chain).
    // Sets result.verified when the selection was SMT-proved.
    std::shared_ptr<ir::Function> verify_and_select(
        const ir::Function& original,
        const ir::Function& current,
        const std::vector<search::Candidate>& candidates,
        PipelineResult::FunctionResult& result,
        Searchers& s);

    // Seconds until the active deadline (+inf when no deadline is set).
    double remaining_seconds() const;

#ifdef CLUNK_HAS_GPU
    std::shared_ptr<ir::Function> gpu_optimise(
        const ir::Function& fn);
#endif

    void report_progress(const std::string& stage,
                          const std::string& fn_name,
                          double progress);

    PipelineConfig config_;
    evaluator::EvaluationEngine eval_engine_;
    // llvm-mca final ranker (opt-in, see PipelineConfig::use_mca_ranker).
    // One shared instance so the measurement cache spans rounds and
    // worker threads (its methods are thread-safe).
    evaluator::MCACostModel mca_;
    Searchers searchers_;              // shared sequential search state
    pattern::PatternLibrary pattern_lib_;
    ProgressCallback progress_cb_;
    std::mutex progress_mutex_;        // guards progress_cb_ across workers

    // ── Persistent SMT rewrite cache ──────────────────────────────────
    // Shared across all worker threads. Created lazily in the constructor
    // when config_.smt_cache_path is non-empty. Plumbed into each worker's
    // SMTVerifier via SMTConfig::cache.
    std::shared_ptr<search::RewriteCache> rewrite_cache_;

    // Wall-clock deadline shared by run() and run_on_function() so each
    // refinement round sees the true remaining budget.
    std::chrono::steady_clock::time_point deadline_{};
    bool has_deadline_ = false;
};

} // namespace clunk