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
 * Clunk Pattern Library — persistent store of optimisation patterns.
 * Self-improving: discovered patterns are stored and reused, and
 * the pattern matcher itself is Clunk-optimised.
 */
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <optional>
#include "clunk/IR/Module.h"
#include "clunk/IR/Function.h"

namespace clunk::pattern {

// ── Architecture descriptor ─────────────────────────────────────────────
struct ArchDescriptor {
    std::string name;             // e.g. "x86_64", "sm_80", "sm_90"
    std::string vendor;           // e.g. "intel", "nvidia", "amd"
    bool is_gpu = false;
    unsigned compute_capability = 0; // GPU compute capability (e.g. 80, 90)
    unsigned vector_width = 256;     // SIMD width in bits
    bool has_fma = false;
    bool has_avx2 = false;
    bool has_avx512 = false;
    bool has_sve = false;

    // Memory model differences
    unsigned l1_cache_kb = 32;
    unsigned l2_cache_kb = 256;
    unsigned l3_cache_kb = 8192;
    unsigned shared_mem_kb = 0;   // GPU shared memory
    unsigned warp_size = 1;       // GPU warp size

    // How different is this from another architecture?
    double distance(const ArchDescriptor& other) const;
};

// ── Optimisation pattern ────────────────────────────────────────────────
struct OptimisationPattern {
    std::string id;               // Unique pattern identifier
    std::string name;             // Human-readable name
    std::string description;      // What this pattern does

    // The source pattern — what to look for
    std::string source_ir;        // IR text of the source pattern
    std::shared_ptr<ir::Function> source_function;

    // The replacement pattern — what to replace with
    std::string replacement_ir;   // IR text of the replacement
    std::shared_ptr<ir::Function> replacement_function;

    // Architecture where this pattern was discovered
    ArchDescriptor discovered_arch;

    // Effectiveness metrics
    double avg_speedup = 1.0;     // Average speedup factor
    size_t application_count = 0; // Times successfully applied
    size_t verification_count = 0;// Times verified by SMT

    // Pattern scope
    enum class Scope {
        InstructionLevel,         // Single instruction replacement
        BlockLevel,              // Basic block transformation
        FunctionLevel,           // Cross-block optimisation
        KernelLevel              // GPU kernel optimisation
    };
    Scope scope = Scope::InstructionLevel;

    // Tags for search/matching
    std::vector<std::string> tags;
};

// ── Pattern match result ────────────────────────────────────────────────
struct PatternMatch {
    std::string pattern_id;
    std::string block_name;
    size_t instruction_start;
    size_t instruction_end;
    double confidence;           // 0.0 to 1.0
    double estimated_speedup;
};

// ── Pattern Library ─────────────────────────────────────────────────────
class PatternLibrary final {
public:
    PatternLibrary();

    // Load/store the library from/to disk
    bool load(const std::string& path);
    bool save(const std::string& path) const;

    // Add a pattern
    void add_pattern(const OptimisationPattern& pattern);

    // Remove a pattern
    bool remove_pattern(const std::string& pattern_id);

    // Look up patterns by tag or name
    std::vector<const OptimisationPattern*> find_by_tag(const std::string& tag) const;
    std::vector<const OptimisationPattern*> find_by_name(const std::string& name) const;

    // Match patterns against a function
    std::vector<PatternMatch> match(const ir::Function& fn,
                                     const ArchDescriptor& target_arch) const;

    // Apply a matched pattern, returning the optimised function
    std::shared_ptr<ir::Function> apply(
        const ir::Function& fn,
        const PatternMatch& match,
        const ArchDescriptor& target_arch) const;

    // Architecture-aware pattern adaptation
    // Adapts a pattern discovered on one arch for use on another
    std::optional<OptimisationPattern> adapt_pattern(
        const OptimisationPattern& pattern,
        const ArchDescriptor& target_arch) const;

    // Get all patterns
    const std::unordered_map<std::string, OptimisationPattern>& patterns() const { return patterns_; }

    // Statistics
    size_t size() const { return patterns_.size(); }
    void record_application(const std::string& pattern_id);
    void record_verification(const std::string& pattern_id);

    // Seed the library with built-in patterns
    void seed_builtin_patterns();

private:
    // Pattern matching engine
    bool match_instruction_sequence(
        const ir::BasicBlock& block,
        size_t start,
        const ir::Function& pattern_fn,
        double& confidence) const;

    // Architecture compatibility check
    bool is_compatible(const OptimisationPattern& pattern,
                       const ArchDescriptor& target) const;

    // Helpers for maintaining tag_index_
    void index_pattern_tags(const std::string& pattern_id, const std::vector<std::string>& tags);
    void rebuild_tag_index();

    std::unordered_map<std::string, OptimisationPattern> patterns_;
    // Tag index: maps tag -> list of pattern ids that have that tag.
    // Maintained by add_pattern / remove_pattern for O(1) lookup in find_by_tag.
    std::unordered_map<std::string, std::vector<std::string>> tag_index_;
    std::string library_path_;

    // A single PatternLibrary instance is shared (read-write) across all
    // of Pipeline's worker threads: apply_patterns() runs once per
    // function, on every thread, against the SAME pattern_lib_ member.
    // match()/apply() only read patterns_, but record_application() and
    // record_verification() mutate an existing entry's counters — without
    // a lock that's a data race (concurrent non-atomic increments), so
    // guard just those two writers. match()/apply() stay lock-free since
    // patterns_ is not otherwise mutated once the library is loaded/seeded
    // (before any worker threads are spawned).
    mutable std::mutex stats_mutex_;
};

} // namespace clunk::pattern
