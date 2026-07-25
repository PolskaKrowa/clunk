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
 * Clunk FP Tests — ConstantFP parsing and the IEEE-exact floating-point
 * rewrites (identity folds + fdiv-by-2^k strength reduction).
 *
 * FP functions cannot be SMT-verified (the verifier's sound_float_fallback
 * refuses them) and cannot be interpreted (integer-only oracle), so every
 * FP rewrite must be sound by construction. These tests pin down exactly
 * which rewrites the search is allowed to make — and, as importantly,
 * which it is NOT (sign-of-zero traps like x + +0.0).
 */
#include <iostream>
#include <memory>
#include <string>

#include "clunk/IR/Function.h"
#include "clunk/IR/Instruction.h"
#include "clunk/IR/Module.h"
#include "clunk/IR/Value.h"
#include "clunk/Parser/IRParser.h"
#include "clunk/Pipeline.h"

static int g_pass = 0, g_fail = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { std::cerr << "FAIL: " << msg << " at " << __FILE__ << ":" << __LINE__ << "\n"; g_fail++; } \
    else { g_pass++; } \
} while(0)

using namespace clunk;
using namespace clunk::ir;

static std::shared_ptr<Module> parse(const char* src) {
    parser::IRParser p;
    return p.parse_string(src);
}

static size_t count_opcode(const Function& fn, Opcode op) {
    size_t n = 0;
    for (auto& bb : fn.blocks()) {
        for (auto& inst : bb->instructions()) {
            if (inst && inst->opcode() == op) ++n;
        }
    }
    return n;
}

// Run one function through the pipeline with a small budget.
static std::shared_ptr<Function> optimise(std::shared_ptr<Module> mod,
                                          const std::string& name) {
    PipelineConfig cfg;
    cfg.opt_level = 2;
    cfg.time_budget = 8.0;
    cfg.max_time_per_function = 6.0;
    cfg.max_rounds = 3;
    cfg.num_threads = 1;
    Pipeline pipeline(cfg);
    auto result = pipeline.run(*mod);
    auto it = result.function_results.find(name);
    return it == result.function_results.end() ? nullptr : it->second.optimised;
}

static void test_constantfp_parsing() {
    auto mod = parse(R"(
define double @f(double %x) {
entry:
  %a = fmul double %x, 1.0
  %b = fadd double %a, 0x4010000000000000
  ret double %b
}
)");
    auto fn = mod->function("f");
    CHECK(fn != nullptr, "FP fixture parses");
    if (!fn) return;
    auto mul = fn->entry_block()->instruction(0);
    auto c1 = std::dynamic_pointer_cast<ConstantFP>(mul->operand(1));
    CHECK(c1 != nullptr, "1.0 parses as ConstantFP (was ConstantInt before the fix)");
    if (c1) CHECK(c1->value() == 1.0, "decimal FP value correct");
    auto add = fn->entry_block()->instruction(1);
    auto c2 = std::dynamic_pointer_cast<ConstantFP>(add->operand(1));
    CHECK(c2 != nullptr, "hex FP literal parses as ConstantFP");
    if (c2) CHECK(c2->value() == 4.0, "hex literal 0x4010... decodes to 4.0 (raw IEEE bits)");
}

static void test_fp_identity_folds() {
    // fmul x, 1.0 and fdiv x, 1.0 fold away entirely.
    auto mod = parse(R"(
define double @mul_one(double %x) {
entry:
  %a = fmul double %x, 1.0
  %b = fdiv double %a, 1.0
  %c = fsub double %b, 0.0
  ret double %c
}
)");
    auto opt = optimise(mod, "mul_one");
    CHECK(opt != nullptr, "pipeline processed mul_one");
    if (!opt) return;
    CHECK(opt->instruction_count() == 1, "all three FP identities folded to `ret %x`");
}

static void test_fdiv_strength_reduction() {
    auto mod = parse(R"(
define double @div4(double %x) {
entry:
  %r = fdiv double %x, 4.0
  ret double %r
}
)");
    auto opt = optimise(mod, "div4");
    CHECK(opt != nullptr, "pipeline processed div4");
    if (!opt) return;
    CHECK(count_opcode(*opt, Opcode::FDiv) == 0, "fdiv by 4.0 is gone");
    CHECK(count_opcode(*opt, Opcode::FMul) == 1, "replaced by fmul");
    // The reciprocal must be exactly 0.25.
    for (auto& inst : opt->entry_block()->instructions()) {
        if (inst->opcode() != Opcode::FMul) continue;
        auto c = std::dynamic_pointer_cast<ConstantFP>(inst->operand(1));
        CHECK(c && c->value() == 0.25, "reciprocal is exactly 0.25");
    }
}

static void test_fp_unsafe_rewrites_blocked() {
    // fadd x, +0.0 is NOT an identity (-0.0 + +0.0 = +0.0 flips the sign
    // of zero) and fdiv x, 3.0 has no exact reciprocal — both must
    // survive optimisation untouched.
    auto mod = parse(R"(
define double @unsafe(double %x) {
entry:
  %a = fadd double %x, 0.0
  %b = fdiv double %a, 3.0
  ret double %b
}
)");
    auto opt = optimise(mod, "unsafe");
    CHECK(opt != nullptr, "pipeline processed unsafe");
    if (!opt) return;
    CHECK(count_opcode(*opt, Opcode::FAdd) == 1, "fadd x, +0.0 NOT folded (zero sign)");
    CHECK(count_opcode(*opt, Opcode::FDiv) == 1, "fdiv x, 3.0 NOT strength-reduced (inexact)");
}

int main() {
    std::cerr << "test_fp_opt: ConstantFP parsing + exact FP rewrites\n";
    test_constantfp_parsing();
    test_fp_identity_folds();
    test_fdiv_strength_reduction();
    test_fp_unsafe_rewrites_blocked();
    std::cerr << "passed " << g_pass << ", failed " << g_fail << "\n";
    return g_fail == 0 ? 0 : 1;
}
