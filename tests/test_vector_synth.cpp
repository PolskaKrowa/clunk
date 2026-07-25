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
 * Clunk Vector-Intrinsic Synthesis Tests.
 *
 * Covers the full vector stack added for the VectorSynth feature:
 *   - VectorType / ConstantVector IR support and parser round-trip
 *   - ir::Scalarizer lane-blasting (validated against the Interpreter)
 *   - SMTVerifier's scalarize-and-recurse hook for vector functions
 *   - VectorSynthesizer rewrites (lane fusion, reduction synthesis,
 *     shuffle algebra) and their cost/SMT gates
 *   - Pipeline integration (vector_phase)
 */
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "clunk/IR/Type.h"
#include "clunk/IR/Value.h"
#include "clunk/IR/Instruction.h"
#include "clunk/IR/BasicBlock.h"
#include "clunk/IR/Function.h"
#include "clunk/IR/Module.h"
#include "clunk/IR/Scalarizer.h"
#include "clunk/Parser/IRParser.h"
#include "clunk/Evaluator/Interpreter.h"
#include "clunk/Evaluator/EvaluationEngine.h"
#include "clunk/Search/SMTVerifier.h"
#include "clunk/Search/VectorSynth.h"
#include "clunk/Pipeline.h"

static int g_pass = 0, g_fail = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { std::cerr << "FAIL: " << msg << " at " << __FILE__ << ":" << __LINE__ << "\n"; g_fail++; } \
    else { g_pass++; } \
} while(0)

using namespace clunk;
using namespace clunk::ir;
using namespace clunk::search;

// ═══════════════════════════════════════════════════════════════════════════
//  IR sources
// ═══════════════════════════════════════════════════════════════════════════

// Fully scalarised dot product over two <4 x i32> arguments — the flagship
// input: 8 extracts + 4 muls + 3 adds that should become mul + reduce.add.
static const char* kDot4 = R"(
define i32 @dot4(<4 x i32> %a, <4 x i32> %b) {
entry:
  %a0 = extractelement <4 x i32> %a, i32 0
  %a1 = extractelement <4 x i32> %a, i32 1
  %a2 = extractelement <4 x i32> %a, i32 2
  %a3 = extractelement <4 x i32> %a, i32 3
  %b0 = extractelement <4 x i32> %b, i32 0
  %b1 = extractelement <4 x i32> %b, i32 1
  %b2 = extractelement <4 x i32> %b, i32 2
  %b3 = extractelement <4 x i32> %b, i32 3
  %m0 = mul i32 %a0, %b0
  %m1 = mul i32 %a1, %b1
  %m2 = mul i32 %a2, %b2
  %m3 = mul i32 %a3, %b3
  %s0 = add i32 %m0, %m1
  %s1 = add i32 %s0, %m2
  %s2 = add i32 %s1, %m3
  ret i32 %s2
}
)";

// Horizontal add over one vector (pure reduction, no fusion needed).
static const char* kHadd4 = R"(
define i32 @hadd4(<4 x i32> %v) {
entry:
  %e0 = extractelement <4 x i32> %v, i32 0
  %e1 = extractelement <4 x i32> %v, i32 1
  %e2 = extractelement <4 x i32> %v, i32 2
  %e3 = extractelement <4 x i32> %v, i32 3
  %t0 = add i32 %e0, %e1
  %t1 = add i32 %t0, %e2
  %t2 = add i32 %t1, %e3
  ret i32 %t2
}
)";

// Vector-returning: shuffle-of-shuffle that composes to one shuffle.
static const char* kShufShuf = R"(
define <4 x i32> @shufshuf(<4 x i32> %a, <4 x i32> %b) {
entry:
  %s1 = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 0, i32 1, i32 4, i32 5>
  %s2 = shufflevector <4 x i32> %s1, <4 x i32> %s1, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
  ret <4 x i32> %s2
}
)";

// Vector-returning lane-wise add (scalarize_lane target).
static const char* kVAdd = R"(
define <4 x i32> @vadd(<4 x i32> %a, <4 x i32> %b) {
entry:
  %r = add <4 x i32> %a, %b
  ret <4 x i32> %r
}
)";

// Partial lane coverage — must NOT be rewritten into a reduction.
static const char* kPartial = R"(
define i32 @partial(<4 x i32> %v) {
entry:
  %e0 = extractelement <4 x i32> %v, i32 0
  %e1 = extractelement <4 x i32> %v, i32 1
  %e2 = extractelement <4 x i32> %v, i32 2
  %t0 = add i32 %e0, %e1
  %t1 = add i32 %t0, %e2
  ret i32 %t1
}
)";

static std::shared_ptr<Module> parse(const char* src) {
    parser::IRParser p;
    return p.parse_string(src);
}

// ═══════════════════════════════════════════════════════════════════════════
//  1. Type system + parser round-trip
// ═══════════════════════════════════════════════════════════════════════════

static void test_vector_type() {
    TypeContext ctx;
    auto v4i32 = ctx.get_vector(ctx.int32(), 4);
    CHECK(v4i32->is_vector(), "get_vector yields a vector type");
    CHECK(v4i32->to_string() == "<4 x i32>", "vector type prints LLVM syntax");
    CHECK(v4i32->count() == 4, "lane count");
    CHECK(v4i32->bit_width() == 128, "vector bit width = lanes * elem");
    CHECK(ctx.get_vector(ctx.int32(), 4) == v4i32, "TypeContext caches vector types");
    auto v8i16 = ctx.get_vector(ctx.int16(), 8);
    CHECK(!(*v4i32 == *v8i16), "different vector types compare unequal");
    CHECK(*v4i32 == *std::make_shared<VectorType>(4, ctx.int32()),
          "structural equality across instances");

    auto cv = ConstantVector::get_int_lanes(ctx, {1, 2, 3, 4}, 32);
    CHECK(cv->lane_count() == 4, "constant vector lanes");
    CHECK(cv->to_string() == "<i32 1, i32 2, i32 3, i32 4>",
          "constant vector prints LLVM syntax");
}

static void test_parser_roundtrip() {
    auto mod = parse(kDot4);
    auto fn = mod ? mod->function("dot4") : nullptr;
    CHECK(fn != nullptr, "dot4 parses");
    if (!fn) return;
    CHECK(fn->argument_count() == 2, "dot4 has 2 args");
    CHECK(fn->arguments()[0].type->is_vector(), "arg 0 parsed as vector type");
    CHECK(fn->arguments()[0].type->to_string() == "<4 x i32>", "arg 0 type text");
    CHECK(fn->entry_block()->size() == 16, "all 16 instructions parsed");
    CHECK(fn->entry_block()->instruction(0)->opcode() == Opcode::ExtractElement,
          "extractelement parsed as a real instruction");

    // Print → reparse → print must be stable (syntactic round-trip).
    std::string printed = fn->to_string();
    auto mod2 = parse(printed.c_str());
    auto fn2 = mod2 ? mod2->function("dot4") : nullptr;
    CHECK(fn2 != nullptr, "printed dot4 reparses");
    if (fn2) CHECK(fn2->to_string() == printed, "print/reparse is stable");

    // Shuffle masks (constant vectors) round-trip too.
    auto smod = parse(kShufShuf);
    auto sfn = smod ? smod->function("shufshuf") : nullptr;
    CHECK(sfn != nullptr, "shufshuf parses");
    if (sfn) {
        CHECK(sfn->entry_block()->instruction(0)->opcode() == Opcode::ShuffleVector,
              "shufflevector parsed");
        auto mask = std::dynamic_pointer_cast<ConstantVector>(
            sfn->entry_block()->instruction(0)->operand(2));
        CHECK(mask != nullptr, "shuffle mask parsed as ConstantVector");
        auto smod2 = parse(sfn->to_string().c_str());
        auto sfn2 = smod2 ? smod2->function("shufshuf") : nullptr;
        CHECK(sfn2 && sfn2->to_string() == sfn->to_string(),
              "shuffle round-trip stable");
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  2. Scalarizer (validated against the Interpreter)
// ═══════════════════════════════════════════════════════════════════════════

static void test_scalarizer_dot4() {
    auto mod = parse(kDot4);
    auto fn = mod->function("dot4");
    CHECK(function_has_vector_ops(*fn), "dot4 has vector ops");

    auto sc = scalarize_function(*fn);
    CHECK(sc != nullptr, "dot4 scalarizes");
    if (!sc) return;
    CHECK(sc->argument_count() == 8, "vector args expand to 8 lanes");
    CHECK(!function_has_vector_ops(*sc), "scalarized dot4 is vector-free");

    // dot([1,2,3,4],[5,6,7,8]) = 5 + 12 + 21 + 32 = 70
    auto r = evaluator::Interpreter::interpret(*sc, {1, 2, 3, 4, 5, 6, 7, 8});
    CHECK(r.has_value() && *r == 70, "scalarized dot4 computes the dot product");
}

static void test_scalarizer_lane() {
    auto mod = parse(kVAdd);
    auto fn = mod->function("vadd");
    CHECK(scalarize_function(*fn) == nullptr,
          "scalarize_function refuses vector-returning fn");
    for (size_t lane = 0; lane < 4; ++lane) {
        auto sc = scalarize_lane(*fn, lane);
        CHECK(sc != nullptr, "vadd lane scalarizes");
        if (!sc) continue;
        auto r = evaluator::Interpreter::interpret(*sc, {10, 20, 30, 40, 1, 2, 3, 4});
        CHECK(r.has_value() && *r == static_cast<int64_t>(11 * (lane + 1)),
              "vadd lane value correct");
    }
    CHECK(scalarize_lane(*fn, 4) == nullptr, "out-of-range lane refused");
}

static void test_scalarizer_reduce_intrinsic() {
    // Hand-build: i32 @f(<4 x i32> %v) { %r = call @clunk.vector.reduce.add.v4i32(%v); ret %r }
    Module mod("m");
    TypeContext& ctx = mod.type_context();
    auto v4 = ctx.get_vector(ctx.int32(), 4);
    auto fnty = std::make_shared<FunctionType>(
        ctx.int32(), std::vector<std::shared_ptr<Type>>{v4});
    auto& fn = mod.add_function("f", fnty);
    fn.add_argument(v4, "v");
    auto& bb = fn.add_block("entry");
    auto v = std::make_shared<Value>(v4, "v");
    auto call = inst::make_call(ctx.int32(), "clunk.vector.reduce.add.v4i32", {v}, "r");
    bb.add_instruction(call);
    bb.add_instruction(inst::make_ret(call));

    Opcode op;
    CHECK(parse_reduce_intrinsic("clunk.vector.reduce.add.v4i32", &op) &&
              op == Opcode::Add,
          "reduce intrinsic name parses");
    CHECK(!parse_reduce_intrinsic("clunk.vector.reduce.smax.v4i32", nullptr),
          "unknown reduce op rejected");

    auto sc = scalarize_function(*mod.function("f"));
    CHECK(sc != nullptr, "reduce intrinsic scalarizes");
    if (sc) {
        auto r = evaluator::Interpreter::interpret(*sc, {3, 5, 7, 11});
        CHECK(r.has_value() && *r == 26, "reduce.add tree sums the lanes");
    }
}

static void test_scalarizer_refusals() {
    // Non-constant extract index must be refused.
    auto mod = parse(R"(
define i32 @dyn(<4 x i32> %v, i32 %i) {
entry:
  %e = extractelement <4 x i32> %v, i32 %i
  ret i32 %e
}
)");
    auto fn = mod->function("dyn");
    CHECK(fn && scalarize_function(*fn) == nullptr,
          "non-constant lane index refused");
}

// ═══════════════════════════════════════════════════════════════════════════
//  3. SMT verification of vector functions (scalarize-and-recurse hook)
// ═══════════════════════════════════════════════════════════════════════════

static void test_smt_vector_hook() {
    if (!SMTVerifier::is_z3_available()) {
        std::cerr << "  (Z3 unavailable — SMT hook tests skipped)\n";
        return;
    }
    auto morig = parse(kHadd4);
    auto orig = morig->function("hadd4");

    // Equivalent rewrite: the reduce intrinsic.
    auto mgood = parse(R"(
define i32 @hadd4(<4 x i32> %v) {
entry:
  %t2 = call i32 @clunk.vector.reduce.add.v4i32(<4 x i32> %v)
  ret i32 %t2
}
)");
    auto good = mgood->function("hadd4");
    SMTVerifier verifier;
    auto res = verifier.verify(*orig, *good);
    CHECK(res.status == VerificationResult::Equivalent,
          "vector reduce rewrite proven equivalent (got: " + res.message + ")");

    // Wrong rewrite (mul instead of add) must be refuted.
    auto mbad = parse(R"(
define i32 @hadd4(<4 x i32> %v) {
entry:
  %t2 = call i32 @clunk.vector.reduce.mul.v4i32(<4 x i32> %v)
  ret i32 %t2
}
)");
    auto bad = mbad->function("hadd4");
    auto res2 = verifier.verify(*orig, *bad);
    CHECK(res2.status == VerificationResult::NotEquivalent,
          "wrong vector rewrite refuted");

    // Vector-returning pair: per-lane equivalence.
    auto ma = parse(kShufShuf);
    auto mb = parse(R"(
define <4 x i32> @shufshuf(<4 x i32> %a, <4 x i32> %b) {
entry:
  %s2 = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 5, i32 4, i32 1, i32 0>
  ret <4 x i32> %s2
}
)");
    auto res3 = verifier.verify(*ma->function("shufshuf"), *mb->function("shufshuf"));
    CHECK(res3.status == VerificationResult::Equivalent,
          "composed shuffle proven equivalent lane-wise (got: " + res3.message + ")");
}

// ═══════════════════════════════════════════════════════════════════════════
//  4. VectorSynthesizer rewrites
// ═══════════════════════════════════════════════════════════════════════════

static bool has_reduce_call(const Function& fn, const std::string& op) {
    for (auto& bb : fn.blocks()) {
        for (auto& inst : bb->instructions()) {
            if (inst->opcode() != Opcode::Call) continue;
            auto it = inst->metadata().find("callee");
            if (it != inst->metadata().end() &&
                it->second.rfind("clunk.vector.reduce." + op, 0) == 0) {
                return true;
            }
        }
    }
    return false;
}

static size_t count_opcode(const Function& fn, Opcode op) {
    size_t n = 0;
    for (auto& bb : fn.blocks()) {
        for (auto& inst : bb->instructions()) {
            if (inst->opcode() == op) ++n;
        }
    }
    return n;
}

static void test_synth_dot4() {
    auto mod = parse(kDot4);
    auto fn = mod->function("dot4");
    evaluator::EvaluationEngine engine;
    VectorSynthConfig cfg;
    cfg.trust_unverified = !SMTVerifier::is_z3_available();
    VectorSynthesizer synth(&engine, cfg);

    bool proven = false;
    auto rewritten = synth.synthesize(*fn, &proven);
    CHECK(rewritten != nullptr, "dot4 synthesises a vector-intrinsic form");
    if (!rewritten) return;

    CHECK(count_opcode(*rewritten, Opcode::Mul) == 1,
          "4 scalar muls fused into 1 vector mul");
    CHECK(rewritten->entry_block()->instruction(0)->type()->is_vector(),
          "fused mul is vector-typed");
    CHECK(has_reduce_call(*rewritten, "add"), "add tree became reduce.add");
    CHECK(count_opcode(*rewritten, Opcode::Add) == 0, "scalar adds are gone");
    CHECK(count_opcode(*rewritten, Opcode::ExtractElement) == 0,
          "all extracts died in DCE");
    CHECK(rewritten->instruction_count() == 3,
          "dot4 collapses to mul + reduce + ret");
    if (SMTVerifier::is_z3_available()) {
        CHECK(proven, "dot4 rewrite carries an SMT proof");
    }
    CHECK(engine.score_candidate(*fn, *rewritten) > 1.0,
          "rewrite scores strictly cheaper");

    // Semantics: scalarized rewrite must equal scalarized original.
    auto so = scalarize_function(*fn);
    auto sr = scalarize_function(*rewritten);
    CHECK(so && sr, "both forms scalarize");
    if (so && sr) {
        std::vector<int64_t> args = {2, -3, 4, 5, 6, 7, -8, 9};
        auto r0 = evaluator::Interpreter::interpret(*so, args);
        auto r1 = evaluator::Interpreter::interpret(*sr, args);
        CHECK(r0 && r1 && *r0 == *r1, "rewrite agrees on a concrete input");
    }
}

static void test_synth_hadd4() {
    auto mod = parse(kHadd4);
    auto fn = mod->function("hadd4");
    evaluator::EvaluationEngine engine;
    VectorSynthConfig cfg;
    cfg.trust_unverified = !SMTVerifier::is_z3_available();
    VectorSynthesizer synth(&engine, cfg);

    auto rewritten = synth.synthesize(*fn, nullptr);
    CHECK(rewritten != nullptr, "hadd4 synthesises");
    if (rewritten) {
        CHECK(has_reduce_call(*rewritten, "add"), "hadd4 uses reduce.add");
        CHECK(rewritten->instruction_count() == 2, "hadd4 is reduce + ret");
    }
}

static void test_synth_shuffle() {
    auto mod = parse(kShufShuf);
    auto fn = mod->function("shufshuf");
    evaluator::EvaluationEngine engine;
    VectorSynthConfig cfg;
    cfg.trust_unverified = !SMTVerifier::is_z3_available();
    VectorSynthesizer synth(&engine, cfg);

    auto rewritten = synth.synthesize(*fn, nullptr);
    CHECK(rewritten != nullptr, "shuffle chain synthesises");
    if (rewritten) {
        CHECK(count_opcode(*rewritten, Opcode::ShuffleVector) == 1,
              "two shuffles composed into one");
    }
}

static void test_synth_negative() {
    // Partial lane coverage: no rewrite may fire.
    auto mod = parse(kPartial);
    auto fn = mod->function("partial");
    evaluator::EvaluationEngine engine;
    VectorSynthConfig cfg;
    cfg.trust_unverified = true; // even unverified, nothing should match
    VectorSynthesizer synth(&engine, cfg);
    CHECK(synth.synthesize(*fn, nullptr) == nullptr,
          "partial lane coverage is not rewritten");

    // Vector-free function: pass is inert.
    auto mod2 = parse(R"(
define i32 @scalar(i32 %x) {
entry:
  %r = add i32 %x, 1
  ret i32 %r
}
)");
    CHECK(synth.synthesize(*mod2->function("scalar"), nullptr) == nullptr,
          "vector-free function is untouched");
}

// ═══════════════════════════════════════════════════════════════════════════
//  5. Pipeline integration
// ═══════════════════════════════════════════════════════════════════════════

static void test_pipeline_integration() {
    auto mod = parse(kDot4);

    PipelineConfig cfg;
    cfg.opt_level = 2;
    cfg.time_budget = 8.0;
    cfg.max_time_per_function = 6.0;
    cfg.max_rounds = 1;   // the vector phase fires in round 1
    cfg.num_threads = 1;
    cfg.verbose = false;

    Pipeline pipeline(cfg);
    auto result = pipeline.run(*mod);
    auto it = result.function_results.find("dot4");
    CHECK(it != result.function_results.end(), "pipeline processed dot4");
    if (it == result.function_results.end()) return;

    if (SMTVerifier::is_z3_available()) {
        CHECK(it->second.improvement_ratio > 1.0,
              "pipeline improved dot4 via vector synthesis");
        CHECK(has_reduce_call(*it->second.optimised, "add"),
              "optimised dot4 contains the reduce.add intrinsic");
        CHECK(it->second.verified, "adopted vector rewrite is SMT-verified");
    } else {
        std::cerr << "  (Z3 unavailable — pipeline adoption checks skipped)\n";
    }
}

// ═══════════════════════════════════════════════════════════════════════════

int main() {
    std::cerr << "test_vector_synth: vector-intrinsic synthesis\n";
    test_vector_type();
    test_parser_roundtrip();
    test_scalarizer_dot4();
    test_scalarizer_lane();
    test_scalarizer_reduce_intrinsic();
    test_scalarizer_refusals();
    test_smt_vector_hook();
    test_synth_dot4();
    test_synth_hadd4();
    test_synth_shuffle();
    test_synth_negative();
    test_pipeline_integration();
    std::cerr << "passed " << g_pass << ", failed " << g_fail << "\n";
    return g_fail == 0 ? 0 : 1;
}
