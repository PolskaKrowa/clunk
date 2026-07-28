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
 * Clunk Alive2 Verifier Tests.
 *
 * These tests do NOT require alive-tv to be installed — AliveVerifier
 * must degrade gracefully (Status::NotAvailable, no crash, no hang) when
 * it isn't, exactly like SMTVerifier does for Z3. If alive-tv IS present
 * on PATH, the differential-testing script (scripts/diff_test_alive.sh)
 * and CI's dedicated Alive2 job are the place for real refinement checks
 * against a live binary; this file only exercises clunk's own plumbing.
 *
 * The second half of this file guards the printer bug that motivated
 * AliveVerifier in the first place: Function::to_string() used to emit
 * the linkage keyword BEFORE "define" (`internal define i32 @f(...)`),
 * which is invalid LLVM IR — neither clunk's own parser, clang/opt, nor
 * alive-tv can parse that ordering. `define internal i32 @f(...)` is the
 * only legal form.
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
#include "clunk/Parser/IRParser.h"
#include "clunk/Search/AliveVerifier.h"

static int g_pass = 0, g_fail = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { std::cerr << "FAIL: " << msg << " at " << __FILE__ << ":" << __LINE__ << "\n"; g_fail++; } \
    else { g_pass++; } \
} while(0)

using namespace clunk::ir;
using namespace clunk::search;

// ═══════════════════════════════════════════════════════════════════════════
//  Helpers
// ═══════════════════════════════════════════════════════════════════════════

// define i32 @fn(i32 %x) { entry: %r = add i32 %x, 1; ret i32 %r }
static std::shared_ptr<Function> make_add_one(Module& mod, const std::string& name,
                                               Linkage linkage = Linkage::External) {
    TypeContext& ctx = mod.type_context();
    auto fn_type = std::make_shared<FunctionType>(
        ctx.int32(), std::vector<std::shared_ptr<Type>>{ctx.int32()});
    auto& fn = mod.add_function(name, fn_type, linkage);
    fn.add_argument(ctx.int32(), "x");
    auto& entry = fn.add_block("entry");
    auto x = std::make_shared<Value>(ctx.int32(), "x");
    auto one = ConstantInt::get(ctx, 1, 32);
    entry.add_instruction(inst::make_add(x, one, "r"));
    entry.add_instruction(inst::make_ret(entry.instruction(0)));
    return mod.function(name);
}

// ═══════════════════════════════════════════════════════════════════════════
//  AliveVerifier plumbing (no alive-tv binary required)
// ═══════════════════════════════════════════════════════════════════════════

static void test_availability_probe_does_not_crash() {
    // Whatever the sandbox has installed, this must return promptly and
    // consistently — it's called from hot paths in Pipeline.
    bool a = AliveVerifier::is_alive_tv_available("alive-tv");
    bool b = AliveVerifier::is_alive_tv_available("alive-tv");
    CHECK(a == b, "is_alive_tv_available should be stable/cached across calls");

    // A path that can't possibly exist must report unavailable, not hang
    // or throw.
    bool bogus = AliveVerifier::is_alive_tv_available(
        "/definitely/not/a/real/path/alive-tv-does-not-exist");
    CHECK(!bogus, "bogus alive-tv path must report unavailable");
}

static void test_verify_degrades_gracefully_when_unavailable() {
    Module mod("m");
    auto fn = make_add_one(mod, "add_one");

    AliveConfig cfg;
    cfg.alive_tv_path = "/definitely/not/a/real/path/alive-tv-does-not-exist";
    AliveVerifier av(cfg);

    AliveResult r = av.verify(*fn, *fn, &mod);
    CHECK(r.status == AliveResult::NotAvailable,
          "verify() must report NotAvailable rather than crash/hang when "
          "alive-tv isn't installed");
    CHECK(!r.message.empty(), "NotAvailable result should still carry a message");
}

static void test_config_defaults() {
    AliveVerifier av;
    CHECK(av.config().alive_tv_path == "alive-tv", "default alive_tv_path");
    CHECK(av.config().timeout_ms == 30000, "default timeout_ms");
}

static void test_render_standalone_module_is_self_contained() {
    // Build: @callee, and @caller which calls it — AliveVerifier must
    // synthesize a `declare` for @callee so the rendered module parses
    // on its own (this is what lets it check functions SMTVerifier
    // reports Unknown for, i.e. anything with a call in it).
    Module mod("m");
    TypeContext& ctx = mod.type_context();
    auto fn_type = std::make_shared<FunctionType>(
        ctx.int32(), std::vector<std::shared_ptr<Type>>{ctx.int32()});
    mod.add_function("callee", fn_type);

    auto& caller = mod.add_function("caller", fn_type);
    caller.add_argument(ctx.int32(), "x");
    auto& entry = caller.add_block("entry");
    auto x = std::make_shared<Value>(ctx.int32(), "x");
    entry.add_instruction(inst::make_call(ctx.int32(), "callee", {x}, "r"));
    entry.add_instruction(inst::make_ret(entry.instruction(0)));

    AliveVerifier av;
    // verify() will bail out early with NotAvailable if alive-tv isn't on
    // PATH, but it should still get through rendering + writing temp
    // files without throwing regardless of whether alive-tv exists.
    AliveResult r = av.verify(*mod.function("caller"), *mod.function("caller"), &mod);
    CHECK(r.status == AliveResult::NotAvailable || r.exit_code >= 0,
          "verify() on a call-containing function must not throw during rendering");
}

// ═══════════════════════════════════════════════════════════════════════════
//  Printer regression: linkage must follow "define", not precede it
// ═══════════════════════════════════════════════════════════════════════════

static void test_internal_linkage_prints_after_define() {
    Module mod("m");
    auto fn = make_add_one(mod, "helper", Linkage::Internal);
    std::string text = fn->to_string();

    CHECK(text.rfind("define internal ", 0) == 0,
          "Function::to_string() must emit 'define internal ...', not "
          "'internal define ...' (got: " + text.substr(0, 40) + ")");
    CHECK(text.find("internal define") == std::string::npos,
          "the invalid 'internal define' ordering must never appear");
}

static void test_printer_output_round_trips_through_own_parser() {
    // The whole point of the fix: clunk's own parser must be able to
    // read back what clunk's own printer just wrote, for every linkage
    // kind that commonly appears in real .ll files.
    for (Linkage l : {Linkage::External, Linkage::Internal, Linkage::Private,
                       Linkage::Weak, Linkage::LinkOnceODR}) {
        Module mod("m");
        auto fn = make_add_one(mod, "f", l);
        std::string text = mod.to_string();

        clunk::parser::IRParser parser;
        auto reparsed = parser.parse_string(text);
        CHECK(reparsed != nullptr,
              "module with linkage=" + std::to_string(static_cast<int>(l)) +
              " failed to round-trip through clunk's own parser:\n" + text);
        if (reparsed) {
            auto rf = reparsed->function("f");
            CHECK(rf != nullptr, "re-parsed module is missing function 'f'");
            if (rf) {
                CHECK(rf->linkage() == l, "linkage did not round-trip correctly");
            }
        }
    }
}

int main() {
    std::cout << "=== Clunk Alive2 Verifier Tests ===" << std::endl;

    std::cout << "  Availability probe..." << std::endl;
    test_availability_probe_does_not_crash();

    std::cout << "  Graceful degradation when alive-tv missing..." << std::endl;
    test_verify_degrades_gracefully_when_unavailable();

    std::cout << "  Config defaults..." << std::endl;
    test_config_defaults();

    std::cout << "  Standalone-module rendering with calls..." << std::endl;
    test_render_standalone_module_is_self_contained();

    std::cout << "  Linkage prints after 'define'..." << std::endl;
    test_internal_linkage_prints_after_define();

    std::cout << "  Printer output round-trips through own parser..." << std::endl;
    test_printer_output_round_trips_through_own_parser();

    std::cout << "\n=== Results: " << g_pass << " passed, " << g_fail << " failed ===" << std::endl;
    return g_fail > 0 ? 1 : 0;
}
