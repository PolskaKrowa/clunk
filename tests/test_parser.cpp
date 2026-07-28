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
 * Clunk Parser Tests — test the IR parser with various LLVM IR strings.
 */
#include <iostream>
#include <memory>
#include <string>
#include <sstream>

#include "clunk/IR/Type.h"
#include "clunk/IR/Value.h"
#include "clunk/IR/Instruction.h"
#include "clunk/IR/BasicBlock.h"
#include "clunk/IR/Function.h"
#include "clunk/IR/Module.h"
#include "clunk/Parser/IRParser.h"

static int g_pass = 0, g_fail = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { std::cerr << "FAIL: " << msg << " at " << __FILE__ << ":" << __LINE__ << "\n"; g_fail++; } \
    else { g_pass++; } \
} while(0)

using namespace clunk::ir;
using namespace clunk::parser;

// ═══════════════════════════════════════════════════════════════════════════

void test_parse_simple_function() {
    IRParser parser;
    const char* ir = R"(
        define i32 @add(i32 %a, i32 %b) {
        entry:
          %r = add i32 %a, %b
          ret i32 %r
        }
    )";
    auto mod = parser.parse_string(ir);
    CHECK(mod != nullptr, "parse_string returned non-null");
    if (!mod) return;
    CHECK(mod->function_count() == 1, "module has 1 function");
    auto fn = mod->function("add");
    CHECK(fn != nullptr, "function 'add' found");
    if (!fn) return;
    CHECK(fn->argument_count() == 2, "function has 2 arguments");
    CHECK(fn->blocks().size() == 1, "function has 1 block");
    CHECK(fn->instruction_count() >= 2, "function has >= 2 instructions");
    CHECK(fn->entry_block() != nullptr, "function has entry block");
    CHECK(fn->entry_block()->name() == "entry", "entry block name");

    // Check the instructions
    auto& instrs = fn->entry_block()->instructions();
    CHECK(instrs.size() == 2, "entry has 2 instructions");
    CHECK(instrs[0]->opcode() == Opcode::Add, "first instruction is add");
    CHECK(instrs[1]->opcode() == Opcode::Ret, "second instruction is ret");
}

void test_parse_multiple_blocks_and_branches() {
    IRParser parser;
    const char* ir = R"(
        define i32 @abs(i32 %x) {
        entry:
          %cond = icmp sge i32 %x, 0
          br i1 %cond, label %pos, label %neg

        pos:
          ret i32 %x

        neg:
          %neg_x = sub i32 0, %x
          ret i32 %neg_x
        }
    )";
    auto mod = parser.parse_string(ir);
    CHECK(mod != nullptr, "parse returned non-null");
    if (!mod) return;
    auto fn = mod->function("abs");
    CHECK(fn != nullptr, "function 'abs' found");
    if (!fn) return;
    CHECK(fn->blocks().size() == 3, "function has 3 blocks");

    // Entry block has icmp + br
    auto entry = fn->block("entry");
    CHECK(entry != nullptr, "entry block found");
    CHECK(entry->size() == 2, "entry has 2 instructions");
    CHECK(entry->instruction(0)->opcode() == Opcode::ICmp, "entry[0] is icmp");
    CHECK(entry->instruction(1)->opcode() == Opcode::Br, "entry[1] is br");

    auto pos = fn->block("pos");
    CHECK(pos != nullptr, "pos block found");
    CHECK(pos->size() == 1, "pos has 1 instruction (ret)");
    CHECK(pos->terminator()->opcode() == Opcode::Ret, "pos terminator is ret");

    auto neg = fn->block("neg");
    CHECK(neg != nullptr, "neg block found");
    CHECK(neg->size() == 2, "neg has 2 instructions (sub + ret)");
}

void test_parse_target_triple() {
    IRParser parser;
    const char* ir = R"(
        target triple = "x86_64-unknown-linux-gnu"
        target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

        define i32 @foo() {
        entry:
          ret i32 0
        }
    )";
    auto mod = parser.parse_string(ir);
    CHECK(mod != nullptr, "parse returned non-null");
    if (!mod) return;
    CHECK(mod->has_target(), "module has target info");
    CHECK(mod->target().triple == "x86_64-unknown-linux-gnu", "target triple matches");
    CHECK(!mod->target().datalayout.empty(), "datalayout is non-empty");
}

void test_parse_global_variables() {
    IRParser parser;
    const char* ir = R"(
        @g_counter = global i32 0
        @g_pi = constant double 3.14159

        define i32 @read_counter() {
        entry:
          ret i32 0
        }
    )";
    auto mod = parser.parse_string(ir);
    CHECK(mod != nullptr, "parse returned non-null");
    if (!mod) return;
    CHECK(mod->globals().size() >= 1, "module has >= 1 global");
    CHECK(mod->function_count() == 1, "module has 1 function");
}

// Regression test: `@name = external global <type>, align N` (a
// declaration — no initializer) must round-trip through
// parse -> to_string() -> parse without becoming invalid IR. This used
// to print as `@name = global <type> , align N` — missing the `external`
// keyword and with a stray leading comma where LLVM expects a value —
// which llvm-as/opt/alive-tv all reject with "expected value token".
void test_external_global_declaration_round_trips() {
    IRParser parser;
    const char* ir = R"(
        @stdin = external dso_local global ptr, align 8

        define ptr @get_stdin() {
        entry:
          %v = load ptr, ptr @stdin, align 8
          ret ptr %v
        }
    )";
    auto mod = parser.parse_string(ir);
    CHECK(mod != nullptr, "parse returned non-null");
    if (!mod) return;

    const auto& globals = mod->globals();
    CHECK(globals.size() == 1, "module has exactly 1 global");
    if (globals.empty()) return;
    CHECK(globals[0].is_declaration, "'external global' must be marked as a declaration");
    CHECK(globals[0].init_value.empty() || globals[0].init_value.front() == ',',
          "a declaration must not have captured a bogus initializer value");

    std::string text = mod->to_string();
    CHECK(text.find("@stdin = external global") != std::string::npos,
          "printed global must include the 'external' keyword (got:\n" + text + ")");
    CHECK(text.find("global ptr ,") == std::string::npos,
          "printed global must NOT have a bare type-then-comma with no value "
          "(the exact 'expected value token' bug)");
    // The load must reference the GLOBAL @stdin, not a phantom local
    // %stdin — see test_global_operand_keeps_at_sigil for the isolated
    // regression this depends on.
    CHECK(text.find("ptr @stdin") != std::string::npos,
          "load operand must reference @stdin, not %stdin (got:\n" + text + ")");

    // And it must re-parse without error — the actual round-trip check.
    IRParser parser2;
    auto reparsed = parser2.parse_string(text);
    CHECK(reparsed != nullptr,
          "printer output for an external global declaration failed to "
          "re-parse:\n" + text);
}

// Regression test: an operand that references a global directly (not
// via a `call`) — e.g. `load ptr, ptr @stdin` — must keep its `@` sigil.
// Value::to_string()/print_as_operand() used to always print "%name"
// for every named Value, so parsing `@stdin` and re-printing it produced
// `%stdin`: a reference to an undefined local SSA value.
void test_global_operand_keeps_at_sigil() {
    IRParser parser;
    const char* ir = R"(
        @g = global i32 0, align 4

        define i32 @read_g() {
        entry:
          %v = load i32, ptr @g, align 4
          ret i32 %v
        }
    )";
    auto mod = parser.parse_string(ir);
    CHECK(mod != nullptr, "parse returned non-null");
    if (!mod) return;
    auto fn = mod->function("read_g");
    CHECK(fn != nullptr, "function 'read_g' found");
    if (!fn) return;
    auto entry = fn->entry_block();
    CHECK(entry != nullptr, "entry block found");
    if (!entry) return;
    CHECK(entry->size() >= 1, "entry block has instructions");
    auto load_inst = entry->instruction(0);
    CHECK(load_inst != nullptr, "load instruction found");
    if (!load_inst || load_inst->num_operands() == 0) return;
    auto ptr_operand = load_inst->operand(load_inst->num_operands() - 1);
    CHECK(ptr_operand->is_global(), "load's pointer operand must be marked is_global()");
    CHECK(ptr_operand->print_as_operand() == "@g",
          "pointer operand must print as '@g', got '" +
          ptr_operand->print_as_operand() + "'");
}

// Regression test: LLVM 18+'s `captures(none)` parameter attribute (the
// replacement for the older standalone `nocapture`) must not desync the
// parameter-list parser. Before "captures" was added to the keyword
// table, the lexer tokenized it as a plain Identifier, so
// collect_param_attrs stopped consuming right before it, and "captures",
// "(", and "none" were each fed to parse_type() as if they were separate
// parameters — which doesn't recognize any of them and silently returns
// void, turning one real `ptr` parameter into `(ptr, void, void, void)`
// and consuming the wrong closing paren for the rest of the signature.
void test_captures_attribute_does_not_corrupt_params() {
    IRParser parser;
    const char* ir = R"(
        define dso_local void @rot13(ptr noundef captures(none) %s) local_unnamed_addr #0 {
        entry:
          ret void
        }
    )";
    auto mod = parser.parse_string(ir);
    CHECK(mod != nullptr, "parse returned non-null");
    if (!mod) return;
    auto fn = mod->function("rot13");
    CHECK(fn != nullptr, "function 'rot13' found");
    if (!fn) return;
    CHECK(fn->argument_count() == 1,
          "rot13 must have exactly 1 parameter, got " +
          std::to_string(fn->argument_count()));
    if (fn->argument_count() >= 1) {
        CHECK(!fn->arguments()[0].type->is_void(),
              "the single parameter must not have been corrupted into void");
        CHECK(fn->arguments()[0].type->is_pointer(),
              "the single parameter should still be a pointer type");
    }
    std::string text = fn->to_string();
    CHECK(text.find("void, void") == std::string::npos,
          "printed signature must not contain cascaded void parameters "
          "(got: " + text.substr(0, 80) + ")");

    // Multi-component form: captures(address, read_provenance)
    IRParser parser2;
    const char* ir2 = R"(
        define ptr @get_env(ptr noundef dereferenceable_or_null(64) captures(address, read_provenance) %s) {
        entry:
          ret ptr %s
        }
    )";
    auto mod2 = parser2.parse_string(ir2);
    CHECK(mod2 != nullptr, "second parse returned non-null");
    if (!mod2) return;
    auto fn2 = mod2->function("get_env");
    CHECK(fn2 != nullptr, "function 'get_env' found");
    if (fn2) {
        CHECK(fn2->argument_count() == 1,
              "get_env must have exactly 1 parameter, got " +
              std::to_string(fn2->argument_count()));
    }
}

void test_parse_void_function() {
    IRParser parser;
    const char* ir = R"(
        define void @noop() {
        entry:
          ret void
        }
    )";
    auto mod = parser.parse_string(ir);
    CHECK(mod != nullptr, "parse returned non-null");
    if (!mod) return;
    auto fn = mod->function("noop");
    CHECK(fn != nullptr, "function 'noop' found");
    if (!fn) return;
    CHECK(fn->return_type()->is_void(), "return type is void");
    CHECK(fn->argument_count() == 0, "no arguments");
}

void test_parse_phi_instruction() {
    IRParser parser;
    const char* ir = R"(
        define i32 @loop_test(i32 %n) {
        entry:
          br label %loop

        loop:
          %i = phi i32 [0, %entry], [%next, %loop]
          %next = add i32 %i, 1
          %cond = icmp slt i32 %next, %n
          br i1 %cond, label %loop, label %exit

        exit:
          ret i32 %i
        }
    )";
    auto mod = parser.parse_string(ir);
    CHECK(mod != nullptr, "parse returned non-null");
    if (!mod) return;
    auto fn = mod->function("loop_test");
    CHECK(fn != nullptr, "function 'loop_test' found");
    if (!fn) return;
    CHECK(fn->blocks().size() == 3, "function has 3 blocks");

    auto loop = fn->block("loop");
    CHECK(loop != nullptr, "loop block found");
    if (!loop) return;
    // Check for phi instruction
    bool has_phi = false;
    for (auto& inst : loop->instructions()) {
        if (inst->opcode() == Opcode::Phi) {
            has_phi = true;
            break;
        }
    }
    CHECK(has_phi, "loop block contains phi instruction");
}

void test_parse_call_instruction() {
    IRParser parser;
    const char* ir = R"(
        define i32 @caller(i32 %x) {
        entry:
          %result = call i32 @callee(i32 %x)
          ret i32 %result
        }

        define i32 @callee(i32 %val) {
        entry:
          ret i32 %val
        }
    )";
    auto mod = parser.parse_string(ir);
    CHECK(mod != nullptr, "parse returned non-null");
    if (!mod) return;
    CHECK(mod->function_count() == 2, "module has 2 functions");

    auto caller = mod->function("caller");
    CHECK(caller != nullptr, "function 'caller' found");
    if (!caller) return;
    auto& instrs = caller->entry_block()->instructions();
    bool has_call = false;
    for (auto& inst : instrs) {
        if (inst->opcode() == Opcode::Call) {
            has_call = true;
            break;
        }
    }
    CHECK(has_call, "caller contains a call instruction");
}

void test_parse_memory_instructions() {
    IRParser parser;
    const char* ir = R"(
        define i32 @memtest() {
        entry:
          %ptr = alloca i32
          store i32 42, i32* %ptr
          %val = load i32, i32* %ptr
          ret i32 %val
        }
    )";
    auto mod = parser.parse_string(ir);
    CHECK(mod != nullptr, "parse returned non-null");
    if (!mod) return;
    auto fn = mod->function("memtest");
    CHECK(fn != nullptr, "function 'memtest' found");
    if (!fn) return;

    auto& instrs = fn->entry_block()->instructions();
    bool has_alloca = false, has_store = false, has_load = false;
    for (auto& inst : instrs) {
        if (inst->opcode() == Opcode::Alloca) has_alloca = true;
        if (inst->opcode() == Opcode::Store) has_store = true;
        if (inst->opcode() == Opcode::Load) has_load = true;
    }
    CHECK(has_alloca, "has alloca");
    CHECK(has_store, "has store");
    CHECK(has_load, "has load");
}

void test_parse_select_instruction() {
    IRParser parser;
    const char* ir = R"(
        define i32 @select_test(i32 %x) {
        entry:
          %cond = icmp sgt i32 %x, 0
          %result = select i1 %cond, i32 %x, i32 0
          ret i32 %result
        }
    )";
    auto mod = parser.parse_string(ir);
    CHECK(mod != nullptr, "parse returned non-null");
    if (!mod) return;
    auto fn = mod->function("select_test");
    CHECK(fn != nullptr, "function 'select_test' found");
    if (!fn) return;

    bool has_select = false;
    for (auto& inst : fn->entry_block()->instructions()) {
        if (inst->opcode() == Opcode::Select) has_select = true;
    }
    CHECK(has_select, "contains select instruction");
}

void test_parse_multiple_functions() {
    IRParser parser;
    const char* ir = R"(
        define i32 @add(i32 %a, i32 %b) {
        entry:
          %r = add i32 %a, %b
          ret i32 %r
        }

        define i32 @sub(i32 %a, i32 %b) {
        entry:
          %r = sub i32 %a, %b
          ret i32 %r
        }

        define i32 @mul(i32 %a, i32 %b) {
        entry:
          %r = mul i32 %a, %b
          ret i32 %r
        }
    )";
    auto mod = parser.parse_string(ir);
    CHECK(mod != nullptr, "parse returned non-null");
    if (!mod) return;
    CHECK(mod->function_count() == 3, "module has 3 functions");
    CHECK(mod->function("add") != nullptr, "add found");
    CHECK(mod->function("sub") != nullptr, "sub found");
    CHECK(mod->function("mul") != nullptr, "mul found");
}

void test_parse_error_handling() {
    // Invalid IR should produce warnings, not crashes
    IRParser parser;
    const char* bad_ir = R"(
        this is not valid IR at all
    )";
    // Should not crash
    auto mod = parser.parse_string(bad_ir);
    // It's OK if mod is null or empty; the key is no crash
    CHECK(true, "parsing invalid IR did not crash");

    // Also test empty input
    IRParser parser2;
    auto mod2 = parser2.parse_string("");
    CHECK(mod2 != nullptr, "empty string gives non-null module");
    CHECK(mod2->function_count() == 0, "empty string gives empty module");
}

void test_parse_file_not_found() {
    IRParser parser;
    try {
        auto mod = parser.parse_file("/nonexistent/path/to/file.ll");
        // Should not crash; may return nullptr or empty module
    } catch (const ParseError&) {
        // Throwing is acceptable for missing files
    }
    CHECK(true, "parsing nonexistent file did not crash");
}

void test_parse_warnings() {
    IRParser parser;
    const char* ir = R"(
        define i32 @simple() {
        entry:
          ret i32 42
        }
    )";
    auto mod = parser.parse_string(ir);
    CHECK(mod != nullptr, "parse returned non-null");
    if (!mod) return;
    // Warnings may or may not be present; just check we can access them
    const auto& warnings = parser.warnings();
    // No assertion on count; just checking the interface works
    CHECK(true, "warnings() accessor works without crash");
}

void test_parse_internal_linkage() {
    IRParser parser;
    const char* ir = R"(
        define internal i32 @helper() {
        entry:
          ret i32 1
        }
    )";
    auto mod = parser.parse_string(ir);
    CHECK(mod != nullptr, "parse returned non-null");
    if (!mod) return;
    auto fn = mod->function("helper");
    CHECK(fn != nullptr, "function 'helper' found");
    if (!fn) return;
    CHECK(fn->linkage() == Linkage::Internal, "function has internal linkage");
}

// ═══════════════════════════════════════════════════════════════════════════

int main() {
    std::cout << "=== Clunk Parser Tests ===" << std::endl;

    std::cout << "  Simple function..." << std::endl;
    test_parse_simple_function();

    std::cout << "  Multiple blocks and branches..." << std::endl;
    test_parse_multiple_blocks_and_branches();

    std::cout << "  Target triple..." << std::endl;
    test_parse_target_triple();

    std::cout << "  Global variables..." << std::endl;
    test_parse_global_variables();

    std::cout << "  External global declaration round-trips..." << std::endl;
    test_external_global_declaration_round_trips();

    std::cout << "  Global operand keeps '@' sigil..." << std::endl;
    test_global_operand_keeps_at_sigil();

    std::cout << "  captures(none)/dereferenceable_or_null don't corrupt params..." << std::endl;
    test_captures_attribute_does_not_corrupt_params();

    std::cout << "  Void function..." << std::endl;
    test_parse_void_function();

    std::cout << "  Phi instruction..." << std::endl;
    test_parse_phi_instruction();

    std::cout << "  Call instruction..." << std::endl;
    test_parse_call_instruction();

    std::cout << "  Memory instructions..." << std::endl;
    test_parse_memory_instructions();

    std::cout << "  Select instruction..." << std::endl;
    test_parse_select_instruction();

    std::cout << "  Multiple functions..." << std::endl;
    test_parse_multiple_functions();

    std::cout << "  Error handling..." << std::endl;
    test_parse_error_handling();

    std::cout << "  File not found..." << std::endl;
    test_parse_file_not_found();

    std::cout << "  Warnings..." << std::endl;
    test_parse_warnings();

    std::cout << "  Internal linkage..." << std::endl;
    test_parse_internal_linkage();

    std::cout << "\n=== Results: " << g_pass << " passed, " << g_fail << " failed ===" << std::endl;
    return g_fail > 0 ? 1 : 0;
}
