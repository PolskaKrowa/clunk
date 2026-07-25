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
