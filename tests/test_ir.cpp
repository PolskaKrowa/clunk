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
 * Clunk IR Tests — comprehensive test suite for the IR representation layer.
 * Tests: Type system, Values, Instructions, BasicBlock, Function, Module, IRBuilder.
 */
#include <cassert>
#include <cstdint>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>

#include "clunk/IR/Type.h"
#include "clunk/IR/Value.h"
#include "clunk/IR/Instruction.h"
#include "clunk/IR/BasicBlock.h"
#include "clunk/IR/Function.h"
#include "clunk/IR/Module.h"
#include "clunk/IR/IRBuilder.h"

static int g_pass = 0, g_fail = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { std::cerr << "FAIL: " << msg << " at " << __FILE__ << ":" << __LINE__ << "\n"; g_fail++; } \
    else { g_pass++; } \
} while(0)

using namespace clunk::ir;

// ═══════════════════════════════════════════════════════════════════════════
//  Type System Tests
// ═══════════════════════════════════════════════════════════════════════════

void test_void_type() {
    VoidType vt;
    CHECK(vt.type_id() == TypeID::Void, "VoidType type_id");
    CHECK(vt.is_void(), "VoidType is_void");
    CHECK(!vt.is_integer(), "VoidType not is_integer");
    CHECK(vt.to_string() == "void", "VoidType to_string");
    CHECK(vt == VoidType(), "VoidType equality");

    VoidType vt2;
    CHECK(vt == vt2, "VoidType equality (two instances)");
}

void test_integer_type() {
    auto i1 = IntegerType::i1();
    auto i8 = IntegerType::i8();
    auto i16 = IntegerType::i16();
    auto i32 = IntegerType::i32();
    auto i64 = IntegerType::i64();

    CHECK(i32->type_id() == TypeID::Integer, "IntegerType type_id");
    CHECK(i32->is_integer(), "IntegerType is_integer");
    CHECK(!i32->is_void(), "IntegerType not is_void");

    CHECK(i1->bits() == 1, "i1 bits");
    CHECK(i8->bits() == 8, "i8 bits");
    CHECK(i16->bits() == 16, "i16 bits");
    CHECK(i32->bits() == 32, "i32 bits");
    CHECK(i64->bits() == 64, "i64 bits");

    CHECK(i1->bit_width() == 1, "i1 bit_width");
    CHECK(i32->bit_width() == 32, "i32 bit_width");
    CHECK(i64->bit_width() == 64, "i64 bit_width");

    CHECK(i1->size_bytes() == 1, "i1 size_bytes");
    CHECK(i8->size_bytes() == 1, "i8 size_bytes");
    CHECK(i16->size_bytes() == 2, "i16 size_bytes");
    CHECK(i32->size_bytes() == 4, "i32 size_bytes");
    CHECK(i64->size_bytes() == 8, "i64 size_bytes");

    CHECK(i32->to_string() == "i32", "i32 to_string");
    CHECK(i64->to_string() == "i64", "i64 to_string");

    CHECK(*i32 == *IntegerType::i32(), "i32 equality");
    CHECK(!(*i32 == *i64), "i32 != i64");
    CHECK(*i32 != *i64, "i32 != i64 (operator!=)");
}

void test_float_type() {
    FloatType ft;
    CHECK(ft.type_id() == TypeID::Float, "FloatType type_id");
    CHECK(ft.is_float(), "FloatType is_float");
    CHECK(!ft.is_double(), "FloatType not is_double");
    CHECK(ft.bit_width() == 32, "FloatType bit_width");
    CHECK(ft.size_bytes() == 4, "FloatType size_bytes");
    CHECK(ft.to_string() == "float", "FloatType to_string");
    CHECK(ft == FloatType(), "FloatType equality");
}

void test_double_type() {
    DoubleType dt;
    CHECK(dt.type_id() == TypeID::Double, "DoubleType type_id");
    CHECK(dt.is_double(), "DoubleType is_double");
    CHECK(!dt.is_float(), "DoubleType not is_float");
    CHECK(dt.bit_width() == 64, "DoubleType bit_width");
    CHECK(dt.size_bytes() == 8, "DoubleType size_bytes");
    CHECK(dt.to_string() == "double", "DoubleType to_string");
    CHECK(dt == DoubleType(), "DoubleType equality");
}

void test_pointer_type() {
    auto i32 = IntegerType::i32();
    auto ptr_i32 = std::make_shared<PointerType>(i32);
    CHECK(ptr_i32->type_id() == TypeID::Pointer, "PointerType type_id");
    CHECK(ptr_i32->is_pointer(), "PointerType is_pointer");
    CHECK(ptr_i32->bit_width() == 64, "PointerType bit_width (64-bit)");
    CHECK(ptr_i32->size_bytes() == 8, "PointerType size_bytes");
    CHECK(ptr_i32->pointee() == i32, "PointerType pointee");
    CHECK(ptr_i32->address_space() == 0, "PointerType default address_space");
    CHECK(ptr_i32->to_string() == "i32*", "PointerType to_string");

    // Address space
    auto ptr_as = std::make_shared<PointerType>(i32, 3);
    CHECK(ptr_as->address_space() == 3, "PointerType address_space=3");
    CHECK(ptr_as->to_string() == "i32* addrspace(3)", "PointerType to_string with addrspace");

    // Equality
    auto ptr_i32_2 = std::make_shared<PointerType>(i32);
    CHECK(*ptr_i32 == *ptr_i32_2, "PointerType equality (same pointee)");
    auto ptr_i64 = std::make_shared<PointerType>(IntegerType::i64());
    CHECK(!(*ptr_i32 == *ptr_i64), "PointerType inequality (different pointee)");
}

void test_array_type() {
    auto i32 = IntegerType::i32();
    auto arr = std::make_shared<ArrayType>(10, i32);
    CHECK(arr->type_id() == TypeID::Array, "ArrayType type_id");
    CHECK(arr->is_array(), "ArrayType is_array");
    CHECK(arr->count() == 10, "ArrayType count");
    CHECK(arr->element_type() == i32, "ArrayType element_type");
    CHECK(arr->size_bytes() == 40, "ArrayType size_bytes (10 * 4)");
    CHECK(arr->to_string() == "[10 x i32]", "ArrayType to_string");

    auto arr2 = std::make_shared<ArrayType>(10, i32);
    CHECK(*arr == *arr2, "ArrayType equality");
    auto arr3 = std::make_shared<ArrayType>(5, i32);
    CHECK(!(*arr == *arr3), "ArrayType inequality (different count)");
}

void test_struct_type() {
    auto i32 = IntegerType::i32();
    auto f64 = std::make_shared<DoubleType>();

    // Named struct
    auto named = std::make_shared<StructType>(
        std::vector<std::shared_ptr<Type>>{i32, f64}, false, "Point");
    CHECK(named->type_id() == TypeID::Struct, "StructType type_id");
    CHECK(named->is_struct(), "StructType is_struct");
    CHECK(named->field_count() == 2, "StructType field_count");
    CHECK(named->name() == "Point", "StructType name");
    CHECK(named->size_bytes() == 12, "StructType size_bytes (4+8)");
    CHECK(named->to_string() == "%Point", "Named StructType to_string");

    // Anonymous struct
    auto anon = std::make_shared<StructType>(
        std::vector<std::shared_ptr<Type>>{i32, f64});
    CHECK(anon->name().empty(), "Anonymous StructType has no name");
    CHECK(anon->to_string() == "{ i32, double }", "Anonymous StructType to_string");

    // Packed struct
    auto packed = std::make_shared<StructType>(
        std::vector<std::shared_ptr<Type>>{i32, f64}, true);
    CHECK(packed->is_packed(), "Packed StructType");
    CHECK(packed->to_string() == "<{ i32, double }>", "Packed StructType to_string");

    // Equality
    auto anon2 = std::make_shared<StructType>(
        std::vector<std::shared_ptr<Type>>{i32, f64});
    CHECK(*anon == *anon2, "StructType equality");
}

void test_function_type() {
    auto i32 = IntegerType::i32();
    auto ftype = std::make_shared<FunctionType>(i32, std::vector<std::shared_ptr<Type>>{i32, i32});
    CHECK(ftype->type_id() == TypeID::Function, "FunctionType type_id");
    CHECK(ftype->is_function(), "FunctionType is_function");
    CHECK(ftype->return_type() == i32, "FunctionType return_type");
    CHECK(ftype->params().size() == 2, "FunctionType params size");
    CHECK(!ftype->is_vararg(), "FunctionType not vararg");
    CHECK(ftype->to_string() == "i32 (i32, i32)", "FunctionType to_string");

    // Vararg
    auto vtype = std::make_shared<FunctionType>(i32, std::vector<std::shared_ptr<Type>>{i32}, true);
    CHECK(vtype->is_vararg(), "FunctionType is_vararg");
    CHECK(vtype->to_string() == "i32 (i32, ...)", "FunctionType to_string vararg");
}

void test_type_context() {
    TypeContext ctx;
    auto i32_a = ctx.int32();
    auto i32_b = ctx.int32();
    CHECK(i32_a == i32_b, "TypeContext caches int32 (same pointer)");
    CHECK(*i32_a == *i32_b, "TypeContext caches int32 (equal)");

    auto i64_a = ctx.int64();
    CHECK(i64_a != i32_a, "TypeContext int64 != int32");

    auto ft = ctx.float_type();
    CHECK(ft->is_float(), "TypeContext float_type");
    auto dt = ctx.double_type();
    CHECK(dt->is_double(), "TypeContext double_type");
    auto vt = ctx.void_type();
    CHECK(vt->is_void(), "TypeContext void_type");

    // Pointer caching
    auto p1 = ctx.pointer_to(i32_a);
    auto p2 = ctx.pointer_to(i32_a);
    CHECK(p1 == p2, "TypeContext caches pointer_to (same pointer)");

    auto p3 = ctx.pointer_to(i64_a);
    CHECK(p1 != p3, "TypeContext different pointees = different pointers");

    // get_int
    auto custom = ctx.get_int(42);
    CHECK(custom->bits() == 42, "TypeContext get_int(42)");
}

// ═══════════════════════════════════════════════════════════════════════════
//  Value Tests
// ═══════════════════════════════════════════════════════════════════════════

void test_constant_int() {
    TypeContext ctx;
    auto c = ConstantInt::get(ctx, 42, 32);
    CHECK(c->value() == 42, "ConstantInt value");
    CHECK(c->bit_width() == 32, "ConstantInt bit_width");
    CHECK(c->type()->is_integer(), "ConstantInt type is integer");
    CHECK(c->to_string() == "42", "ConstantInt to_string");

    auto c_neg = ConstantInt::get(ctx, -1, 64);
    CHECK(c_neg->value() == -1, "ConstantInt negative value");
    CHECK(c_neg->bit_width() == 64, "ConstantInt negative bit_width");

    auto c_bool = ConstantInt::get(ctx, 1, 1);
    CHECK(c_bool->value() == 1, "ConstantInt bool value");
    CHECK(c_bool->bit_width() == 1, "ConstantInt bool bit_width");
}

void test_constant_fp() {
    auto ft = std::make_shared<FloatType>();
    auto cf = std::make_shared<ConstantFP>(ft, 3.14);
    CHECK(cf->value() == 3.14, "ConstantFP value");
    CHECK(cf->type()->is_float(), "ConstantFP type is float");

    auto dt = std::make_shared<DoubleType>();
    auto cd = std::make_shared<ConstantFP>(dt, 2.718281828);
    CHECK(cd->type()->is_double(), "ConstantFP double type");
}

void test_special_values() {
    auto i32 = IntegerType::i32();
    auto undef = std::make_shared<UndefValue>(i32);
    CHECK(undef->to_string() == "undef", "UndefValue to_string");
    CHECK(undef->type() == i32, "UndefValue type");

    auto poison = std::make_shared<PoisonValue>(i32);
    CHECK(poison->to_string() == "poison", "PoisonValue to_string");
    CHECK(poison->type() == i32, "PoisonValue type");

    auto ptr_ty = std::make_shared<PointerType>(i32);
    auto null_ptr = std::make_shared<ConstantPointerNull>(ptr_ty);
    CHECK(null_ptr->to_string() == "null", "ConstantPointerNull to_string");
}

void test_value_naming() {
    auto i32 = IntegerType::i32();
    Value v(i32, "result");
    CHECK(v.has_name(), "Value has_name");
    CHECK(v.name() == "result", "Value name");
    CHECK(v.to_string() == "%result", "Value to_string with name");

    Value v2(i32);
    CHECK(!v2.has_name(), "Value no name");
    CHECK(v2.to_string() == "<unnamed>", "Value to_string without name");

    v2.set_name("x");
    CHECK(v2.has_name(), "Value set_name works");
    CHECK(v2.name() == "x", "Value name after set_name");
}

// ═══════════════════════════════════════════════════════════════════════════
//  Instruction Tests
// ═══════════════════════════════════════════════════════════════════════════

void test_make_add() {
    TypeContext ctx;
    auto a = ConstantInt::get(ctx, 10, 32);
    auto b = ConstantInt::get(ctx, 20, 32);
    auto inst = inst::make_add(a, b, "sum");
    CHECK(inst->opcode() == Opcode::Add, "make_add opcode");
    CHECK(inst->num_operands() == 2, "make_add num_operands");
    CHECK(inst->operand(0) == a, "make_add operand 0");
    CHECK(inst->operand(1) == b, "make_add operand 1");
    CHECK(inst->name() == "sum", "make_add name");
    CHECK(inst->is_binary_op(), "add is_binary_op");
    CHECK(!inst->is_terminator(), "add not terminator");
    CHECK(!inst->is_memory_op(), "add not memory_op");
}

void test_make_sub() {
    TypeContext ctx;
    auto a = ConstantInt::get(ctx, 30, 32);
    auto b = ConstantInt::get(ctx, 10, 32);
    auto inst = inst::make_sub(a, b, "diff");
    CHECK(inst->opcode() == Opcode::Sub, "make_sub opcode");
    CHECK(inst->num_operands() == 2, "make_sub num_operands");
    CHECK(inst->is_binary_op(), "sub is_binary_op");
}

void test_make_mul() {
    TypeContext ctx;
    auto a = ConstantInt::get(ctx, 5, 32);
    auto b = ConstantInt::get(ctx, 6, 32);
    auto inst = inst::make_mul(a, b, "prod");
    CHECK(inst->opcode() == Opcode::Mul, "make_mul opcode");
    CHECK(inst->is_binary_op(), "mul is_binary_op");
}

void test_make_icmp() {
    TypeContext ctx;
    auto a = ConstantInt::get(ctx, 1, 32);
    auto b = ConstantInt::get(ctx, 2, 32);
    auto inst = inst::make_icmp(CmpPredicate::SLT, a, b, "cmp");
    CHECK(inst->opcode() == Opcode::ICmp, "make_icmp opcode");
    CHECK(inst->is_cmp(), "icmp is_cmp");
    CHECK(!inst->is_binary_op(), "icmp not binary_op");
    CHECK(inst->num_operands() == 2, "icmp num_operands");
}

void test_make_ret() {
    TypeContext ctx;
    auto val = ConstantInt::get(ctx, 0, 32);
    auto inst = inst::make_ret(val);
    CHECK(inst->opcode() == Opcode::Ret, "make_ret opcode");
    CHECK(inst->is_terminator(), "ret is_terminator");
    CHECK(inst->num_operands() == 1, "make_ret num_operands");
    CHECK(inst->operand(0) == val, "make_ret operand");

    auto void_ret = inst::make_ret_void();
    CHECK(void_ret->opcode() == Opcode::Ret, "make_ret_void opcode");
    CHECK(void_ret->is_terminator(), "ret_void is_terminator");
    CHECK(void_ret->num_operands() == 0, "make_ret_void num_operands");
}

void test_make_br() {
    TypeContext ctx;
    auto cond = ConstantInt::get(ctx, 1, 1);
    auto br = inst::make_br(cond, "then", "else");
    CHECK(br->opcode() == Opcode::Br, "make_br opcode");
    CHECK(br->is_terminator(), "br is_terminator");
    CHECK(br->num_operands() == 1, "conditional br num_operands (cond only)");

    auto ubr = inst::make_br_uncond("target");
    CHECK(ubr->opcode() == Opcode::Br, "make_br_uncond opcode");
    CHECK(ubr->is_terminator(), "uncond br is_terminator");
}

void test_make_alloca() {
    auto i32 = IntegerType::i32();
    auto inst = inst::make_alloca(i32, "ptr", 8);
    CHECK(inst->opcode() == Opcode::Alloca, "make_alloca opcode");
    CHECK(inst->is_memory_op(), "alloca is_memory_op");
    CHECK(!inst->is_terminator(), "alloca not terminator");
    CHECK(inst->name() == "ptr", "alloca name");
    CHECK(inst->alignment().has_value(), "alloca has alignment");
    CHECK(inst->alignment().value() == 8, "alloca alignment value");
}

void test_make_load() {
    TypeContext ctx;
    auto ptr = std::make_shared<ConstantPointerNull>(std::make_shared<PointerType>(ctx.int32()));
    auto inst = inst::make_load(ptr, "val", 4);
    CHECK(inst->opcode() == Opcode::Load, "make_load opcode");
    CHECK(inst->is_memory_op(), "load is_memory_op");
    CHECK(inst->alignment().value() == 4, "load alignment");
}

void test_make_store() {
    TypeContext ctx;
    auto val = ConstantInt::get(ctx, 42, 32);
    auto ptr = std::make_shared<ConstantPointerNull>(std::make_shared<PointerType>(ctx.int32()));
    auto inst = inst::make_store(val, ptr, 4);
    CHECK(inst->opcode() == Opcode::Store, "make_store opcode");
    CHECK(inst->is_memory_op(), "store is_memory_op");
}

void test_make_call() {
    TypeContext ctx;
    auto arg1 = ConstantInt::get(ctx, 1, 32);
    auto arg2 = ConstantInt::get(ctx, 2, 32);
    auto inst = inst::make_call(ctx.int32(), "foo", {arg1, arg2}, "result");
    CHECK(inst->opcode() == Opcode::Call, "make_call opcode");
    CHECK(inst->num_operands() == 2, "call num_operands (args)");
    CHECK(inst->name() == "result", "call name");
}

void test_make_phi() {
    auto i32 = IntegerType::i32();
    auto inst = inst::make_phi(i32, "merged");
    CHECK(inst->opcode() == Opcode::Phi, "make_phi opcode");
    CHECK(inst->name() == "merged", "phi name");
}

void test_make_select() {
    TypeContext ctx;
    auto cond = ConstantInt::get(ctx, 1, 1);
    auto tv = ConstantInt::get(ctx, 10, 32);
    auto fv = ConstantInt::get(ctx, 20, 32);
    auto inst = inst::make_select(cond, tv, fv, "sel");
    CHECK(inst->opcode() == Opcode::Select, "make_select opcode");
    CHECK(inst->num_operands() == 3, "select num_operands");
    CHECK(inst->name() == "sel", "select name");
}

void test_instruction_classification() {
    // Terminators
    TypeContext ctx;
    auto val = ConstantInt::get(ctx, 0, 32);
    CHECK(inst::make_ret(val)->is_terminator(), "ret is_terminator");
    CHECK(inst::make_ret_void()->is_terminator(), "ret_void is_terminator");
    CHECK(inst::make_br_uncond("bb")->is_terminator(), "br is_terminator");

    // Binary ops
    auto a = ConstantInt::get(ctx, 1, 32);
    auto b = ConstantInt::get(ctx, 2, 32);
    CHECK(inst::make_add(a, b)->is_binary_op(), "add is_binary_op");
    CHECK(inst::make_sub(a, b)->is_binary_op(), "sub is_binary_op");
    CHECK(inst::make_mul(a, b)->is_binary_op(), "mul is_binary_op");

    // Memory ops
    CHECK(inst::make_alloca(ctx.int32())->is_memory_op(), "alloca is_memory_op");
    auto nullp = std::make_shared<ConstantPointerNull>(std::make_shared<PointerType>(ctx.int32()));
    CHECK(inst::make_load(nullp)->is_memory_op(), "load is_memory_op");
    CHECK(inst::make_store(val, nullp)->is_memory_op(), "store is_memory_op");

    // Compare
    CHECK(inst::make_icmp(CmpPredicate::EQ, a, b)->is_cmp(), "icmp is_cmp");

    // Cast (using raw Instruction construction for coverage)
    auto trunc = std::make_shared<Instruction>(Opcode::Trunc, ctx.int16(), "t");
    CHECK(trunc->is_cast(), "trunc is_cast");

    auto zext = std::make_shared<Instruction>(Opcode::ZExt, ctx.int32(), "z");
    CHECK(zext->is_cast(), "zext is_cast");
}

void test_binop_flags() {
    TypeContext ctx;
    auto a = ConstantInt::get(ctx, 1, 32);
    auto b = ConstantInt::get(ctx, 2, 32);
    BinOpFlags flags;
    flags.nsw = true;
    flags.nuw = true;
    auto inst = inst::make_add(a, b, "add_nsw", flags);
    CHECK(inst->binop_flags().nsw, "add nsw flag");
    CHECK(inst->binop_flags().nuw, "add nuw flag");
    CHECK(!inst->binop_flags().exact, "add not exact");

    BinOpFlags div_flags;
    div_flags.exact = true;
    auto sdiv = std::make_shared<Instruction>(Opcode::SDiv, ctx.int32(), "d");
    sdiv->add_operand(a);
    sdiv->add_operand(b);
    sdiv->binop_flags() = div_flags;
    CHECK(sdiv->binop_flags().exact, "sdiv exact flag");
    CHECK(!sdiv->binop_flags().nsw, "sdiv not nsw");
}

void test_instruction_metadata() {
    TypeContext ctx;
    auto a = ConstantInt::get(ctx, 1, 32);
    auto b = ConstantInt::get(ctx, 2, 32);
    auto inst = inst::make_add(a, b, "x");
    inst->set_metadata("dbg", "line 42");
    inst->set_metadata("prof", "hot");
    const auto& md = inst->metadata();
    CHECK(md.count("dbg"), "metadata has dbg");
    CHECK(md.count("prof"), "metadata has prof");
    CHECK(md.at("dbg") == "line 42", "metadata dbg value");
    CHECK(md.at("prof") == "hot", "metadata prof value");
}

void test_opcode_name() {
    CHECK(std::string(Instruction::opcode_name(Opcode::Add)) == "add", "opcode_name(Add)");
    CHECK(std::string(Instruction::opcode_name(Opcode::Ret)) == "ret", "opcode_name(Ret)");
    CHECK(std::string(Instruction::opcode_name(Opcode::Br)) == "br", "opcode_name(Br)");
    CHECK(std::string(Instruction::opcode_name(Opcode::Load)) == "load", "opcode_name(Load)");
    CHECK(std::string(Instruction::opcode_name(Opcode::ICmp)) == "icmp", "opcode_name(ICmp)");
}

// ═══════════════════════════════════════════════════════════════════════════
//  BasicBlock Tests
// ═══════════════════════════════════════════════════════════════════════════

void test_basic_block() {
    TypeContext ctx;
    BasicBlock bb("entry");
    CHECK(bb.name() == "entry", "BasicBlock name");
    CHECK(bb.empty(), "BasicBlock initially empty");
    CHECK(bb.size() == 0, "BasicBlock size 0");
    CHECK(bb.terminator() == nullptr, "BasicBlock no terminator initially");
    CHECK(!bb.is_well_formed(), "Empty block not well-formed");

    // Add instructions
    auto a = ConstantInt::get(ctx, 1, 32);
    auto b = ConstantInt::get(ctx, 2, 32);
    auto add = inst::make_add(a, b, "sum");
    bb.add_instruction(add);
    CHECK(bb.size() == 1, "BasicBlock size after add");
    CHECK(!bb.is_well_formed(), "Block without terminator not well-formed");

    auto ret = inst::make_ret(add);
    bb.add_instruction(ret);
    CHECK(bb.size() == 2, "BasicBlock size after ret");
    CHECK(bb.terminator() == ret, "BasicBlock terminator");
    CHECK(bb.is_well_formed(), "Block with terminator is well-formed");

    // Access instructions
    CHECK(bb.instruction(0) == add, "BasicBlock instruction(0)");
    CHECK(bb.instruction(1) == ret, "BasicBlock instruction(1)");

    // Insert
    auto sub = inst::make_sub(a, b, "diff");
    bb.insert_instruction(1, sub);
    CHECK(bb.size() == 3, "BasicBlock size after insert");
    CHECK(bb.instruction(1) == sub, "BasicBlock inserted at pos 1");

    // Replace
    auto mul = inst::make_mul(a, b, "prod");
    bb.replace_instruction(1, mul);
    CHECK(bb.instruction(1) == mul, "BasicBlock replaced instruction");

    // Remove
    bb.remove_instruction(1);
    CHECK(bb.size() == 2, "BasicBlock size after remove");
}

void test_basic_block_successors() {
    TypeContext ctx;
    BasicBlock bb("loop");
    auto cond = ConstantInt::get(ctx, 1, 1);
    auto br = inst::make_br(cond, "body", "exit");
    bb.add_instruction(br);

    auto succs = bb.successors();
    CHECK(succs.size() == 2, "Conditional br has 2 successors");
    // Successor names depend on implementation; just check non-empty

    BasicBlock bb2("end");
    bb2.add_instruction(inst::make_ret_void());
    auto succs2 = bb2.successors();
    CHECK(succs2.empty(), "Ret has no successors");
}

void test_basic_block_predecessors() {
    BasicBlock bb("merge");
    CHECK(bb.predecessors().empty(), "No predecessors initially");
    bb.add_predecessor("entry");
    bb.add_predecessor("loop");
    CHECK(bb.predecessors().size() == 2, "Two predecessors");
    bb.clear_predecessors();
    CHECK(bb.predecessors().empty(), "Predecessors cleared");
}

void test_basic_block_to_string() {
    TypeContext ctx;
    BasicBlock bb("entry");
    auto a = ConstantInt::get(ctx, 1, 32);
    auto b = ConstantInt::get(ctx, 2, 32);
    bb.add_instruction(inst::make_add(a, b, "sum"));
    bb.add_instruction(inst::make_ret(bb.instruction(0)));
    std::string s = bb.to_string();
    CHECK(s.find("entry") != std::string::npos, "to_string contains block name");
}

// ═══════════════════════════════════════════════════════════════════════════
//  Function Tests
// ═══════════════════════════════════════════════════════════════════════════

void test_function() {
    TypeContext ctx;
    auto fn_type = std::make_shared<FunctionType>(ctx.int32(), std::vector<std::shared_ptr<Type>>{ctx.int32(), ctx.int32()});
    Function fn("add", fn_type);
    CHECK(fn.name() == "add", "Function name");
    CHECK(fn.return_type() == ctx.int32(), "Function return_type");
    CHECK(fn.function_type() == fn_type, "Function function_type");
    CHECK(fn.linkage() == Linkage::External, "Function default linkage");
    CHECK(fn.argument_count() == 0, "Function no arguments initially");

    // Add arguments
    fn.add_argument(ctx.int32(), "a");
    fn.add_argument(ctx.int32(), "b");
    CHECK(fn.argument_count() == 2, "Function 2 arguments");
    CHECK(fn.arguments()[0].name == "a", "Function arg 0 name");
    CHECK(fn.arguments()[1].name == "b", "Function arg 1 name");

    // Add basic blocks
    auto& entry = fn.add_block("entry");
    CHECK(fn.blocks().size() == 1, "Function 1 block");
    CHECK(fn.entry_block() != nullptr, "Function has entry_block");
    CHECK(fn.entry_block()->name() == "entry", "Function entry_block name");
    CHECK(fn.block("entry") != nullptr, "Function block by name");
    CHECK(fn.block("nonexistent") == nullptr, "Function block not found");

    // Add instructions to the block
    auto a_val = ConstantInt::get(ctx, 1, 32);
    auto b_val = ConstantInt::get(ctx, 2, 32);
    entry.add_instruction(inst::make_add(a_val, b_val, "sum"));
    entry.add_instruction(inst::make_ret(entry.instruction(0)));

    CHECK(fn.instruction_count() == 2, "Function instruction_count");
}

void test_function_linkage() {
    TypeContext ctx;
    auto fn_type = std::make_shared<FunctionType>(ctx.void_type(), std::vector<std::shared_ptr<Type>>{});
    Function fn("internal_fn", fn_type, Linkage::Internal);
    CHECK(fn.linkage() == Linkage::Internal, "Function internal linkage");
    fn.set_linkage(Linkage::Private);
    CHECK(fn.linkage() == Linkage::Private, "Function changed to private");
}

void test_function_compute_predecessors() {
    TypeContext ctx;
    auto fn_type = std::make_shared<FunctionType>(ctx.int32(), std::vector<std::shared_ptr<Type>>{ctx.int32()});
    Function fn("test", fn_type);
    fn.add_argument(ctx.int32(), "x");

    auto& entry = fn.add_block("entry");
    auto& then_bb = fn.add_block("then");
    auto& else_bb = fn.add_block("else");
    auto& merge = fn.add_block("merge");

    // entry -> then or else
    auto cond = ConstantInt::get(ctx, 1, 1);
    entry.add_instruction(inst::make_br(cond, "then", "else"));

    // then -> merge
    then_bb.add_instruction(inst::make_br_uncond("merge"));

    // else -> merge
    else_bb.add_instruction(inst::make_br_uncond("merge"));

    // merge -> ret
    auto val = ConstantInt::get(ctx, 0, 32);
    merge.add_instruction(inst::make_ret(val));

    fn.compute_predecessors();

    CHECK(merge.predecessors().size() == 2, "merge has 2 predecessors");
    CHECK(then_bb.predecessors().size() == 1, "then has 1 predecessor");
    CHECK(else_bb.predecessors().size() == 1, "else has 1 predecessor");
    CHECK(entry.predecessors().empty(), "entry has no predecessors");
}

void test_function_attributes() {
    TypeContext ctx;
    auto fn_type = std::make_shared<FunctionType>(ctx.void_type(), std::vector<std::shared_ptr<Type>>{});
    Function fn("kernel_fn", fn_type);
    fn.set_attribute("kernel", "1");
    CHECK(fn.is_gpu_kernel(), "Function is_gpu_kernel with kernel attr");

    Function fn2("normal", fn_type);
    CHECK(!fn2.is_gpu_kernel(), "Function not gpu_kernel without attr");
}

void test_function_to_string() {
    TypeContext ctx;
    auto fn_type = std::make_shared<FunctionType>(ctx.int32(), std::vector<std::shared_ptr<Type>>{ctx.int32(), ctx.int32()});
    Function fn("add", fn_type);
    fn.add_argument(ctx.int32(), "a");
    fn.add_argument(ctx.int32(), "b");
    auto& entry = fn.add_block("entry");
    auto a_val = ConstantInt::get(ctx, 1, 32);
    auto b_val = ConstantInt::get(ctx, 2, 32);
    entry.add_instruction(inst::make_add(a_val, b_val, "sum"));
    entry.add_instruction(inst::make_ret(entry.instruction(0)));

    std::string s = fn.to_string();
    CHECK(s.find("add") != std::string::npos, "Function to_string contains name");
    CHECK(s.find("entry") != std::string::npos, "Function to_string contains block name");
}

// ═══════════════════════════════════════════════════════════════════════════
//  Module Tests
// ═══════════════════════════════════════════════════════════════════════════

void test_module() {
    Module mod("test_module");
    CHECK(mod.name() == "test_module", "Module name");
    CHECK(mod.function_count() == 0, "Module no functions initially");
    CHECK(mod.instruction_count() == 0, "Module no instructions initially");

    // Add function
    TypeContext& ctx = mod.type_context();
    auto fn_type = std::make_shared<FunctionType>(ctx.int32(), std::vector<std::shared_ptr<Type>>{ctx.int32(), ctx.int32()});
    auto& fn = mod.add_function("add", fn_type);
    fn.add_argument(ctx.int32(), "a");
    fn.add_argument(ctx.int32(), "b");
    auto& entry = fn.add_block("entry");
    auto a_val = ConstantInt::get(ctx, 1, 32);
    auto b_val = ConstantInt::get(ctx, 2, 32);
    entry.add_instruction(inst::make_add(a_val, b_val, "sum"));
    entry.add_instruction(inst::make_ret(entry.instruction(0)));

    CHECK(mod.function_count() == 1, "Module 1 function");
    CHECK(mod.function("add") != nullptr, "Module function by name");
    CHECK(mod.function("nonexistent") == nullptr, "Module function not found");
    CHECK(mod.instruction_count() == 2, "Module instruction_count");
}

void test_module_globals() {
    Module mod("globals_test");
    TypeContext& ctx = mod.type_context();
    GlobalValue gv;
    gv.name = "g_counter";
    gv.type = ctx.int32();
    gv.init_value = "0";
    gv.is_constant = false;
    mod.add_global(gv);
    CHECK(mod.globals().size() == 1, "Module 1 global");
    CHECK(mod.globals()[0].name == "g_counter", "Module global name");
}

void test_module_target() {
    Module mod("target_test");
    CHECK(!mod.has_target(), "Module no target initially");
    TargetInfo info;
    info.triple = "x86_64-unknown-linux-gnu";
    info.datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128";
    mod.set_target(info);
    CHECK(mod.has_target(), "Module has target");
    CHECK(mod.target().triple == "x86_64-unknown-linux-gnu", "Module target triple");
}

void test_module_named_types() {
    Module mod("types_test");
    TypeContext& ctx = mod.type_context();
    auto point = std::make_shared<StructType>(
        std::vector<std::shared_ptr<Type>>{ctx.int32(), ctx.int32()}, false, "Point");
    mod.add_named_type("Point", point);
    CHECK(mod.named_type("Point") != nullptr, "Module named_type found");
    CHECK(mod.named_type("Missing") == nullptr, "Module named_type not found");
}

void test_module_to_string() {
    Module mod("string_test");
    TypeContext& ctx = mod.type_context();
    auto fn_type = std::make_shared<FunctionType>(ctx.int32(), std::vector<std::shared_ptr<Type>>{});
    auto& fn = mod.add_function("foo", fn_type);
    auto& entry = fn.add_block("entry");
    entry.add_instruction(inst::make_ret(ConstantInt::get(ctx, 42, 32)));

    std::string s = mod.to_string();
    CHECK(!s.empty(), "Module to_string not empty");
}

// ═══════════════════════════════════════════════════════════════════════════
//  IRBuilder Tests
// ═══════════════════════════════════════════════════════════════════════════

void test_ir_builder_simple_function() {
    Module mod("builder_test");
    TypeContext& ctx = mod.type_context();
    IRBuilder builder(ctx);

    auto fn_type = std::make_shared<FunctionType>(ctx.int32(), std::vector<std::shared_ptr<Type>>{ctx.int32(), ctx.int32()});
    auto& fn = mod.add_function("add", fn_type);
    fn.add_argument(ctx.int32(), "a");
    fn.add_argument(ctx.int32(), "b");

    auto& entry = fn.add_block("entry");
    builder.set_insert_point(&entry);

    // Build: %sum = add i32 %a, %b; ret i32 %sum
    auto a_arg = fn.arguments()[0].type;  // We need actual value refs; use constants for now
    auto val_a = builder.get_int32(10);
    auto val_b = builder.get_int32(20);
    auto sum = builder.create_add(val_a, val_b, "sum");
    builder.create_ret(sum);

    CHECK(entry.size() == 2, "IRBuilder entry block has 2 instructions");
    CHECK(entry.instruction(0)->opcode() == Opcode::Add, "IRBuilder first inst is add");
    CHECK(entry.instruction(1)->opcode() == Opcode::Ret, "IRBuilder second inst is ret");
    CHECK(entry.is_well_formed(), "IRBuilder block is well-formed");
}

void test_ir_builder_branching() {
    Module mod("branch_test");
    TypeContext& ctx = mod.type_context();
    IRBuilder builder(ctx);

    auto fn_type = std::make_shared<FunctionType>(ctx.int32(), std::vector<std::shared_ptr<Type>>{ctx.int32()});
    auto& fn = mod.add_function("abs", fn_type);
    fn.add_argument(ctx.int32(), "x");

    auto& entry = fn.add_block("entry");
    auto& pos = fn.add_block("pos");
    auto& neg = fn.add_block("neg");
    auto& merge = fn.add_block("merge");

    builder.set_insert_point(&entry);
    auto zero = builder.get_int32(0);
    auto x = builder.get_int32(5); // placeholder
    auto cmp = builder.create_icmp(CmpPredicate::SGE, x, zero, "is_pos");
    builder.create_cond_br(cmp, "pos", "neg");

    builder.set_insert_point(&pos);
    auto pos_val = builder.get_int32(5);
    builder.create_br("merge");

    builder.set_insert_point(&neg);
    auto neg_val = builder.get_int32(-5);
    builder.create_br("merge");

    builder.set_insert_point(&merge);
    builder.create_ret(zero);

    CHECK(fn.blocks().size() == 4, "IRBuilder 4 blocks");
    CHECK(entry.is_well_formed(), "entry well-formed");
    CHECK(pos.is_well_formed(), "pos well-formed");
    CHECK(neg.is_well_formed(), "neg well-formed");
    CHECK(merge.is_well_formed(), "merge well-formed");
}

void test_ir_builder_memory_ops() {
    Module mod("mem_test");
    TypeContext& ctx = mod.type_context();
    IRBuilder builder(ctx);

    auto fn_type = std::make_shared<FunctionType>(ctx.int32(), std::vector<std::shared_ptr<Type>>{});
    auto& fn = mod.add_function("memtest", fn_type);
    auto& entry = fn.add_block("entry");
    builder.set_insert_point(&entry);

    auto ptr = builder.create_alloca(ctx.int32(), "slot", 4);
    auto val = builder.get_int32(42);
    builder.create_store(val, ptr, 4);
    auto loaded = builder.create_load(ptr, "result", 4);
    builder.create_ret(loaded);

    CHECK(entry.size() == 4, "IRBuilder memory ops: 4 instructions");
    CHECK(entry.instruction(0)->opcode() == Opcode::Alloca, "First inst is alloca");
    CHECK(entry.instruction(1)->opcode() == Opcode::Store, "Second inst is store");
    CHECK(entry.instruction(2)->opcode() == Opcode::Load, "Third inst is load");
    CHECK(entry.instruction(3)->opcode() == Opcode::Ret, "Fourth inst is ret");
}

void test_ir_builder_select_and_call() {
    Module mod("select_call_test");
    TypeContext& ctx = mod.type_context();
    IRBuilder builder(ctx);

    auto fn_type = std::make_shared<FunctionType>(ctx.int32(), std::vector<std::shared_ptr<Type>>{});
    auto& fn = mod.add_function("test", fn_type);
    auto& entry = fn.add_block("entry");
    builder.set_insert_point(&entry);

    auto cond = builder.get_int1(true);
    auto tv = builder.get_int32(1);
    auto fv = builder.get_int32(0);
    auto sel = builder.create_select(cond, tv, fv, "picked");
    builder.create_ret(sel);

    CHECK(entry.instruction(0)->opcode() == Opcode::Select, "First inst is select");
    CHECK(entry.instruction(1)->opcode() == Opcode::Ret, "Second inst is ret");
}

// ═══════════════════════════════════════════════════════════════════════════
//  Main
// ═══════════════════════════════════════════════════════════════════════════

int main() {
    std::cout << "=== Clunk IR Tests ===" << std::endl;

    // Type system
    std::cout << "  Type system..." << std::endl;
    test_void_type();
    test_integer_type();
    test_float_type();
    test_double_type();
    test_pointer_type();
    test_array_type();
    test_struct_type();
    test_function_type();
    test_type_context();

    // Values
    std::cout << "  Values..." << std::endl;
    test_constant_int();
    test_constant_fp();
    test_special_values();
    test_value_naming();

    // Instructions
    std::cout << "  Instructions..." << std::endl;
    test_make_add();
    test_make_sub();
    test_make_mul();
    test_make_icmp();
    test_make_ret();
    test_make_br();
    test_make_alloca();
    test_make_load();
    test_make_store();
    test_make_call();
    test_make_phi();
    test_make_select();
    test_instruction_classification();
    test_binop_flags();
    test_instruction_metadata();
    test_opcode_name();

    // BasicBlock
    std::cout << "  BasicBlock..." << std::endl;
    test_basic_block();
    test_basic_block_successors();
    test_basic_block_predecessors();
    test_basic_block_to_string();

    // Function
    std::cout << "  Function..." << std::endl;
    test_function();
    test_function_linkage();
    test_function_compute_predecessors();
    test_function_attributes();
    test_function_to_string();

    // Module
    std::cout << "  Module..." << std::endl;
    test_module();
    test_module_globals();
    test_module_target();
    test_module_named_types();
    test_module_to_string();

    // IRBuilder
    std::cout << "  IRBuilder..." << std::endl;
    test_ir_builder_simple_function();
    test_ir_builder_branching();
    test_ir_builder_memory_ops();
    test_ir_builder_select_and_call();

    std::cout << "\n=== Results: " << g_pass << " passed, " << g_fail << " failed ===" << std::endl;
    return g_fail > 0 ? 1 : 0;
}
