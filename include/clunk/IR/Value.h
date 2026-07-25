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
 * Clunk IR Value — base class for all SSA values.
 */
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>
#include "clunk/IR/Type.h"

namespace clunk::ir {

class Value {
public:
    Value(std::shared_ptr<Type> type, const std::string& name = "")
        : type_(type), name_(name) {}
    virtual ~Value() = default;

    std::shared_ptr<Type> type() const { return type_; }
    const std::string& name() const { return name_; }
    void set_name(const std::string& n) { name_ = n; }

    bool has_name() const { return !name_.empty(); }

    virtual std::string to_string() const {
        if (has_name()) return "%" + name_;
        return "<unnamed>";
    }

    // How this value should be rendered when used as an operand of
    // another instruction. The default is the same as to_string(), but
    // Instruction overrides this to print just "%name" (not the full
    // instruction definition) — otherwise `ret i32 %r` would print as
    // `ret i32 %r = add i32 %a, %b`, recursing into the operand's def.
    virtual std::string print_as_operand() const {
        return to_string();
    }

protected:
    std::shared_ptr<Type> type_;
    std::string name_;
};

// ── Constant values ─────────────────────────────────────────────────────
class ConstantInt final : public Value {
public:
    ConstantInt(std::shared_ptr<IntegerType> type, int64_t value)
        : Value(type), value_(value) {}

    int64_t value() const { return value_; }
    unsigned bit_width() const { return type_->bit_width(); }

    std::string to_string() const override {
        return std::to_string(value_);
    }

    static std::shared_ptr<ConstantInt> get(TypeContext& ctx, int64_t val, unsigned bits = 32) {
        return std::make_shared<ConstantInt>(ctx.get_int(bits), val);
    }

private:
    int64_t value_;
};

class ConstantFP final : public Value {
public:
    ConstantFP(std::shared_ptr<Type> type, double value)
        : Value(type), value_(value) {}

    double value() const { return value_; }

    std::string to_string() const override {
        // Emit VALID LLVM FP syntax. %.17g prints 1.0 as "1", which LLVM
        // rejects ("integer constant must have integer type") — so ensure
        // a '.' or exponent is present. Non-finite values use LLVM's
        // raw-bits hex form (0x...), which our parser also reads back.
        if (!std::isfinite(value_)) {
            uint64_t bits = 0;
            static_assert(sizeof(bits) == sizeof(value_), "64-bit double");
            std::memcpy(&bits, &value_, sizeof(bits));
            char buf[32];
            snprintf(buf, sizeof(buf), "0x%016llX",
                     static_cast<unsigned long long>(bits));
            return std::string(buf);
        }
        char buf[64];
        snprintf(buf, sizeof(buf), "%.17g", value_);
        std::string s(buf);
        if (s.find('.') == std::string::npos &&
            s.find('e') == std::string::npos &&
            s.find('E') == std::string::npos) {
            s += ".0";
        }
        return s;
    }

private:
    double value_;
};

// A constant fixed-width vector, e.g. `<i32 1, i32 2, i32 3, i32 4>`.
// Elements are typically ConstantInt (one per lane, in lane order); undef
// lanes are represented by UndefValue elements. Also used as the constant
// mask operand of shufflevector (a vector of i32 lane indices, where -1
// encodes an undef lane).
class ConstantVector final : public Value {
public:
    ConstantVector(std::shared_ptr<VectorType> type,
                   std::vector<std::shared_ptr<Value>> elements)
        : Value(type), elements_(std::move(elements)) {}

    const std::vector<std::shared_ptr<Value>>& elements() const { return elements_; }
    size_t lane_count() const { return elements_.size(); }
    std::shared_ptr<Value> element(size_t i) const { return elements_.at(i); }

    std::string to_string() const override {
        auto* vt = static_cast<const VectorType*>(type_.get());
        std::string elem_ty = vt->element_type()->to_string();
        std::string s = "<";
        for (size_t i = 0; i < elements_.size(); ++i) {
            if (i > 0) s += ", ";
            s += elem_ty + " " + elements_[i]->to_string();
        }
        s += ">";
        return s;
    }

    // Build a constant vector of integer lanes.
    static std::shared_ptr<ConstantVector> get_int_lanes(
        TypeContext& ctx, const std::vector<int64_t>& lanes, unsigned elem_bits) {
        std::vector<std::shared_ptr<Value>> elems;
        elems.reserve(lanes.size());
        for (int64_t v : lanes) elems.push_back(ConstantInt::get(ctx, v, elem_bits));
        auto vt = ctx.get_vector(ctx.get_int(elem_bits), lanes.size());
        return std::make_shared<ConstantVector>(vt, std::move(elems));
    }

private:
    std::vector<std::shared_ptr<Value>> elements_;
};

class ConstantPointerNull final : public Value {
public:
    ConstantPointerNull(std::shared_ptr<PointerType> ptr_type)
        : Value(ptr_type) {}
    std::string to_string() const override { return "null"; }
};

class UndefValue final : public Value {
public:
    explicit UndefValue(std::shared_ptr<Type> type) : Value(type) {}
    std::string to_string() const override { return "undef"; }
};

class PoisonValue final : public Value {
public:
    explicit PoisonValue(std::shared_ptr<Type> type) : Value(type) {}
    std::string to_string() const override { return "poison"; }
};

} // namespace clunk::ir
