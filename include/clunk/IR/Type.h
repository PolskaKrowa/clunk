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
 * Clunk IR Type System
 * Models a subset of LLVM IR types: void, integers, floats, pointers,
 * arrays, structs, and function types.
 */
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <functional>

namespace clunk::ir {

enum class TypeID {
    Void,
    Integer,
    Float,     // float (32-bit)
    Double,    // double (64-bit)
    Pointer,
    Array,
    Struct,
    Function,
    Vector
};

class Type {
public:
    explicit Type(TypeID id) : id_(id) {}
    virtual ~Type() = default;

    TypeID type_id() const { return id_; }

    virtual std::string to_string() const = 0;
    virtual bool operator==(const Type& other) const = 0;
    bool operator!=(const Type& other) const { return !(*this == other); }

    virtual size_t bit_width() const { return 0; }
    virtual size_t size_bytes() const { return 0; }

    bool is_void()     const { return id_ == TypeID::Void; }
    bool is_integer()  const { return id_ == TypeID::Integer; }
    bool is_float()    const { return id_ == TypeID::Float; }
    bool is_double()   const { return id_ == TypeID::Double; }
    bool is_pointer()  const { return id_ == TypeID::Pointer; }
    bool is_array()    const { return id_ == TypeID::Array; }
    bool is_struct()   const { return id_ == TypeID::Struct; }
    bool is_function() const { return id_ == TypeID::Function; }
    bool is_vector()   const { return id_ == TypeID::Vector; }

protected:
    TypeID id_;
};

// ── Singleton void type ──────────────────────────────────────────────────
class VoidType : public Type {
public:
    VoidType() : Type(TypeID::Void) {}
    std::string to_string() const override { return "void"; }
    bool operator==(const Type& other) const override {
        return other.type_id() == TypeID::Void;
    }
};

// ── Integer type (1-bit to 64-bit) ──────────────────────────────────────
class IntegerType : public Type {
public:
    explicit IntegerType(unsigned bits) : Type(TypeID::Integer), bits_(bits) {}
    unsigned bits() const { return bits_; }
    size_t bit_width() const override { return bits_; }
    size_t size_bytes() const override { return (bits_ + 7) / 8; }

    std::string to_string() const override {
        return "i" + std::to_string(bits_);
    }
    bool operator==(const Type& other) const override {
        return other.type_id() == TypeID::Integer &&
               static_cast<const IntegerType&>(other).bits_ == bits_;
    }

    // Common integer types
    static std::shared_ptr<IntegerType> i1()  { return std::make_shared<IntegerType>(1); }
    static std::shared_ptr<IntegerType> i8()  { return std::make_shared<IntegerType>(8); }
    static std::shared_ptr<IntegerType> i16() { return std::make_shared<IntegerType>(16); }
    static std::shared_ptr<IntegerType> i32() { return std::make_shared<IntegerType>(32); }
    static std::shared_ptr<IntegerType> i64() { return std::make_shared<IntegerType>(64); }

private:
    unsigned bits_;
};

// ── Float type (32-bit IEEE 754) ────────────────────────────────────────
class FloatType : public Type {
public:
    FloatType() : Type(TypeID::Float) {}
    size_t bit_width() const override { return 32; }
    size_t size_bytes() const override { return 4; }
    std::string to_string() const override { return "float"; }
    bool operator==(const Type& other) const override {
        return other.type_id() == TypeID::Float;
    }
};

// ── Double type (64-bit IEEE 754) ──────────────────────────────────────
class DoubleType : public Type {
public:
    DoubleType() : Type(TypeID::Double) {}
    size_t bit_width() const override { return 64; }
    size_t size_bytes() const override { return 8; }
    std::string to_string() const override { return "double"; }
    bool operator==(const Type& other) const override {
        return other.type_id() == TypeID::Double;
    }
};

// ── Pointer type ────────────────────────────────────────────────────────
class PointerType : public Type {
public:
    explicit PointerType(std::shared_ptr<Type> pointee, unsigned addrspace = 0)
        : Type(TypeID::Pointer), pointee_(pointee), addrspace_(addrspace) {}

    std::shared_ptr<Type> pointee() const { return pointee_; }
    unsigned address_space() const { return addrspace_; }
    size_t bit_width() const override { return 64; } // Assume 64-bit pointers
    size_t size_bytes() const override { return 8; }

    std::string to_string() const override {
        // A void pointee means an OPAQUE pointer (LLVM 15+ `ptr`): the
        // parser maps `ptr` to pointer-to-void, and `void*` is not valid
        // LLVM syntax, so print the opaque spelling back.
        if (pointee_ && pointee_->type_id() == TypeID::Void) {
            std::string s = "ptr";
            if (addrspace_ != 0) s += " addrspace(" + std::to_string(addrspace_) + ")";
            return s;
        }
        std::string s = pointee_->to_string() + "*";
        if (addrspace_ != 0) s += " addrspace(" + std::to_string(addrspace_) + ")";
        return s;
    }
    bool operator==(const Type& other) const override {
        if (other.type_id() != TypeID::Pointer) return false;
        auto& ptr = static_cast<const PointerType&>(other);
        return *pointee_ == *ptr.pointee_ && addrspace_ == ptr.addrspace_;
    }

private:
    std::shared_ptr<Type> pointee_;
    unsigned addrspace_;
};

// ── Array type ──────────────────────────────────────────────────────────
class ArrayType : public Type {
public:
    ArrayType(uint64_t count, std::shared_ptr<Type> elem)
        : Type(TypeID::Array), count_(count), elem_(elem) {}

    uint64_t count() const { return count_; }
    std::shared_ptr<Type> element_type() const { return elem_; }
    size_t size_bytes() const override { return count_ * elem_->size_bytes(); }

    std::string to_string() const override {
        return "[" + std::to_string(count_) + " x " + elem_->to_string() + "]";
    }
    bool operator==(const Type& other) const override {
        if (other.type_id() != TypeID::Array) return false;
        auto& arr = static_cast<const ArrayType&>(other);
        return count_ == arr.count_ && *elem_ == *arr.elem_;
    }

private:
    uint64_t count_;
    std::shared_ptr<Type> elem_;
};

// ── Vector type (fixed-width SIMD vector: <N x elem>) ───────────────────
class VectorType : public Type {
public:
    VectorType(uint64_t count, std::shared_ptr<Type> elem)
        : Type(TypeID::Vector), count_(count), elem_(elem) {}

    uint64_t count() const { return count_; }
    std::shared_ptr<Type> element_type() const { return elem_; }
    size_t bit_width() const override { return count_ * elem_->bit_width(); }
    size_t size_bytes() const override { return count_ * elem_->size_bytes(); }

    std::string to_string() const override {
        return "<" + std::to_string(count_) + " x " + elem_->to_string() + ">";
    }
    bool operator==(const Type& other) const override {
        if (other.type_id() != TypeID::Vector) return false;
        auto& v = static_cast<const VectorType&>(other);
        return count_ == v.count_ && *elem_ == *v.elem_;
    }

private:
    uint64_t count_;
    std::shared_ptr<Type> elem_;
};

// ── Struct type ─────────────────────────────────────────────────────────
class StructType : public Type {
public:
    explicit StructType(std::vector<std::shared_ptr<Type>> fields,
                        bool is_packed = false, const std::string& name = "")
        : Type(TypeID::Struct), fields_(std::move(fields)),
          packed_(is_packed), name_(name) {}

    const std::vector<std::shared_ptr<Type>>& fields() const { return fields_; }
    bool is_packed() const { return packed_; }
    const std::string& name() const { return name_; }
    size_t field_count() const { return fields_.size(); }

    size_t size_bytes() const override {
        size_t s = 0;
        for (auto& f : fields_) s += f->size_bytes();
        return s;
    }

    std::string to_string() const override {
        if (!name_.empty()) return "%" + name_;
        std::string s = packed_ ? "<{ " : "{ ";
        for (size_t i = 0; i < fields_.size(); ++i) {
            if (i > 0) s += ", ";
            s += fields_[i]->to_string();
        }
        s += packed_ ? " }>" : " }";
        return s;
    }
    bool operator==(const Type& other) const override {
        if (other.type_id() != TypeID::Struct) return false;
        auto& st = static_cast<const StructType&>(other);
        if (fields_.size() != st.fields_.size()) return false;
        for (size_t i = 0; i < fields_.size(); ++i) {
            if (*fields_[i] != *st.fields_[i]) return false;
        }
        return true;
    }

private:
    std::vector<std::shared_ptr<Type>> fields_;
    bool packed_;
    std::string name_;
};

// ── Function type ───────────────────────────────────────────────────────
class FunctionType : public Type {
public:
    FunctionType(std::shared_ptr<Type> ret,
                 std::vector<std::shared_ptr<Type>> params,
                 bool is_vararg = false)
        : Type(TypeID::Function), ret_(ret), params_(std::move(params)),
          vararg_(is_vararg) {}

    std::shared_ptr<Type> return_type() const { return ret_; }
    const std::vector<std::shared_ptr<Type>>& params() const { return params_; }
    bool is_vararg() const { return vararg_; }

    std::string to_string() const override {
        std::string s = ret_->to_string() + " (";
        for (size_t i = 0; i < params_.size(); ++i) {
            if (i > 0) s += ", ";
            s += params_[i]->to_string();
        }
        if (vararg_) s += ", ...";
        s += ")";
        return s;
    }
    bool operator==(const Type& other) const override {
        if (other.type_id() != TypeID::Function) return false;
        auto& ft = static_cast<const FunctionType&>(other);
        if (*ret_ != *ft.ret_ || params_.size() != ft.params_.size()) return false;
        for (size_t i = 0; i < params_.size(); ++i) {
            if (*params_[i] != *ft.params_[i]) return false;
        }
        return vararg_ == ft.vararg_;
    }

private:
    std::shared_ptr<Type> ret_;
    std::vector<std::shared_ptr<Type>> params_;
    bool vararg_;
};

// ── Type cache ──────────────────────────────────────────────────────────
class TypeContext {
public:
    struct PtrKey {
        Type* pointee;
        unsigned addrspace;
        bool operator==(const PtrKey& o) const noexcept {
            return pointee == o.pointee && addrspace == o.addrspace;
        }
    };
    struct PtrKeyHash {
        size_t operator()(const PtrKey& k) const noexcept {
            // Mix pointer bits and addrspace
            auto h = std::hash<void*>{}(static_cast<void*>(k.pointee));
            h ^= std::hash<unsigned>{}(k.addrspace) + 0x9e3779b9u + (h << 6) + (h >> 2);
            return h;
        }
    };
    struct VecKey {
        Type* elem;
        uint64_t count;
        bool operator==(const VecKey& o) const noexcept {
            return elem == o.elem && count == o.count;
        }
    };
    struct VecKeyHash {
        size_t operator()(const VecKey& k) const noexcept {
            auto h = std::hash<void*>{}(static_cast<void*>(k.elem));
            h ^= std::hash<uint64_t>{}(k.count) + 0x9e3779b9u + (h << 6) + (h >> 2);
            return h;
        }
    };

    std::shared_ptr<VoidType> void_type() { return void_ty_; }
    std::shared_ptr<IntegerType> int1()  { return get_int(1); }
    std::shared_ptr<IntegerType> int8()  { return get_int(8); }
    std::shared_ptr<IntegerType> int16() { return get_int(16); }
    std::shared_ptr<IntegerType> int32() { return get_int(32); }
    std::shared_ptr<IntegerType> int64() { return get_int(64); }
    std::shared_ptr<FloatType>   float_type()  { return float_ty_; }
    std::shared_ptr<DoubleType>  double_type() { return double_ty_; }

    std::shared_ptr<PointerType> pointer_to(std::shared_ptr<Type> pointee, unsigned as = 0) {
        PtrKey key{pointee.get(), as};
        auto it = ptr_cache_.find(key);
        if (it != ptr_cache_.end()) return it->second;
        auto p = std::make_shared<PointerType>(pointee, as);
        ptr_cache_[key] = p;
        return p;
    }

    std::shared_ptr<IntegerType> get_int(unsigned bits) {
        auto it = int_cache_.find(bits);
        if (it != int_cache_.end()) return it->second;
        auto t = std::make_shared<IntegerType>(bits);
        int_cache_[bits] = t;
        return t;
    }

    std::shared_ptr<VectorType> get_vector(std::shared_ptr<Type> elem, uint64_t count) {
        VecKey key{elem.get(), count};
        auto it = vec_cache_.find(key);
        if (it != vec_cache_.end()) return it->second;
        auto v = std::make_shared<VectorType>(count, elem);
        vec_cache_[key] = v;
        return v;
    }

private:
    std::shared_ptr<VoidType>  void_ty_  = std::make_shared<VoidType>();
    std::shared_ptr<FloatType> float_ty_ = std::make_shared<FloatType>();
    std::shared_ptr<DoubleType> double_ty_ = std::make_shared<DoubleType>();
    std::unordered_map<unsigned, std::shared_ptr<IntegerType>> int_cache_;
    std::unordered_map<PtrKey, std::shared_ptr<PointerType>, PtrKeyHash> ptr_cache_;
    std::unordered_map<VecKey, std::shared_ptr<VectorType>, VecKeyHash> vec_cache_;
};

} // namespace clunk::ir
