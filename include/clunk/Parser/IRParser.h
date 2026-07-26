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
 * Clunk IR Parser — reads LLVM IR text (.ll) into our in-memory representation.
 * Handles a practical subset of LLVM IR sufficient for superoptimisation.
 */
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <functional>
#include "clunk/IR/Module.h"

namespace clunk::parser {

struct ParseError : public std::runtime_error {
    using std::runtime_error::runtime_error;
};

struct ParseLocation {
    size_t line = 0;
    size_t col = 0;
};

class IRParser final {
public:
    IRParser();

    // Parse a complete .ll file
    std::shared_ptr<ir::Module> parse_file(const std::string& path);
    std::shared_ptr<ir::Module> parse_string(const std::string& ir_text);

    // Access any warnings generated during parsing
    const std::vector<std::string>& warnings() const { return warnings_; }

    // Register custom type resolvers for types the parser doesn't know
    using TypeResolver = std::function<std::shared_ptr<ir::Type>(const std::string&)>;
    void register_type_resolver(TypeResolver resolver);

private:
    // ── Token types ──────────────────────────────────────────────────────
    enum class TokenType {
        Identifier, Number, String, Punctuation,
        Keyword, Type, Eof
    };

    struct Token {
        TokenType type;
        std::string text;
        ParseLocation loc;
    };

    // ── Lexer ───────────────────────────────────────────────────────────
    std::vector<Token> tokenize(const std::string& source);
    Token next_token();
    Token peek_token() const;
    bool match(TokenType type, const std::string& text = "");
    Token expect(TokenType type, const std::string& text = "");

    // ── Top-level parsing ───────────────────────────────────────────────
    void parse_module(std::shared_ptr<ir::Module> mod);
    void parse_target(std::shared_ptr<ir::Module> mod);
    void parse_global(std::shared_ptr<ir::Module> mod);
    void parse_function_decl(std::shared_ptr<ir::Module> mod);
    void parse_function_def(std::shared_ptr<ir::Module> mod);

    // ── Type parsing ────────────────────────────────────────────────────
    std::shared_ptr<ir::Type> parse_type();

    // ── Instruction parsing ─────────────────────────────────────────────
    std::shared_ptr<ir::Instruction> parse_instruction(ir::Function& fn);
    std::shared_ptr<ir::Instruction> parse_binop(const std::string& opcode_name,
                                                   const std::string& result_name);
    std::shared_ptr<ir::Instruction> parse_memory_op(const std::string& opcode_name,
                                                       const std::string& result_name);
    std::shared_ptr<ir::Instruction> parse_terminator(const std::string& opcode_name);
    std::shared_ptr<ir::Instruction> parse_conversion_op(const std::string& opcode_name,
                                                          const std::string& result_name);

    // ── Value/operand parsing ───────────────────────────────────────────
    std::shared_ptr<ir::Value> parse_value(std::shared_ptr<ir::Type> expected_type);

    // ── Helpers ─────────────────────────────────────────────────────────
    std::shared_ptr<ir::Type> resolve_type(const std::string& name);
    std::string parse_name(); // %name or @name

    // ── Skip helpers (silently consume tokens) ──────────────────────────
    void skip_metadata();               // skip one !metadata node
    void skip_trailing_metadata();      // skip [, !kind !val]* after instructions
    void skip_balanced(char open, char close); // skip balanced parens/braces/brackets
    void skip_param_attrs();            // skip parameter attributes before a type
    void skip_function_attrs_until_brace(); // skip function attrs between ) and {
    std::vector<std::string> collect_function_attrs(); // collect function attrs between ) and {

    // ── Module-level metadata parsing ───────────────────────────────────
    void parse_llvm_module_flags(std::shared_ptr<ir::Module> mod);
    bool try_parse_module_flag_entry(std::shared_ptr<ir::Module> mod);

    // ── Round-trip preservation helpers ─────────────────────────────────
    // These capture constructs the parser doesn't semantically model but
    // must round-trip verbatim so clang can recompile the emitted IR.
    void parse_attribute_group(std::shared_ptr<ir::Module> mod);          // `attributes #N = { ... }`
    void parse_named_metadata(std::shared_ptr<ir::Module> mod);           // `!name = !{ ... }`
    void parse_metadata_def(std::shared_ptr<ir::Module> mod);             // `!N = !{ ... }`
    void parse_module_asm(std::shared_ptr<ir::Module> mod);               // `module asm "..."`

    // Reconstruct the textual form of a token (handling quoted strings
    // and punctuation) so we can store raw RHS bodies verbatim.  Returns
    // the reconstructed text and consumes exactly one token from pos_.
    std::string token_to_text(const Token& tok) const;

    // ── State ───────────────────────────────────────────────────────────
    std::vector<Token> tokens_;
    size_t pos_ = 0;
    std::vector<std::string> warnings_;
    std::unordered_map<std::string, std::shared_ptr<ir::Value>> value_table_; // %name -> Value
    std::unordered_map<std::string, std::shared_ptr<ir::Type>> type_table_;   // %struct.X -> Type
    std::vector<TypeResolver> type_resolvers_;
    ir::TypeContext type_ctx_;
};

} // namespace clunk::parser
