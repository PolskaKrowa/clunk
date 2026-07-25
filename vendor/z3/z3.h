/*
 * Vendored Z3 C API declarations — subset needed by Clunk.
 * This allows compilation when Z3 development headers are not installed
 * but libz3.so is available at link time.
 *
 * Based on the Z3 public C API (MIT License, Microsoft Research).
 */
#ifndef CLUNK_VENDOR_Z3_H
#define CLUNK_VENDOR_Z3_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Forward declarations ──────────────────────────────────────────────── */
typedef struct _Z3_context*   Z3_context;
typedef struct _Z3_ast*       Z3_ast;
typedef struct _Z3_sort*      Z3_sort;
typedef struct _Z3_func_decl* Z3_func_decl;
typedef struct _Z3_app*       Z3_app;
typedef struct _Z3_expr*      Z3_expr;
typedef struct _Z3_solver*    Z3_solver;
typedef struct _Z3_model*     Z3_model;
typedef struct _Z3_params*    Z3_params;
typedef struct _Z3_tactic*    Z3_tactic;
typedef struct _Z3_goal*      Z3_goal;
typedef struct _Z3_apply_result* Z3_apply_result;
typedef struct _Z3_symbol*    Z3_symbol;
typedef struct _Z3_pattern*   Z3_pattern;
typedef struct _Z3_config*    Z3_config;

/* ── Basic typedefs ────────────────────────────────────────────────────── */
typedef int Z3_bool;
typedef const char* Z3_string;

/* ── Enums ─────────────────────────────────────────────────────────────── */
typedef enum {
    Z3_FALSE = 0,
    Z3_TRUE  = 1,
    Z3_UNDEFINED = 2
} Z3_lbool;

typedef enum {
    Z3_OK              = 0,
    Z3_SORT_ERROR      = 1,
    Z3_IOB             = 2,
    Z3_PARSER_ERROR    = 3,
    Z3_NO_PARSER       = 4,
    Z3_INVALID_ARG     = 5,
    Z3_TYPE_ERROR      = 6
} Z3_error_code;

typedef enum {
    Z3_INT_SORT  = 2,
    Z3_REAL_SORT = 3,
    Z3_BOOL_SORT = 4,
    Z3_BV_SORT   = 5
} Z3_sort_kind;

typedef enum {
    Z3_NUMERAL_AST        = 0,
    Z3_APP_AST            = 1,
    Z3_VAR_AST            = 2,
    Z3_QUANTIFIER_AST     = 3,
    Z3_SORT_AST           = 4,
    Z3_FUNC_DECL_AST      = 5,
    Z3_UNKNOWN_AST        = 1000
} Z3_ast_kind;

/* ── Config & Context ──────────────────────────────────────────────────── */
Z3_config Z3_mk_config(void);
void      Z3_del_config(Z3_config c);
Z3_context Z3_mk_context(Z3_config c);
Z3_context Z3_mk_context_rc(Z3_config c);
void       Z3_del_context(Z3_context c);
void       Z3_set_error_handler(Z3_context c, void (*h)(Z3_context, Z3_error_code));

/* ── Params ────────────────────────────────────────────────────────────── */
Z3_params Z3_mk_params(Z3_context c);
void      Z3_params_inc_ref(Z3_context c, Z3_params p);
void      Z3_params_dec_ref(Z3_context c, Z3_params p);
void      Z3_params_set_uint(Z3_context c, Z3_params p, Z3_symbol k, unsigned v);

/* ── Symbols ───────────────────────────────────────────────────────────── */
Z3_symbol Z3_mk_int_symbol(Z3_context c, int i);
Z3_symbol Z3_mk_string_symbol(Z3_context c, const char* s);

/* ── Sorts ─────────────────────────────────────────────────────────────── */
Z3_sort Z3_mk_bool_sort(Z3_context c);
Z3_sort Z3_mk_int_sort(Z3_context c);
Z3_sort Z3_mk_real_sort(Z3_context c);
Z3_sort Z3_mk_bv_sort(Z3_context c, unsigned sz);
Z3_sort_kind Z3_get_sort_kind(Z3_context c, Z3_sort s);

/* ── Constants & Expressions ───────────────────────────────────────────── */
Z3_ast   Z3_mk_true(Z3_context c);
Z3_ast   Z3_mk_false(Z3_context c);
Z3_ast   Z3_mk_eq(Z3_context c, Z3_ast l, Z3_ast r);
Z3_ast   Z3_mk_distinct(Z3_context c, unsigned n, Z3_ast const* args);
Z3_ast   Z3_mk_not(Z3_context c, Z3_ast a);
Z3_ast   Z3_mk_and(Z3_context c, unsigned n, Z3_ast const* args);
Z3_ast   Z3_mk_or(Z3_context c, unsigned n, Z3_ast const* args);
Z3_ast   Z3_mk_iff(Z3_context c, Z3_ast t1, Z3_ast t2);
Z3_ast   Z3_mk_implies(Z3_context c, Z3_ast t1, Z3_ast t2);
Z3_ast   Z3_mk_ite(Z3_context c, Z3_ast cond, Z3_ast then_val, Z3_ast else_val);

/* ── Integer / BV arithmetic ───────────────────────────────────────────── */
Z3_ast   Z3_mk_int(Z3_context c, int v, Z3_sort s);
Z3_ast   Z3_mk_unsigned_int64(Z3_context c, uint64_t v, Z3_sort s);
Z3_ast   Z3_mk_add(Z3_context c, unsigned n, Z3_ast const* args);
Z3_ast   Z3_mk_mul(Z3_context c, unsigned n, Z3_ast const* args);
Z3_ast   Z3_mk_sub(Z3_context c, unsigned n, Z3_ast const* args);
Z3_ast   Z3_mk_neg(Z3_context c, Z3_ast t);
Z3_ast   Z3_mk_div(Z3_context c, Z3_ast t1, Z3_ast t2);
Z3_ast   Z3_mk_mod(Z3_context c, Z3_ast t1, Z3_ast t2);
Z3_ast   Z3_mk_rem(Z3_context c, Z3_ast t1, Z3_ast t2);
Z3_ast   Z3_mk_lt(Z3_context c, Z3_ast t1, Z3_ast t2);
Z3_ast   Z3_mk_le(Z3_context c, Z3_ast t1, Z3_ast t2);
Z3_ast   Z3_mk_gt(Z3_context c, Z3_ast t1, Z3_ast t2);
Z3_ast   Z3_mk_ge(Z3_context c, Z3_ast t1, Z3_ast t2);

/* ── Bit-vector operations ─────────────────────────────────────────────── */
Z3_ast   Z3_mk_bvadd(Z3_context c, Z3_ast t1, Z3_ast t2);
Z3_ast   Z3_mk_bvsub(Z3_context c, Z3_ast t1, Z3_ast t2);
Z3_ast   Z3_mk_bvmul(Z3_context c, Z3_ast t1, Z3_ast t2);
Z3_ast   Z3_mk_bvudiv(Z3_context c, Z3_ast t1, Z3_ast t2);
Z3_ast   Z3_mk_bvsdiv(Z3_context c, Z3_ast t1, Z3_ast t2);
Z3_ast   Z3_mk_bvurem(Z3_context c, Z3_ast t1, Z3_ast t2);
Z3_ast   Z3_mk_bvsrem(Z3_context c, Z3_ast t1, Z3_ast t2);
Z3_ast   Z3_mk_bvnot(Z3_context c, Z3_ast t1);
Z3_ast   Z3_mk_bvand(Z3_context c, Z3_ast t1, Z3_ast t2);
Z3_ast   Z3_mk_bvor(Z3_context c, Z3_ast t1, Z3_ast t2);
Z3_ast   Z3_mk_bvxor(Z3_context c, Z3_ast t1, Z3_ast t2);
Z3_ast   Z3_mk_bvshl(Z3_context c, Z3_ast t1, Z3_ast t2);
Z3_ast   Z3_mk_bvlshr(Z3_context c, Z3_ast t1, Z3_ast t2);
Z3_ast   Z3_mk_bvashr(Z3_context c, Z3_ast t1, Z3_ast t2);
Z3_ast   Z3_mk_concat(Z3_context c, Z3_ast hi, Z3_ast lo);
Z3_ast   Z3_mk_extract(Z3_context c, unsigned hi, unsigned lo, Z3_ast t);
Z3_ast   Z3_mk_sign_ext(Z3_context c, unsigned i, Z3_ast t);
Z3_ast   Z3_mk_zero_ext(Z3_context c, unsigned i, Z3_ast t);
Z3_ast   Z3_mk_bvult(Z3_context c, Z3_ast t1, Z3_ast t2);
Z3_ast   Z3_mk_bvslt(Z3_context c, Z3_ast t1, Z3_ast t2);
Z3_ast   Z3_mk_bvule(Z3_context c, Z3_ast t1, Z3_ast t2);
Z3_ast   Z3_mk_bvsle(Z3_context c, Z3_ast t1, Z3_ast t2);

/* ── Variables & Quantifiers ───────────────────────────────────────────── */
Z3_ast   Z3_mk_bound(Z3_context c, unsigned idx, Z3_sort s);
Z3_ast   Z3_mk_forall(Z3_context c, unsigned weight,
                       unsigned num_patterns, Z3_pattern const* patterns,
                       unsigned num_decls, Z3_sort const* sorts,
                       Z3_symbol const* decl_names,
                       Z3_ast body);
Z3_ast   Z3_mk_exists(Z3_context c, unsigned weight,
                       unsigned num_patterns, Z3_pattern const* patterns,
                       unsigned num_decls, Z3_sort const* sorts,
                       Z3_symbol const* decl_names,
                       Z3_ast body);
Z3_pattern Z3_mk_pattern(Z3_context c, unsigned n, Z3_ast const* terms);

/* ── Functions ─────────────────────────────────────────────────────────── */
Z3_func_decl Z3_mk_func_decl(Z3_context c, Z3_symbol s,
                              unsigned n, Z3_sort const* domain,
                              Z3_sort range);
Z3_ast       Z3_mk_app(Z3_context c, Z3_func_decl d,
                        unsigned n, Z3_ast const* args);
Z3_ast       Z3_mk_const(Z3_context c, Z3_symbol s, Z3_sort srt);

/* ── Solver ────────────────────────────────────────────────────────────── */
Z3_solver Z3_mk_solver(Z3_context c);
Z3_solver Z3_mk_simple_solver(Z3_context c);
void      Z3_solver_inc_ref(Z3_context c, Z3_solver s);
void      Z3_solver_dec_ref(Z3_context c, Z3_solver s);
void      Z3_solver_push(Z3_context c, Z3_solver s);
void      Z3_solver_pop(Z3_context c, Z3_solver s, unsigned n);
void      Z3_solver_reset(Z3_context c, Z3_solver s);
void      Z3_solver_assert(Z3_context c, Z3_solver s, Z3_ast a);
void      Z3_solver_assert_and_track(Z3_context c, Z3_solver s, Z3_ast a, Z3_ast p);
Z3_lbool  Z3_solver_check(Z3_context c, Z3_solver s);
Z3_lbool  Z3_solver_check_assumptions(Z3_context c, Z3_solver s,
                                       unsigned n, Z3_ast const* assumptions);
Z3_model  Z3_solver_get_model(Z3_context c, Z3_solver s);

/* ── Tactic ────────────────────────────────────────────────────────────── */
Z3_tactic      Z3_mk_tactic(Z3_context c, const char* name);
void           Z3_tactic_inc_ref(Z3_context c, Z3_tactic t);
void           Z3_tactic_dec_ref(Z3_context c, Z3_tactic t);
Z3_apply_result Z3_tactic_apply(Z3_context c, Z3_tactic t, Z3_goal g);
Z3_goal        Z3_mk_goal(Z3_context c, Z3_bool models, Z3_bool unsat_cores, Z3_bool proofs);
void           Z3_goal_inc_ref(Z3_context c, Z3_goal g);
void           Z3_goal_dec_ref(Z3_context c, Z3_goal g);
void           Z3_goal_assert(Z3_context c, Z3_goal g, Z3_ast a);

/* ── Model queries ─────────────────────────────────────────────────────── */
Z3_bool  Z3_model_eval(Z3_context c, Z3_model m, Z3_ast t, Z3_bool model_completion, Z3_ast* v);
void     Z3_model_inc_ref(Z3_context c, Z3_model m);
void     Z3_model_dec_ref(Z3_context c, Z3_model m);

/* ── AST helpers ───────────────────────────────────────────────────────── */
Z3_ast_kind Z3_get_ast_kind(Z3_context c, Z3_ast a);
Z3_string   Z3_ast_to_string(Z3_context c, Z3_ast a);
Z3_sort     Z3_get_sort(Z3_context c, Z3_ast a);
Z3_string   Z3_sort_to_string(Z3_context c, Z3_sort s);
unsigned    Z3_get_bv_sort_size(Z3_context c, Z3_sort s);

/* ── Reference counting ────────────────────────────────────────────────── */
void Z3_inc_ref(Z3_context c, Z3_ast a);
void Z3_dec_ref(Z3_context c, Z3_ast a);

/* ── Simplification ────────────────────────────────────────────────────── */
Z3_ast Z3_simplify(Z3_context c, Z3_ast a);
Z3_ast Z3_simplify_ex(Z3_context c, Z3_ast a, Z3_params p);

#ifdef __cplusplus
}
#endif

#endif /* CLUNK_VENDOR_Z3_H */
