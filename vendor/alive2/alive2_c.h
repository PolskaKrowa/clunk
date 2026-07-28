// SPDX-License-Identifier: MIT
// Clunk — vendored Alive2 C API declarations for dlopen integration.
//
// This file declares the subset of the Alive2 C API that Clunk uses.
// It is NOT a full copy of Alive2's headers — it contains only the
// opaque types and function signatures needed by AliveDynamicLoader.
//
// Alive2 (https://github.com/AliveToolkit/alive2, PLDI'21) exposes a
// C API via `alive2/alive_tvs.h`. The shared library is typically
// `libAlive2.so` (or `libalive2.so`, distribution-dependent).
//
// NOTE: Alive2 depends on LLVM. The library must be built against a
// compatible LLVM version. This is why it's loaded at runtime via
// dlopen — Clunk itself has no LLVM dependency.

#pragma once

#include <cstdint>
#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

// ── Opaque types ────────────────────────────────────────────────────────────

// A single verification result from Alive2.
// The pointer is owned by the caller and must be freed with alive_result_free.
struct alive_result;

// ── alive_result accessors ──────────────────────────────────────────────────

// Status codes returned by alive_result_status().
#define ALIVE_STATUS_CORRECT      0  // Transformation is correct (refinement holds)
#define ALIVE_STATUS_INCORRECT    1  // Transformation is incorrect (counterexample found)
#define ALIVE_STATUS_UNKNOWN      2  // Alive2 could not determine (timeout, etc.)
#define ALIVE_STATUS_ERROR        3  // Tool error (parse failure, etc.)
#define ALIVE_STATUS_UNSUPPORTED  4  // Unsupported construct encountered

// Returns the status code (see ALIVE_STATUS_* defines above).
int alive_result_status(const struct alive_result* r);

// Returns a human-readable message. The returned pointer is valid until
// alive_result_free is called on the owning result.
const char* alive_result_message(const struct alive_result* r);

// Returns a JSON-formatted string with additional details, or NULL.
// The returned pointer is valid until alive_result_free is called.
const char* alive_result_json(const struct alive_result* r);

// Returns the number of source/target IR lines, useful for error messages.
int alive_result_source_lines(const struct alive_result* r);
int alive_result_target_lines(const struct alive_result* r);

// Free a result obtained from alive_tvs_verify.
void alive_result_free(struct alive_result* r);

// ── Verification API ────────────────────────────────────────────────────────

// Verify that transforming `src` IR into `tgt` IR is correct.
//
// Both `src` and `tgt` are strings containing LLVM IR function definitions.
// They should be standalone modules (with target triple/datalayout and any
// necessary declarations).
//
// `src` is the "source" (original) and `tgt` is the "target" (optimised).
// Alive2 checks: for every input that doesn't cause UB in `src`,
// `tgt` must produce the same result and not introduce new UB.
//
// Returns a heap-allocated result; caller must free with alive_result_free.
// If Alive2's internal initialisation fails, returns a result with
// status ALIVE_STATUS_ERROR.
struct alive_result* alive_tvs_verify(const char* src, const char* tgt);

// ── Configuration ───────────────────────────────────────────────────────────

// Set the SMT solver timeout in milliseconds. 0 = no limit (use Alive2's
// default). This is a global setting that affects all subsequent calls.
void alive_set_timeout(unsigned ms);

// Set whether to disable undefined input variables.
// When disabled, all inputs are treated as concrete/defined.
void alive_set_disable_undef_input(int disable);

// Return the Alive2 version string (e.g. "0.1.0").
const char* alive_version(void);

#ifdef __cplusplus
}
#endif
