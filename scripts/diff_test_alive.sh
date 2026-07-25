# SPDX-License-Identifier: GPL-3.0-or-later
# Clunk — LLVM IR superoptimiser
# Copyright (C) 2025 Clunk contributors
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

#!/usr/bin/env bash
#
# diff_test_alive.sh — differential testing of clunk output against clang -O3
# using alive-tv (Alive2) as the gold-standard soundness backstop.
#
# Differential-testing harness using alive-tv (Alive2 PLDI'21,
# https://github.com/AliveToolkit/alive2) as a gold-standard soundness
# backstop. alive-tv checks refinement between two LLVM-IR functions:
# a "source" and a "target". A refinement `src >= tgt` holds iff for
# every input that doesn't trigger UB in `src`, `tgt` produces the same
# value (and doesn't introduce new UB). We use alive-tv as a post-hoc
# check on clunk's own SMT verifier — any disagreement is a SOUNDNESS BUG
# in clunk.
#
# For every C source in the corpus:
#   1. Emit the clang -O3 baseline IR (scalar, no vectorize/unroll so the
#      output stays inside clunk's parser + SMT-modelable subset — mirrors
#      scripts/beat_o3.sh).
#   2. Run clunk on that IR (producing optimised IR).
#   3. If alive-tv is available on PATH, verify that clunk's output is a
#      refinement of the -O3 input. Any alive-tv failure is reported as a
#      SOUNDNESS BUG and exits non-zero.
#   4. Aggregate per-function results into a summary table.
#
# Usage:
#   scripts/diff_test_alive.sh [corpus_dir] [work_dir]
#
# Env overrides:
#   CLUNK         path to the clunk binary (default: $REPO_ROOT/build/clunk)
#   CLANG         path to the clang binary (default: clang)
#   ALIVE_TV      path to the alive-tv binary (default: alive-tv; may be unset)
#   CLUNK_FLAGS   extra flags for clunk (default: "--opt-level 3 --time-budget 8")
#   ALIVE_TV_FLAGS extra flags for alive-tv (default: "")
#
# Exit codes:
#   0 — all alive-tv checks passed (or alive-tv was not installed and the
#       script ran in degraded mode without soundness backstop).
#   1 — at least one alive-tv check FAILED (soundness bug detected).
#   2 — environment error (clunk or clang not found, empty corpus, etc.).
#
# If alive-tv is not on PATH (and $ALIVE_TV is unset or empty), the script
# emits a warning and skips step 3 — it still runs clunk on the corpus and
# reports per-function results, but does NOT claim soundness.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CORPUS_DIR="${1:-$REPO_ROOT/benchmarks/corpus}"
WORK_DIR="${2:-$REPO_ROOT/build/diff_test_alive}"
CLUNK="${CLUNK:-$REPO_ROOT/build/clunk}"
CLANG="${CLANG:-clang}"
ALIVE_TV="${ALIVE_TV:-alive-tv}"
# Keep -O3 output scalar & loop-free so it stays SMT-modelable.
O3_FLAGS=(-O3 -fno-vectorize -fno-slp-vectorize -fno-unroll-loops
          -Xclang -disable-O0-optnone -emit-llvm -S)
# clunk: full search, verify with SMT, short per-function budget.
read -r -a CLUNK_FLAGS <<< "${CLUNK_FLAGS:---opt-level 3 --time-budget 8}"
# alive-tv extra flags (e.g. --no-undef-inputs for some harnesses).
read -r -a ALIVE_TV_FLAGS <<< "${ALIVE_TV_FLAGS:-}"

# ── Preflight checks ────────────────────────────────────────────────────────
if [[ ! -x "$CLUNK" ]]; then
    echo "error: clunk binary not found/executable at '$CLUNK' (build it first, or set \$CLUNK)" >&2
    exit 2
fi
if ! command -v "$CLANG" >/dev/null 2>&1; then
    echo "error: clang not found (set \$CLANG)" >&2
    exit 2
fi

# Detect alive-tv. If not available, warn loudly and continue in degraded
# mode — the script is still useful as a clunk-smoke-test, but the
# soundness backstop is gone.
ALIVE_TV_AVAILABLE=0
ALIVE_TV_PATH=""
if [[ -n "${ALIVE_TV:-}" ]] && command -v "$ALIVE_TV" >/dev/null 2>&1; then
    ALIVE_TV_AVAILABLE=1
    ALIVE_TV_PATH="$(command -v "$ALIVE_TV")"
fi

mkdir -p "$WORK_DIR"
echo "corpus    : $CORPUS_DIR"
echo "clunk     : $CLUNK ${CLUNK_FLAGS[*]}"
echo "clang     : $($CLANG --version | head -1)"
if (( ALIVE_TV_AVAILABLE )); then
    echo "alive-tv  : $ALIVE_TV_PATH ${ALIVE_TV_FLAGS[*]:-}"
else
    echo "alive-tv  : NOT FOUND — soundness backstop DISABLED"
    echo "            (install from https://github.com/AliveToolkit/alive2)"
fi
echo "work dir  : $WORK_DIR"
echo

shopt -s nullglob
sources=("$CORPUS_DIR"/*.c)
if [[ ${#sources[@]} -eq 0 ]]; then
    echo "error: no .c sources in $CORPUS_DIR" >&2
    exit 2
fi

# ── Per-file diff-test loop ─────────────────────────────────────────────────
# Accumulators for the summary.
total_functions=0       # functions seen (excluding declarations)
total_checked=0         # functions actually checked with alive-tv
total_passed=0          # alive-tv said Equivalent / No refinement failure
total_failed=0          # alive-tv said NotEquivalent / refinement failure
total_clunk_errors=0    # clunk itself crashed / returned non-zero

# Use a flat temp file list to print as a single summary at the end.
summary="$WORK_DIR/summary.txt"
: >"$summary"

for src in "${sources[@]}"; do
    base="$(basename "${src%.c}")"
    o3_ll="$WORK_DIR/${base}_o3.ll"
    opt_ll="$WORK_DIR/${base}_clunk.ll"
    json="$WORK_DIR/${base}.json"

    echo ">> $base : emitting -O3 baseline"
    "$CLANG" "${O3_FLAGS[@]}" "$src" -o "$o3_ll"

    echo ">> $base : running clunk"
    # JSON report on stdout; optimised IR to file; clunk chatter to stderr log.
    if ! "$CLUNK" "${CLUNK_FLAGS[@]}" --report-json --output "$opt_ll" "$o3_ll" \
            >"$json" 2>"$WORK_DIR/${base}.log"; then
        echo "   ERROR: clunk failed on $base (see $WORK_DIR/${base}.log)" >&2
        echo "ERROR     clunk-failed    $base" >>"$summary"
        total_clunk_errors=$((total_clunk_errors + 1))
        continue
    fi

    # Count functions in the JSON (skip declarations / pass-throughs).
    fn_count=$(python3 - "$json" <<'PY'
import json, sys
try:
    data = json.load(open(sys.argv[1]))
except Exception:
    print(0); sys.exit(0)
print(sum(1 for r in data if r.get("rounds_run", 0) > 0))
PY
)
    total_functions=$((total_functions + fn_count))

    # ── alive-tv soundness backstop ────────────────────────────────────────
    if (( ! ALIVE_TV_AVAILABLE )); then
        echo "   (skipping alive-tv check — not installed)"
        echo "SKIP      no-alive-tv     $base" >>"$summary"
        continue
    fi

    # alive-tv takes two .ll files (or two functions) and checks refinement
    # of the second into the first (src >= tgt). We pass the -O3 baseline as
    # src and clunk's output as tgt — clunk must refine -O3.
    #
    # The output of alive-tv is human-readable; a refinement failure is
    # signalled by a non-zero exit code AND/OR a line containing
    # "Source: ... Target: ... FAILED" or similar. We treat any non-zero
    # exit as a soundness bug.
    alive_log="$WORK_DIR/${base}_alive.log"
    if "$ALIVE_TV_PATH" "${ALIVE_TV_FLAGS[@]}" "$o3_ll" "$opt_ll" \
            >"$alive_log" 2>&1; then
        echo "   alive-tv: PASS"
        echo "PASS      equiv   $base" >>"$summary"
        total_passed=$((total_passed + 1))
        total_checked=$((total_checked + 1))
    else
        rc=$?
        echo "   alive-tv: *** SOUNDNESS BUG *** (exit $rc; see $alive_log)"
        echo "FAIL      refinement-failed       $base" >>"$summary"
        total_failed=$((total_failed + 1))
        total_checked=$((total_checked + 1))
    fi
done

# ── Summary ─────────────────────────────────────────────────────────────────
echo
echo "=== Differential-test summary ==="
echo "corpus files         : ${#sources[@]}"
echo "functions seen       : $total_functions"
echo "clunk errors         : $total_clunk_errors"
if (( ALIVE_TV_AVAILABLE )); then
    echo "alive-tv checks run  : $total_checked"
    echo "alive-tv PASS        : $total_passed"
    echo "alive-tv FAIL        : $total_failed"
else
    echo "alive-tv checks run  : 0 (alive-tv not installed)"
    echo "alive-tv PASS        : 0 (soundness backstop DISABLED)"
    echo "alive-tv FAIL        : 0"
fi
echo
echo "Per-file verdicts:"
printf '  %-12s  %-22s  %s\n' "verdict" "reason" "file"
while IFS=$'\t' read -r verdict reason file; do
    [[ -z "$verdict" ]] && continue
    printf '  %-12s  %-22s  %s\n' "$verdict" "$reason" "$file"
done <"$summary"

# Exit non-zero iff a soundness bug was detected.
if (( total_failed > 0 )); then
    echo
    echo "ERROR: $total_failed alive-tv refinement failure(s) — SOUNDNESS BUG(S) detected." >&2
    exit 1
fi

exit 0
