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
# beat_o3.sh — measure whether clunk produces stronger results AFTER clang -O3.
#
# For every C source in the corpus it:
#   1. emits the clang -O3 baseline IR (scalar/integer, no vectorize/unroll so
#      the output stays inside clunk's parser + SMT-modelable subset),
#   2. runs clunk on that ALREADY-optimised IR (--report-json),
#   3. aggregates the per-function results into a table.
#
# A function counts as "stronger after -O3" iff clunk adopted a strictly
# cheaper candidate that the SMT verifier PROVED equivalent to the -O3 input
# (improvement_ratio > 1 AND verified == true). Declarations (rounds_run == 0)
# are ignored. The script exits non-zero if any function REGRESSED
# (improvement_ratio < 1), which should never happen — the pipeline only ever
# adopts improvements.
#
# Usage:
#   scripts/beat_o3.sh [corpus_dir] [work_dir]
#
# Env overrides: CLUNK, CLANG, CLUNK_FLAGS.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CORPUS_DIR="${1:-$REPO_ROOT/benchmarks/corpus}"
WORK_DIR="${2:-$REPO_ROOT/build/beat_o3}"
CLUNK="${CLUNK:-$REPO_ROOT/build/clunk}"
CLANG="${CLANG:-clang}"
# Keep -O3 output scalar & loop-free so it stays SMT-modelable.
O3_FLAGS=(-O3 -fno-vectorize -fno-slp-vectorize -fno-unroll-loops
          -Xclang -disable-O0-optnone -emit-llvm -S)
# clunk: full search, verify with SMT, short per-function budget.
read -r -a CLUNK_FLAGS <<< "${CLUNK_FLAGS:---opt-level 3 --time-budget 8}"

if [[ ! -x "$CLUNK" ]]; then
    echo "error: clunk binary not found/executable at '$CLUNK' (build it first, or set \$CLUNK)" >&2
    exit 2
fi
if ! command -v "$CLANG" >/dev/null 2>&1; then
    echo "error: clang not found (set \$CLANG)" >&2
    exit 2
fi

mkdir -p "$WORK_DIR"
echo "corpus : $CORPUS_DIR"
echo "clunk  : $CLUNK ${CLUNK_FLAGS[*]}"
echo "clang  : $($CLANG --version | head -1)"
echo

shopt -s nullglob
sources=("$CORPUS_DIR"/*.c)
if [[ ${#sources[@]} -eq 0 ]]; then
    echo "error: no .c sources in $CORPUS_DIR" >&2
    exit 2
fi

json_files=()
for src in "${sources[@]}"; do
    base="$(basename "${src%.c}")"
    o3_ll="$WORK_DIR/${base}_o3.ll"
    opt_ll="$WORK_DIR/${base}_clunk.ll"
    json="$WORK_DIR/${base}.json"

    echo ">> $base : emitting -O3 baseline"
    "$CLANG" "${O3_FLAGS[@]}" "$src" -o "$o3_ll"

    echo ">> $base : running clunk"
    # JSON report on stdout; optimised IR to file; clunk chatter to stderr log.
    "$CLUNK" "${CLUNK_FLAGS[@]}" --report-json --output "$opt_ll" "$o3_ll" \
        >"$json" 2>"$WORK_DIR/${base}.log" || {
            echo "error: clunk failed on $base (see $WORK_DIR/${base}.log)" >&2
            exit 1
        }
    json_files+=("$json")
done

echo
python3 "$REPO_ROOT/scripts/beat_o3_report.py" "${json_files[@]}"
