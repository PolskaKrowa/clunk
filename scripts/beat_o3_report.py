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

#!/usr/bin/env python3
"""Aggregate clunk --report-json outputs into a beat-O3 summary table.

Reads one or more JSON files (each an array of per-function results from
`clunk --report-json`) and prints a table plus a verdict. Exits non-zero if
any function regressed (improvement_ratio < 1 - eps), which the pipeline
should never produce.
"""
import json
import sys

EPS = 1e-9


def main(paths):
    rows = []
    for p in paths:
        with open(p) as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                print(f"error: {p}: bad JSON ({e})", file=sys.stderr)
                return 2
        for fr in data:
            # Skip declarations / pass-throughs (nothing was searched).
            if fr.get("rounds_run", 0) == 0:
                continue
            rows.append(fr)

    if not rows:
        print("No searchable functions found (all declarations?).")
        return 0

    rows.sort(key=lambda r: r["function"])
    name_w = max(len(r["function"]) for r in rows)
    name_w = max(name_w, len("function"))

    print(f"{'function':<{name_w}}  {'score_-O3':>11}  {'score_clunk':>11}  "
          f"{'ratio':>7}  {'verified':>8}  note")
    print("-" * (name_w + 55))

    improved = improved_verified = regressed = 0
    for r in rows:
        ratio = r["improvement_ratio"]
        verified = r["verified"]
        note = ""
        if ratio > 1 + EPS:
            improved += 1
            if verified:
                improved_verified += 1
                note = "STRONGER (proved)"
            else:
                note = "stronger (unproved)"
        elif ratio < 1 - EPS:
            regressed += 1
            note = "REGRESSION"
        print(f"{r['function']:<{name_w}}  {r['score_original']:>11.4g}  "
              f"{r['score_optimised']:>11.4g}  {ratio:>7.4f}  "
              f"{str(verified):>8}  {note}")

    total = len(rows)
    print()
    print(f"functions searched      : {total}")
    print(f"improved (any)          : {improved}")
    print(f"improved AND SMT-proved : {improved_verified}   <-- 'stronger after -O3'")
    print(f"regressions             : {regressed}")

    if regressed:
        print("\nFAIL: at least one function regressed below the -O3 baseline.",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: beat_o3_report.py <report.json> [...]", file=sys.stderr)
        sys.exit(2)
    sys.exit(main(sys.argv[1:]))
