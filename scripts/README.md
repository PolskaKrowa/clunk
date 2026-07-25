# beat-O3 harness

Measures whether clunk produces **stronger results after clang `-O3`** — the honest,
falsifiable version of "can clunk beat `-O3`?".

## Run

```sh
cmake --build build --target clunk      # build the CLI first
./scripts/beat_o3.sh                     # uses benchmarks/corpus/*.c
```

For each C source it emits the `clang -O3` baseline IR (scalar/integer, no
vectorize/unroll so the output stays inside clunk's parser + SMT-modelable subset),
runs `clunk --opt-level 3 --report-json` on that already-optimised IR, and prints a
per-function table. A function is **"stronger after -O3"** iff clunk adopted a
strictly-cheaper candidate the SMT verifier **proved equivalent** to the `-O3` input
(`improvement_ratio > 1` and `verified == true`). The script exits non-zero on any
regression.

## Pieces

- `benchmarks/corpus/*.c` — small, loop-free, integer kernels (SMT-modelable).
- `clunk --report-json` — emits per-function `{score_original, score_optimised,
  improvement_ratio, verified, ...}` as a JSON array on stdout (optimised IR then
  goes only to `--output`). Implemented in `src/cli/main.cpp`.
- `scripts/beat_o3.sh` — driver: build baselines, run clunk, aggregate.
- `scripts/beat_o3_report.py` — table + verdict from the JSON reports.

## Current baseline

With clunk's present transformation set (a subset of what `-O3` already does), the
harness reports **0 improvements** over `-O3` output — the expected result and the
motivation for the SMT-verified peephole miner (Stage 1). Re-run with mined patterns
loaded (`--pattern-library`) to see the delta.

