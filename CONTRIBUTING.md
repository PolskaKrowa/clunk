# Contributing to Clunk

Thanks for your interest in contributing! Clunk is a fairly deep project
(search strategies, SMT, e-graphs, GPU passes, an IR parser...) so this
guide is meant to get you from clone to first PR without having to read
the whole codebase first.

## Getting set up

### Requirements

- CMake 3.20+
- A C++17 compiler (GCC or Clang)
- Optional: the Z3 and Alive2 shared libraries, for SMT/code verification (loaded at
  runtime via `dlopen`, so you don't need it just to build)
- Optional: `llvm-mca`, for MCA-based cost model work

### Build and test

```sh
git clone https://github.com/PolskaKrowa/clunk.git
cd clunk
mkdir build && cd build
cmake .. -DCLUNK_ENABLE_TESTS=ON -DCLUNK_ENABLE_BENCHMARKS=ON
cmake --build . -j$(nproc)
ctest --output-on-failure
```

If you're only working on one subsystem, turning off the others (e.g.
`-DCLUNK_ENABLE_GPU=OFF`) will speed up your build loop. See the CMake
options table in the [README](README.md) for the full list.

### Sanity-checking a change

Before opening a PR, it's worth running a quick end-to-end check against
one of the example files:

```sh
./build/clunk --opt-level 2 examples/<some-file>.ll -o /tmp/out.ll
```

and, if your change could affect correctness, running the diff-testing /
`-O3` comparison helpers in `scripts/` against a few files in
`benchmarks/`.

## Where to start

If you're not sure where to dig in:

- Check the issue tracker for anything labelled **good first issue**.
- The `GPU/` module is explicitly marked as stubs in the README — there's
  likely low-risk, well-scoped work there (occupancy modelling, divergence
  analysis, etc.) that doesn't require touching the search core.
- Peephole pattern mining (`Search/`, CEGIS-based) is a good place for
  contributions of new pattern candidates without needing to understand
  the whole search loop.
- If nothing's labelled and you want to add a feature, please open an
  issue first to discuss the approach before sending a large PR — this is
  especially true for anything touching the SMT verifier or e-graph
  rewriting, since correctness there matters a lot.

## Code style

- C++17, matching the style of the surrounding file you're editing.
- Prefer small, focused PRs over large ones — easier to review, easier to
  bisect if something regresses.
- New transformations/passes should come with a test under `tests/` and,
  where relevant, a corpus file under `benchmarks/`.
- Any change to a rewrite rule or transformation should be checked against
  the Z3 verifier where possible — correctness bugs in a superoptimiser
  are silent miscompiles, which are worse than a missed optimisation.

## Submitting a change

1. Fork the repo and create a branch off `main`.
2. Make your change, with tests.
3. Run `ctest --output-on-failure` locally and make sure it's green.
4. Open a pull request describing what changed and why. If it fixes an
   open issue, reference it (`Fixes #123`).
5. Be responsive to review comments — for anything touching correctness
   (SMT, e-graphs, cost models) expect a closer look than for, say, CLI
   flag additions.

## Reporting bugs

Please include:

- The `.ll` input that triggers the issue (minimal repro if possible)
- The command line you ran
- What you expected vs. what happened
- Clunk version / commit hash

Miscompiles (Clunk producing IR that isn't actually equivalent to the
input) are the highest-priority class of bug — please flag these clearly
in the issue title.

## Questions

Open an issue, or start a discussion thread if the repo has GitHub
Discussions enabled. Happy hacking!