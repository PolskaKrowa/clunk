<div align=center>
<img width="512" height="512" alt="Clunk logo" src="https://github.com/user-attachments/assets/dc2363aa-d936-4fd7-a952-4b752861e908" />

<br>
<i>With a good set of tools, even anvils can move fast.</i>
<br>
<br>

[![License: GPL v3+](https://img.shields.io/badge/License-GPLv3%2B-blue.svg)](LICENSE)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://en.cppreference.com/w/cpp/17)
[![Z3](https://img.shields.io/badge/SMT-Z3-orange.svg)](https://github.com/Z3Prover/z3)
[![Version](https://img.shields.io/badge/version-v0.1.0-lightgrey.svg)](#)

</div>

**Clunk** is an LLVM IR superoptimiser. It takes LLVM IR (`.ll` files) and
searches for instruction-level optimisations using stochastic search,
evolutionary algorithms, SMT verification (via Z3), e-graph rewriting, and
peephole mining. Written in C++17 and built with CMake.

>[!NOTE]
> This program is ***NOT*** intended to override LLVM's existing optimisation passes.
> Instead, it is to provide *additional* optimisations that clang/llvm-opt have missed
> for larger programs using atypical coding styles.

Version: v0.1.0

## Example

```llvm
; input.ll (derived from examples/cross_fn.ll)

define internal i32 @double(i32 %x) {
entry:
  %r = shl i32 %x, 1
  ret i32 %r
}

; Multi-block callee: double if x>=0 else 0
define internal i32 @double_if_pos(i32 %x) {
entry:
  %cmp = icmp sge i32 %x, 0
  br i1 %cmp, %then, %else
then:
  %r = call i32 @double(i32 %x)
  ret i32 %r
else:
  ret i32 0
}

; Both callers pass the constant 5 → IPCP should specialise.
define i32 @caller_a() {
entry:
  %c = call i32 @double_if_pos(i32 5)
  ret i32 %c
}

define i32 @caller_b() {
entry:
  %c = call i32 @double_if_pos(i32 5)
  ret i32 %c
}

; Dead function — never called.
define internal i32 @dead_helper(i32 %x) {
entry:
  %r = add i32 %x, 1
  ret i32 %r
}

define i32 @main() {
entry:
  %a = call i32 @caller_a()
  %b = call i32 @caller_b()
  %s = add i32 %a, %b
  ret i32 %s
}
```

<div align=center>VVV</div>

```bash
$ clunk input.ll -o output.ll


=== Clunk Summary ===
Functions processed: 5
Functions optimised: 1
Average improvement:  5.53571x
Total time:           553.588 ms

$ █
```

<div align=center>VVV</div>

```llvm
; output.ll

; ModuleID = ''
define internal i32 @double(i32 %x) {
entry:
  %r = shl i32 %x, 1
  ret i32 %r
}

define i32 @caller_a() {
entry:
  %c = call i32 @double_if_pos.ipcp_0_5(i32 5)
  ret i32 %c
}

define i32 @caller_b() {
entry:
  %c = call i32 @double_if_pos.ipcp_0_5(i32 5)
  ret i32 %c
}

define i32 @main() {
entry:
  %a = call i32 @caller_a()
  %b = call i32 @caller_b()
  %s = add i32 %a, %b
  ret i32 %s
}

define internal i32 @double_if_pos.ipcp_0_5(i32 %x) {
entry:
  br label %then
then:
  ret i32 10
}
```

Clunk found 2 dead functions, 1 interprocedural constant and an inlineable call site; removed 1 unreachable block; folded 1 known-constant value, a comarison and a branch, verified it against the
original with Z3 + Alive2, and rewrote the IR accordingly, All while evaluating each function with their own thread (the input program was small enough to allow for that, larger programs with many functions may assign multiple functions per thread, allowing clunk to quickly find cross-function optimisation opportunities, if any.)

A Known limitation of clunk is that it's unable to pre-process arithmetical operations made within a program. LLVM-opt is known for its ability to do so, so the above program is *not* optimal.
The "optimal" code would be to simply have `@main()` return 20, and have every preceding function before it stripped out for dead code. We're currently looking at ways for clunk to quickly step through
the input code and find instances where a function (or a set of functions, collectively) returns a singular constant, or can be simplified to a single operation involving both a constant and a few operations.

We understand that creating an optimisation step that does this to the extreme may result in some programs (especially those designed to use a specific algorithm, or a set thereof, to compute numerical constants) to simply
contain the fully computed constant in a variable and return it, since an optimiser like this would have no idea whether a program is explicitly intended for computational stress-testing or as a single-use constant
calculator that always returns the same value, albeit at differing precision or accuracy.

One future addition to Clunk will be to add comments to the optimised IR which shows exactly what clunk did to optimise the code, making it easier for LLVM developers to write stronger optimisation passes.

## Features

- **Stochastic search** -- simulated annealing over instruction-level
  transformations.
- **Evolutionary search** -- tournament selection, crossover, and mutation
  over program variants.
- **SMT-based equivalence verification** -- Z3 integration (loaded at
  runtime via `dlopen`, no link-time dependency) to prove candidate
  transformations preserve program semantics.
- **E-graph rewriting** -- principled term rewriting with equality saturation
  for exploring equivalent program forms.
- **Peephole pattern mining** -- automatic discovery of optimisation
  patterns using CEGIS (counter-example guided inductive synthesis).
- **Vector synthesis** -- automatic discovery of SIMD intrinsic replacements.
- **Loop optimisation** -- loop-aware transformations and analysis.
- **Memory optimisation** -- memory access pattern improvements.
- **Cost model evaluation** -- TTI-based and MCA-based (llvm-mca) cost
  estimation to rank candidate programs.
- **GPU/PTX optimisation stubs** -- PTX emitter, occupancy model, divergence
  analysis, liveness analysis, and kernel launch optimisation.
- **Interpreter-based evaluation** -- fast program evaluation with caching
  for fitness assessment during search.

>[!WARNING]
> **E-graph rewriting is an experimental pass and is only intended for small-scale programs.**
>
> Instances where E-graphs return non-functional LLVM code should **NOT** be reported as a bug, as it's already a known issue.
> It is therefore disabled by default and can be re-enabled with the `--egraph` flag.

## Building

### Requirements

- CMake 3.20 or later
- A C++17 compiler (GCC or Clang)
- Optional: Z3 + Alive2 shared libraries (loaded at runtime)
- Optional: llvm-mca (for MCA-based cost model)

### Build instructions

```sh
mkdir build && cd build
cmake .. -DCLUNK_ENABLE_TESTS=ON -DCLUNK_ENABLE_BENCHMARKS=ON
cmake --build . -j$(nproc)
ctest --output-on-failure
```

### CMake options

| Option | Default | Description |
|--------|---------|-------------|
| `CLUNK_ENABLE_Z3` | ON | Enable Z3 SMT verification (runtime dlopen) |
| `CLUNK_ENABLE_ALIVE2` | ON | Enable Alive2 code verification (runtime dlopen) |
| `CLUNK_ENABLE_TESTS` | ON | Build test suite |
| `CLUNK_ENABLE_BENCHMARKS` | ON | Build benchmarks |
| `CLUNK_ENABLE_GPU` | ON | Enable GPU/PTX optimisation stubs |
| `CLUNK_ENABLE_LTO` | ON | Link-time optimisation for Release builds |
| `CLUNK_ENABLE_NATIVE` | OFF | Compile with `-march=native` (non-portable) |
| `CLUNK_ENABLE_GC_SECTIONS` | ON | Compile with `-ffunction-sections -fdata-sections -Wl,--gc-sections` (more effective dead-code stripping from clunk) |

## Usage

```
clunk [options] <input.ll>
```

### Key options

| Option | Description |
|--------|-------------|
| `--opt-level <N>` | Optimisation aggressiveness (default: max) |
| `--time-budget <sec>` | Wall-clock time limit for the search (default: 30s) |
| `--output <file>` | Write optimised IR to a file |
| `--verbose` | Increase output verbosity |
| `--mine` | Run peephole pattern mining |
| `--pattern-library <path>` | Path to a pattern library file |
| `--report-json` | Write a JSON report of the search to stdout |
| `--no-z3` | Disable SMT verification |
| `--no-alive2` | Disable Alive2 verification (not recommended, as I don't fully trust clunk's codegen fully yet) |
| `--no-gpu` | Disable GPU/PTX passes |
| `--no-miner` | Disable peephole miner |
| `--no-vector-synth` | Disable vector synthesis |
| `--mca` | Enable MCA-based cost model ranking |

## Project Structure

```
clunk/
  CMakeLists.txt          -- Top-level build configuration
  cmake/                  -- CMake modules (ClunkOpt, ClunkConfig)
  src/
    cli/                  -- Command-line interface
    IR/                   -- LLVM IR data structures and utilities
    Analysis/             -- Program analyses (known bits, dataflow)
    Search/               -- Search strategies, SMT verifier, e-graphs, mining
    Evaluator/            -- Cost models, interpreter, evaluation engine
    GPU/                  -- PTX emitter, occupancy, divergence, liveness
    Pattern/              -- Pattern library management
    Parser/               -- LLVM IR (.ll) parser
  include/clunk/          -- Public headers (mirrors src/ layout)
  tests/                  -- Unit tests
  benchmarks/             -- Benchmarks and corpus files
  scripts/                -- Helper scripts (diff testing, O3 comparison)
  vendor/                 -- Vendored dependencies
  examples/               -- Example .ll files
```

## License

Clunk is released under the [GNU General Public License v3 or later](LICENSE).

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for how to
build from source, run the test suite, and submit a pull request. Please
open an issue or pull request on GitHub.