# Clunk

**Clunk** is an LLVM IR superoptimiser. It takes LLVM IR (`.ll` files) and
searches for instruction-level optimisations using stochastic search,
evolutionary algorithms, SMT verification (via Z3), e-graph rewriting, and
peephole mining. Written in C++17 and built with CMake.

Version: v0.1.0

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

## Building

### Requirements

- CMake 3.20 or later
- A C++17 compiler (GCC or Clang)
- Optional: Z3 shared library (runtime, for SMT verification)
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
| `CLUNK_ENABLE_TESTS` | ON | Build test suite |
| `CLUNK_ENABLE_BENCHMARKS` | ON | Build benchmarks |
| `CLUNK_ENABLE_GPU` | ON | Enable GPU/PTX optimisation stubs |
| `CLUNK_ENABLE_LTO` | ON | Link-time optimisation for Release builds |
| `CLUNK_ENABLE_NATIVE` | OFF | Compile with `-march=native` (non-portable) |

## Usage

```
clunk [options] <input.ll>
```

### Key options

| Option | Description |
|--------|-------------|
| `--opt-level <N>` | Optimisation aggressiveness (default: 2) |
| `--time-budget <sec>` | Wall-clock time limit for the search (default: 30s) |
| `--output <file>` | Write optimised IR to a file |
| `--verbose` | Increase output verbosity |
| `--mine` | Run peephole pattern mining |
| `--pattern-library <path>` | Path to a pattern library file |
| `--report-json` | Write a JSON report of the search to stdout |
| `--no-z3` | Disable SMT verification |
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

Contributions are welcome. Please open an issue or pull request on GitHub.
