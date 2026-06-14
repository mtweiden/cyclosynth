# cyclosynth-rs

Pure-Rust, optimal-T-count synthesis of single-qubit unitaries into
Clifford+T (and Clifford+√T) circuits, with Python bindings.

## What it does

Given a 2×2 target unitary `V` and a tolerance `ε`, the synthesizer
returns a gate sequence `U` with diamond distance `d_diamond(U, V) < ε`
and the (close to) smallest gate count achievable at that `ε`.

- **Clifford+T** (default). Gates `{H, S, T, X, Y, Z}`. Finds the
  *minimal T-count* circuit. Implements Algorithm 3.14 of
  [Morisaki et al., arXiv:2510.05816](https://arxiv.org/abs/2510.05816)
  — an 8-dimensional integer lattice enumeration (Alg. 3.6) over the
  ring `Z[ω]`, with a divide-and-conquer prefix split (Alg. 3.11).
- **Clifford+√T** (`sqrt_t=True`). Adds `Q = √T`, working in `Z[ζ₁₆]`
  on a 16-dimensional lattice. This denser gate set generally yields
  cheaper circuits; `optimize_cost=True` minimizes a weighted cost
  `T_count + c·Q_count` instead of returning the first solution.

## Install

Built with [maturin](https://www.maturin.rs/), which compiles the Rust
extension and installs the `cyclosynth` module into the active
environment. The crate links system `gmp`/`mpfr`:

```sh
brew install gmp mpfr          # macOS  (Debian: apt-get install libgmp-dev libmpfr-dev)
pip install maturin
maturin develop --release
```

## Usage (Python)

```python
import numpy as np
import cyclosynth

# Any 2×2 np.complex128 unitary. Here: Rz(α)·Ry(β)·Rz(γ).
def rz(t): return np.diag([np.exp(-1j*t/2), np.exp(1j*t/2)])
def ry(t):
    c, s = np.cos(t/2), np.sin(t/2)
    return np.array([[c, -s], [s, c]], dtype=np.complex128)

target = rz(4.863069) @ ry(2.757718) @ rz(5.394728)

synth = cyclosynth.Synthesizer(epsilon=1e-5)
result = synth.synthesize(target)

print(result.gates)               # gate string over {H, S, T, X, Y, Z}, or None
print(result.gates.count("T"))    # T-count
print(result.distance)            # diamond distance, < epsilon
```

The composition convention is *leftmost gate is the leftmost matrix
factor*: for `"ABC"` the unitary is `A·B·C`. `examples/verify.py`
round-trips a gate string back to a unitary and re-checks the distance.

Constructor (all keywords optional):

```python
Synthesizer(epsilon, *, sqrt_t=False, max_lde=None, min_lde=None,
            optimize_cost=None, q_cost=None, lde_window=None,
            deadline_ms=None, seq_parity=None)
```

The `optimize_cost`, `q_cost`, `lde_window`, `deadline_ms`, and
`seq_parity` knobs apply to the Clifford+√T backend only.

## Benchmarking & comparison (Python)

The `scripts/` drivers sweep random `U3(α, β, γ)` targets across a range
of `ε`, time the synthesizer, and plot the results. Run them from the
repository root (they read/write under `scripts/data/`) after
`maturin develop --release`.

| script | what it does |
|---|---|
| [`comparison.py`](scripts/comparison.py) | Clifford+T sweep over `ε ∈ {1e-3 … 1e-8}`: T-count, mpmath-verified distance, wall time → `scripts/data/comparison_data.csv`. Optional [`trasyn`](https://pypi.org/project/trasyn/) cross-comparison (commented out). |
| [`comparison_sqrtt.py`](scripts/comparison_sqrtt.py) | Same targets, both backends, on a common cost scale `T + 3.5·Q` → `comparison_sqrtt_data.csv`. |
| [`plot_comparison.py`](scripts/plot_comparison.py) | Grouped-violin plot of the Clifford+T sweep. |
| [`plot_comparison_sqrtt.py`](scripts/plot_comparison_sqrtt.py) / [`plot_comparison_sqrtt_slope.py`](scripts/plot_comparison_sqrtt_slope.py) | Cost violin / paired per-target slope plots for the √T comparison. |

```sh
python scripts/comparison.py && python scripts/plot_comparison.py
python scripts/comparison_sqrtt.py && python scripts/plot_comparison_sqrtt.py
```

Edit the `EPSILONS` / `N_TRIALS` / `SEED` constants at the top of each
driver to change the sweep. Helpers `recompute_csv_cost.py` (re-derive
the cost column) and `profile_summary.py` (aggregate `CYCLOSYNTH_TRACE=1`
logs) round out the directory.

## Usage (Rust)

```rust
use cyclosynth::synthesis::Synthesizer;
use num_complex::Complex;

let theta = 0.3_f64;                         // Rz(0.3)
let target = [
    [Complex::from_polar(1.0, -theta / 2.0), Complex::new(0.0, 0.0)],
    [Complex::new(0.0, 0.0), Complex::from_polar(1.0, theta / 2.0)],
];

let synth = Synthesizer::new(1e-5, /* sqrt_t = */ false);
let result = synth.synthesize(target).unwrap();
let gates = result.gates.unwrap();
println!("T-count = {}, gates = {}", gates.matches('T').count(), gates);
```

The `time_synthesis_omega` / `time_synthesis_zeta` binaries run the
benchmark suites:

```sh
cargo run --release --bin time_synthesis_omega -- --threads 8 --trials 3
```

### Telemetry (`trace` feature)

Diagnostic telemetry — trace counters and per-phase timers — is **off by
default**: `diag::trace_enabled()` is a compile-time `false`, so every
instrumentation site (including the per-leaf hot path) is eliminated and the
default build carries zero overhead. The `probe_*` / `bench_*` bins and the
`#[ignore]` telemetry tests need it enabled:

```sh
cargo run --release --features trace --bin probe_walk_bench_omega -- 0.7 1e-3 16
```

## How it works

The synthesizer tries circuits of increasing length, shortest first, and
returns the first one that lands within `ε` of the target. The hard part
is finding *which* circuit of a given length (if any) is close enough:

- **Short circuits** are found by direct search over a small candidate set.
- **Longer circuits** are split into a fixed prefix and a remainder. For
  each prefix, finding the best remainder becomes a "closest point in a
  grid" problem, solved with lattice reduction. Many prefixes are searched
  in parallel, and the search stops as soon as one of them works.

This is the algorithm of
[Morisaki et al.](https://arxiv.org/abs/2510.05816); see the paper for the
full derivation. The lattice code lives in `src/synthesis/lattice/`, with
one variant per gate set — `omega/` for Clifford+T and `zeta/` for
Clifford+√T. Set `CYCLOSYNTH_TRACE=1` to print per-step timings to stderr.

## Repository layout

```
src/
├── lib.rs              PyO3 module + crate root
├── matrix/, rings/     U2/SO3 types; rings Z[ω], Z[ζ₁₆]
└── synthesis/
    ├── synthesizer.rs      top-level Synthesizer over both backends
    ├── clifford_t/         Clifford+T backend (8-D, Z[ω])
    ├── clifford_sqrt_t/    Clifford+√T backend (16-D, Z[ζ₁₆])
    ├── decomposer.rs       ring unitary → gate string
    ├── distance.rs         diamond distance via Frobenius identity
    └── lattice/            integer-lattice search
        ├── omega/              8-D pipeline (Clifford+T)
        ├── zeta/               16-D pipeline (Clifford+√T)
        └── common.rs           shared L²-LLL helpers
scripts/                Python benchmark drivers + plotting
examples/               Python usage + verification helpers
```

Advanced/internal tuning is exposed through `CYCLOSYNTH_*` environment
variables; the names are frozen and documented inline at their use
sites in `src/synthesis/`.

## Testing

```sh
cargo test --release
```

`verify_correctness_at_1e_X_*` round-trips a synthesized circuit through
the decomposer and asserts `d_diamond(rebuilt, target) < ε`.

## Performance

Per-target wall-clock from `time_synthesis_omega --trials 3` on Apple
M-series, 8 threads, 10 random SU(2) targets per ε (fixed seed). `T-count`
is the range of resolved T-counts across the 10 targets; the time columns
summarize the per-target minimums.

| ε    | T-count | min (ms) | median (ms) | max (ms) | mean (ms) |
| ---- | ------- | -------- | ----------- | -------- | --------- |
| 1e-2 | 12–18   | 0.4      | 0.4         | 0.5      | 0.4       |
| 1e-3 | 18–28   | 0.5      | 0.6         | 0.6      | 0.6       |
| 1e-4 | 30–38   | 0.6      | 1.0         | 1.9      | 1.2       |
| 1e-5 | 41–49   | 0.5      | 1.4         | 9.2      | 2.6       |
| 1e-6 | 54–58   | 0.8      | 10.9        | 21.4     | 9.8       |
| 1e-7 | 62–68   | 0.9      | 7.7         | 35.3     | 10.2      |
| 1e-8 | 72–78   | 20.0     | 364         | 961      | 425       |

(Sum of per-target minimums across all 70 runs: ≈ 4.5 s.)

Times vary widely at the same ε and are not monotonic in ε. The cost is
dominated by how many prefixes the search tries before one succeeds, which
depends on where the target falls relative to the lattice. Some angles —
e.g. `π/n` for small `n` — sit in sparse regions and need longer circuits,
a property of the approximation problem rather than the implementation.
