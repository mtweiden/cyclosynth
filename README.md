# cyclosynth

Optimal-T-count synthesis of single-qubit unitaries into Clifford+T (and Clifford+√T) circuits, with Python bindings.

## What it does

Given a 2×2 target unitary `V` and a tolerance `ε`, the synthesizer returns a gate sequence `U` with diamond distance `d_diamond(U, V) < ε` and the (close to) smallest gate count achievable at that `ε`.

- **Clifford+T** (default).
  Gates `{H, S, T, X, Y, Z}`.
  Finds the *minimal T-count* circuit.
  Implements Algorithm 3.14 of [Morisaki et al., arXiv:2510.05816](https://arxiv.org/abs/2510.05816) — an 8-dimensional integer lattice enumeration (Alg. 3.6) over the ring `Z[ω]`, with a divide-and-conquer prefix split (Alg. 3.11).
- **Clifford+√T** (`sqrt_t=True`).
  Adds `Q = √T`, working in `Z[ζ]` on a 16-dimensional lattice.
  This denser gate set generally yields cheaper circuits; `optimize_cost=True` minimizes a weighted cost `T_count + c·Q_count` instead of returning the first solution.

## Install

Built with [maturin](https://www.maturin.rs/), which compiles the Rust extension and installs the `cyclosynth` module into the active environment.
The crate links system `gmp`/`mpfr`:

```sh
brew install gmp mpfr          # macOS  (Debian: apt-get install libgmp-dev libmpfr-dev)
pip install maturin
maturin develop --release
```

- **mpfr** (arbitrary-precision floating point) is the precision scaffolding of the lattice search.
  The search metric is severely ill-conditioned — its condition number grows like `ε⁻⁴` (≈ 2¹⁰⁷ at `ε = 1e-8`) — so hardware `f64` loses too many bits; MPFR keeps the LLL reduction and Schnorr-Euchner enumeration faithful at deep `ε`.
- **gmp** (arbitrary-precision integers) backs the exact integer Gram arithmetic when the fixed-width `i256` path would overflow at large lde, and is also the foundation MPFR itself is built on.

(`examples/verify.py` additionally needs `pip install mpmath` for its high-precision cross-check.)

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

if result:                        # None if no circuit was found within epsilon
    print(result.gates)           # gate string over {H, S, T, X, Y, Z}
    print(result.t_count)         # T-count (also .q_count, .cost, .lde)
    print(result.distance)        # diamond distance, < epsilon
```

The composition convention is *leftmost gate is the leftmost matrix factor*: for `"ABC"` the unitary is `A·B·C`.
`examples/verify.py` round-trips a gate string back to a unitary and re-checks the distance.

### Choosing Clifford+√T settings

The defaults (`optimize_cost=True`, `lde_window=2`) already minimize the circuit cost — the remaining knobs only trade a little cost for speed at deep `ε`.
The table below is medians over 30 random `U3` targets on Apple M-series, 8 threads; **cost** is `T + 3·Q` *relative to the Clifford+T circuit for the same target* (lower = cheaper):

| ε    | settings              | cost vs Clifford+T | wall (median) |
| ---- | --------------------- | ------------------ | ------------- |
| 1e-7 | defaults              | 0.89×              | 1.9 s         |
| 1e-7 | `deadline_ms=750`     | 0.90×              | 1.2 s         |
| 1e-8 | defaults              | 0.89×              | 21 s          |
| 1e-8 | `deadline_ms=6000`    | 0.92×              | 13 s          |
| 1e-8 | `deadline_ms=2000`    | 0.94×              | 5.4 s         |

- The √T advantage comes entirely from `optimize_cost` (on by default), which also floors the result at the Clifford+T cost — so it never costs *more* than Clifford+T.
  Setting `optimize_cost=False` returns the first circuit found instead: fast, but it can cost more than the Clifford+T baseline, so it is not recommended.
- `deadline_ms` caps the cost-optimization search — lower is faster and slightly more expensive.
  It only bites at deep `ε` (≲1e-7); above that the search self-terminates well before any reasonable deadline.
- `lde_window` (default 2) widens the depth band the optimizer considers; 2 was the cost-minimum at every `ε` tested, so leave it unless profiling your own target distribution.
- `q_cost` (default 3) is the Q-vs-T cost weight — set it if your hardware prices √T differently.
  `seq_parity` is a deep-`ε` reliability knob and does not change cost.

(Wall times vary widely target-to-target; see the [Performance](#performance) note.
These are guidance, not guarantees.)

## Examples (Python)

After `maturin develop --release`, the [`examples/`](examples/) directory shows the Python bindings in use:

| example | what it does |
|---|---|
| [`synth.py`](examples/synth.py) | Synthesize a `U3(α, β, γ)` target with both Clifford+T and Clifford+√T; read the result. |
| [`choosing_epsilon.py`](examples/choosing_epsilon.py) | Sweep ε to see the accuracy/T-count trade-off. |
| [`optimize_cost.py`](examples/optimize_cost.py) | The Clifford+√T `optimize_cost` knob: minimize `T + 3·Q`. |
| [`compare_t_vs_sqrtt.py`](examples/compare_t_vs_sqrtt.py) | Clifford+T vs Clifford+√T cost (`T + 3·Q`) on the same targets. |
| [`verify.py`](examples/verify.py) | Verification harness: independently re-check synthesized circuits at high precision (needs `mpmath`; runs 100 trials, slow). |

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

The `time_synthesis_omega` / `time_synthesis_zeta` binaries run the benchmark suites:

```sh
cargo run --release --bin time_synthesis_omega -- --threads 8 --trials 3
```

### Telemetry (`trace` feature)

Diagnostic telemetry — trace counters and per-phase timers — is **off by default**: `diag::trace_enabled()` is a compile-time `false`, so every instrumentation site (including the per-leaf hot path) is eliminated and the default build carries zero overhead.
The `probe_*` / `bench_*` bins and the `#[ignore]` telemetry tests need it enabled:

```sh
cargo run --release --features trace --bin probe_walk_bench_omega -- 0.7 1e-3 16
```

## How it works

The synthesizer tries circuits of increasing length, shortest first, and returns the first one that lands within `ε` of the target.
The hard part is finding *which* circuit of a given length (if any) is close enough:

- **Short circuits** are found by direct search over a small candidate set.
- **Longer circuits** are split into a fixed prefix and a remainder.
  For each prefix, finding the best remainder becomes a "closest point in a grid" problem, solved with lattice reduction.
  Many prefixes are searched in parallel, and the search stops as soon as one of them works.

This is the algorithm of [Morisaki et al.](https://arxiv.org/abs/2510.05816); see the paper for the full derivation.
The lattice code lives in `src/synthesis/lattice/`, with one variant per gate set — `omega/` for Clifford+T and `zeta/` for Clifford+√T.
Build with `--features trace` and set `CYCLOSYNTH_TRACE=1` to print per-step timings to stderr (see the Telemetry section above — the env var is inert without the feature).

## Glossary

Terms used throughout the code and docs:

- **T gate / Q gate** — `T` is the π/8 phase gate (the Clifford+T generator); `Q` is √T (the Clifford+√T generator).
  The circuit cost we minimize is `T_count + 3·Q_count`.
- **lde** ("least denominator exponent") — the power of √2 in a circuit's ring denominator; the synthesizer uses it as the search depth (`max_lde`).
  Deeper lde = more candidate circuits = tighter achievable `ε`.
- **Z[ω] / Z[ζ]** — the two number rings the lattice search runs in: `omega/` (8-dimensional, Clifford+T) and `zeta/` (16-dimensional, Clifford+√T).
- **Matsumoto-Amano (MA) prefix / FGKM** — the canonical "normal forms" the fixed circuit prefixes are enumerated from: Matsumoto-Amano for Clifford+T, and the FGKM form ([arXiv:1501.04944](https://arxiv.org/abs/1501.04944)) for Clifford+√T.
- **det-phase** — the determinant's root-of-unity class; used to prune prefixes that can't match the target up to global phase.
- **Lattice search internals** — **LLL** / **L²-LLL** (basis reduction), **Schnorr-Euchner** / SE (lattice-point enumeration), **BKZ** (block reduction), **SVP** (shortest-vector problem), **Cholesky** / CFA (factorization), **dd** (double-double, ~106-bit float).
  A fuller version of this list is in the `src/synthesis/` module docs.

## Repository layout

```
src/
├── lib.rs              PyO3 module + crate root
├── matrix/, rings/     U2/SO3 types; rings Z[ω], Z[ζ]
└── synthesis/
    ├── synthesizer.rs      top-level Synthesizer over both backends
    ├── clifford_t/         Clifford+T backend (8-D, Z[ω])
    ├── clifford_sqrt_t/    Clifford+√T backend (16-D, Z[ζ])
    ├── decomposer.rs       ring unitary → gate string
    ├── distance.rs         diamond distance via Frobenius identity
    └── lattice/            integer-lattice search
        ├── omega/              8-D pipeline (Clifford+T)
        ├── zeta/               16-D pipeline (Clifford+√T)
        └── common.rs           shared L²-LLL helpers
examples/               Python usage, verification, and comparison
```

Advanced/internal tuning is exposed through `CYCLOSYNTH_*` environment variables; the names are frozen and documented inline at their use sites in `src/synthesis/`.

## Testing

```sh
cargo test --release
```

## Performance

Per-target wall-clock from `time_synthesis_omega --trials 3` on Apple M-series, 8 threads, 10 random SU(2) targets per ε (fixed seed).
`T-count` is the range of resolved T-counts across the 10 targets; the time columns summarize the per-target minimums.

| ε    | T-count | min (ms) | median (ms) | max (ms) | mean (ms) |
| ---- | ------- | -------- | ----------- | -------- | --------- |
| 1e-2 | 12–18   | 0.4      | 0.4         | 0.5      | 0.4       |
| 1e-3 | 18–28   | 0.5      | 0.6         | 0.6      | 0.6       |
| 1e-4 | 30–38   | 0.6      | 1.0         | 1.9      | 1.2       |
| 1e-5 | 41–49   | 0.5      | 1.4         | 9.2      | 2.6       |
| 1e-6 | 54–58   | 0.8      | 10.9        | 21.4     | 9.8       |
| 1e-7 | 62–68   | 0.9      | 7.7         | 35.3     | 10.2      |
| 1e-8 | 72–78   | 20.0     | 364         | 961      | 425       |

