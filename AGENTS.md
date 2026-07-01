# AGENTS.md

Orientation for AI agents using `cyclosynth`: a pure-Rust library (with a Python
API via PyO3) that synthesizes a single-qubit target unitary into a
**Clifford+T** or **Clifford+√T** circuit by Lenstra-style lattice search
(arXiv:2510.05816). Human-facing intro is in [README.md](README.md).

## Build & install

```sh
cargo build --release                 # Rust library + bins
cargo test  --release --lib           # test suite
maturin develop --release             # build + install the Python module (pip install maturin first)
```

- System deps: `gmp`/`mpfr` (`brew install gmp mpfr`, or `apt-get install
  libgmp-dev libmpfr-dev`), plus `m4` — `gmp-mpfr-sys` compiles them from source.
- The toolchain is pinned in `rust-toolchain.toml` for numerical
  reproducibility; don't bump it casually.
- Build the Python module with `maturin`, not `cargo` — plain `cargo build
  --features python` fails at the link step (the known PyO3-without-maturin
  libpython issue, not a code bug). Use `cargo check --features python` to
  validate compilation only.
- CI (`.github/workflows/ci.yml`) runs build + test + `cargo clippy --release
  --all-targets -- -D warnings` (and again with `--features trace`); keep green.

## Use it

Python:

```python
import cyclosynth
synth  = cyclosynth.Synthesizer(1e-5, sqrt_t=True)   # omit sqrt_t for Clifford+T
result = synth.synthesize(target)                    # target: 2x2 complex unitary (numpy)
if result:                                           # None if no circuit within epsilon
    result.gates       # gate string over {H, S, T, Q(=√T), X, Y, Z}; lowercase q/t/s = Q†/T†/S†
    result.t_count, result.q_count, result.cost, result.lde, result.distance
```

Every returned circuit satisfies `distance < epsilon` (diamond distance).
Runnable examples are in `examples/` (`synth.py`, `compare_t_vs_sqrtt.py`,
`optimize_cost.py`, `choosing_epsilon.py`, `verify.py`).

Rust: `synthesis::Synthesizer::new(epsilon, sqrt_t).synthesize(target)`, with
builder options (`with_optimize_cost`, `with_q_cost`, `with_max_lde`, …).

## Good to know

- **Deep ε is slow** (≤1e-7 is seconds-to-minutes per target); iterate and test
  at coarse ε (1e-2/1e-3). `#[ignore]` tests are diagnostic probes, not the
  default suite.
- **Changes to the lattice / Schnorr-Euchner / precision hot path
  (`src/synthesis/lattice/`, `clifford_t/mod.rs`) MUST pass the deep-ε canary
  before commit:** `cargo test --release -- --ignored deep_eps_canary` (~5s).
  Shallow-ε (≤1e-8) tests are **not** sufficient — deep-ε mis-pruning at
  ε=1e-10 (where √κ(Q)≈2^68 crosses the f64/i128 boundary) silently *misses*
  solutions and reads as slowness, not an error. The canary asserts a
  known target finds at its exact minimum lde; a miss trips it loudly.
- `CYCLOSYNTH_*` env vars are A/B kill-switches; some (e.g.
  `CYCLOSYNTH_BOUND_SQ`) change search *results*, not just performance.
- For work *inside* the code, read the domain glossary at the top of
  `src/synthesis/mod.rs` first. The two lattice backends (`src/synthesis/lattice/omega/`
  — 8D, f64; `zeta/` — 16D, MPFR) are deliberately kept parallel but separate;
  don't try to merge them.
- `docs/`, `scripts/`, `bench_logs/`, `target/` are gitignored.
