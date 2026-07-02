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
# One-shot: the qiskit U-gate family (sqrt_t=True selects Clifford+√T).
result = cyclosynth.synthesize_u1("pi/64", 1e-10)    # also synthesize_u2(φ, λ, ε),
                                                     # synthesize_u3(θ, φ, λ, ε)
# Reusable instance + ZYZ Euler targets + tuning knobs:
synth  = cyclosynth.Synthesizer(1e-5, sqrt_t=True)   # omit sqrt_t for Clifford+T
# Targets are specified by ANGLES, not matrices — the input must carry more
# than f64 precision for deep ε (cos/sin are evaluated to the search precision).
result = synth.synthesize_zyz(alpha, beta, gamma)    # SU(2) = Rz(α)·Ry(β)·Rz(γ)
# or synth.synthesize_u3(theta, phi, lam)            # qiskit/bqskit U3 convention
# angles are floats (radians) or exact 'pi'-strings ("pi/64", "3*pi/4")
if result:                                           # None if no circuit within epsilon
    result.gates       # gate string over {H, S, T, Q(=√T), X, Y, Z}; lowercase q/t/s = Q†/T†/S†
    result.t_count, result.q_count, result.cost, result.lde, result.distance
```

Every returned circuit satisfies `distance < epsilon` (diamond distance).
ε-range policy (Python layer only): `sqrt_t=True` raises `ValueError` below
ε = 1e-8; `sqrt_t=False` warns below ε = 1e-10 and proceeds.
Runnable examples are in `examples/` (`compare_t_vs_sqrtt.py`,
`compare_t_vs_sqrtt_gates.py`).

Rust: `synthesis::Synthesizer::new(epsilon, sqrt_t)`, then `synthesize_with_exact_col`
(deep-ε, exact MPFR column from angles) or `synthesize(target)` (f64 matrix,
ε ≥ 1e-8 only); builder options (`with_optimize_cost`, `with_q_cost`, `with_max_lde`, …).

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
- For work *inside* the code, read the domain glossary at the top of
  `src/synthesis/mod.rs` first. The two lattice backends (`src/synthesis/lattice/omega/`
  — 8D, f64; `zeta/` — 16D, MPFR) are deliberately kept parallel but separate;
  don't try to merge them.
- `docs/`, `scripts/`, `bench_logs/`, `target/` are gitignored.

## Environment variables

Two kinds: **kill-switch** (production default stays on; the var restores the
pre-change behavior for A/B bisection) and **probe** (experiment instrument;
some change search *results*, not just performance — never set in production).

| Variable | Default | Effect | Kind |
| --- | --- | --- | --- |
| `CYCLOSYNTH_TRACE` | off | `=1` enables telemetry counters + stderr trace (needs a `--features trace` build; must be set before first synthesis) | probe |
| `CYCLOSYNTH_SHELL_FILTER` | on | `=0` disables the 8D norm-shell discriminant prune (~1.4× at ε=1e-10) | kill-switch |
| `CYCLOSYNTH_CHUNK` | 64 | rayon `with_max_len` chunk size for the 8D prefix sweep | probe |
| `CYCLOSYNTH_PREDICTIVE_TRUNC` | on | `=0` disables the projected-infeasibility abort in budget-capped 16D flat walks | kill-switch |
| `CYCLOSYNTH_OMEGA_FORCE_EXACT` | off | `=1` forces the deep-ε exact MPFR alignment path (8D) at any ε (exact-vs-f64 equivalence checks) | probe |
| `CYCLOSYNTH_BOUND_SQ` | 1.5 (3.0 on MPFR-Cholesky fallback) | overrides the 16D SE Q-bound; read per call; changes results | probe |
| `CYCLOSYNTH_SE_BOUND_8D` | 1.51 | overrides the 8D SE bound; LazyLock — must be set before the first synthesis; changes results | probe |
| `CYCLOSYNTH_L_COSET` | ε rule | `=0`/`=1` force plain phase-dedup / coset dedup of 8D MA prefixes | kill-switch |
| `CYCLOSYNTH_ZETA_COSET` | on | `=0` disables 16D prefix right-coset dedup | kill-switch |
| `CYCLOSYNTH_BKZ` | 0 (off) | ambient BKZ-β post-pass block size for the √T backend; the `with_bkz` builder overrides it | probe |
| `CYCLOSYNTH_FRONTIER_QUEUE` | on | `=0` restores chunked frontier dispatch (vs floor-priority pull-queue) | kill-switch |

These names are frozen; new experiment toggles get removed (var + losing code
arm) once the experiment concludes.
