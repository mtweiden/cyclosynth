# AGENTS.md

Orientation for AI agents working in `cyclosynth`. (Human-facing intro is in
[README.md](README.md); this file holds the build/test details and the
non-obvious facts needed to navigate the codebase.)

## What this is

A pure-Rust library that synthesizes a single-qubit target unitary into a
**Clifford+T** or **Clifford+√T** circuit, minimizing `T_count + 3·Q_count`
(arXiv:2510.05816). It exposes a Python API via PyO3. The hard part is a
Lenstra-style lattice search (LLL + Schnorr-Euchner enumeration).

The entry point is `synthesis::Synthesizer`. Read the **domain glossary** at the
top of `src/synthesis/mod.rs` first — it defines the vocabulary (T/Q gates, lde,
Z[ω]/Z[ζ₁₆], MA/FGKM, LLL/SE/BKZ/CFA/dd) used everywhere.

## Build / test / run

```sh
cargo build --release                              # library + bins
cargo test  --release --lib                        # the test suite
cargo test  --release --lib --features trace       # diag-counter tests need this
cargo clippy --release --all-targets -- -D warnings
cargo clippy --release --all-targets --features trace -- -D warnings
```

CI (`.github/workflows/ci.yml`) runs exactly the above — keep all of them green.
System deps: `gmp`/`mpfr` (`brew install gmp mpfr` / `apt-get install libgmp-dev
libmpfr-dev`); `gmp-mpfr-sys` compiles them from source and needs `m4`. The
toolchain is pinned in `rust-toolchain.toml` for numerical reproducibility — don't
bump it casually.

Python: `pip install maturin && maturin develop --release`. (Plain
`cargo build --features python` fails at the *link* step — that's the known
PyO3-without-maturin libpython issue, not a code bug; use `cargo check
--features python` to validate compilation.)

## Architecture

Two backends under `src/synthesis/lattice/`, deliberately kept **parallel but
separate**:
- `omega/` — Z[ω], 8-dimensional, Clifford+T, **f64** Gram-Schmidt.
- `zeta/` — Z[ζ₁₆], 16-dimensional, Clifford+√T, **MPFR** Gram-Schmidt.
- `common.rs` — the genuinely dimension-independent core (const-generic i256
  Gram kernels, LLL params, precision helpers), shared by both.

The dimension is a **compile-time constant** and the GS scalar type differs by
backend — this is why the two are not merged into one generic. Don't try to
unify them; an earlier audit confirmed the synthesis algorithms (MA-prefix D&C
vs FGKM-prefix) genuinely differ. `lattice/backend.rs` has a thin
`LatticeBackend` contract; it's intentionally narrow.

Higher layers: `clifford_t/` and `clifford_sqrt_t/` (the per-gate-set synthesis
strategies), `synthesizer.rs` (the unified `Backend{T,Q}` enum wrapper + PyO3),
`decomposer.rs`/`cliffords.rs`/`cost_bound.rs`/`distance.rs`, and `rings/` +
`matrix/` (exact ring/matrix types).

## Things to know

- **Float types**: `Float` = `f64` (hardware), `MpFloat` = `rug::Float` (MPFR,
  arbitrary precision — not a fixed-width "f128"). Those are the only two.
- **Telemetry is `trace`-feature-gated and compile-time-zero-cost when off**
  (`diag::trace_enabled()` is `const false` without `--features trace`), so the
  default build carries none. LLL is ~99% of CPU at deep ε.
- **The hot path is the Schnorr-Euchner walk (`*/se.rs`) and the LLL.** The f64
  Gram-Schmidt path was measured to give no speedup (removed), but the
  per-prefix f64 Cholesky is ~6-8% faster than MPFR (kept).
- The derived Q-band (e.g. `bound_sq`) is a proven invariant, not a tunable.
- **Deep ε is slow** (≤1e-7 is seconds-to-minutes per target); iterate at coarse
  ε (1e-2/1e-3). `#[ignore]` tests are diagnostic *probes* (`*/probes.rs` or
  inline), not part of the default suite.
- `CYCLOSYNTH_*` env vars are A/B kill-switches; some (e.g. `CYCLOSYNTH_BOUND_SQ`)
  affect search *results*.
- `docs/`, `scripts/`, `bench_logs/`, `target/` are gitignored (planning notes /
  internal bench tooling). User-facing examples live in `examples/`.
