# AGENTS.md

Guidance for AI coding agents working in `cyclosynth`. (Human-facing intro is in
[README.md](README.md); this file holds the build/test/convention details and
the non-obvious gotchas an agent needs.)

## What this is

A pure-Rust library that synthesizes a single-qubit target unitary into a
**Clifford+T** or **Clifford+√T** circuit, minimizing `T_count + 3.5·Q_count`
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

## Conventions

- **Commits**: subject ~50 chars; no AI-attribution trailers (no
  "Co-Authored-By"/"Generated with"). Commit only when asked; branch first if on
  the default branch for a large change.
- **Comments explain WHY, not WHAT.** Load-bearing only; no restating the code.
  Spell out an acronym at first use (the glossary is the canonical place).
- **Naming**: domain terms (`q`=√T gate, `lde`, `lll`, `fgkm`, `cfa`, `dd`,
  `zeta`/`zomega`) are kept deliberately and defined in the glossary — do **not**
  "clarify" them by renaming. But tests/probes/specific internal fns should be
  named for what they *do*, not by an investigation code.
- **Float types**: `Float` = `f64` (hardware), `MpFloat` = `rug::Float` (MPFR,
  arbitrary precision — NOT a fixed-width "f128"). These are the only two; use
  them consistently.

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

## Gotchas (the things that will bite you)

- **Telemetry is `trace`-feature-gated and zero-cost when off.**
  `diag::trace_enabled()` is a compile-time `const false` without `--features
  trace`. NEVER add an unconditional `diag::*` counter write — always behind
  `if trace_enabled()`. The default build must carry no telemetry cost (LLL is
  ~99% of CPU at deep ε).
- **The hot path is the Schnorr-Euchner walk (`*/se.rs`) and the LLL.** Any
  change there needs an A/B wall-time measurement, not a guess. "Cold per-prefix
  f64 → MPFR is free" is NOT universal: the f64 GS *did* turn out free (removed),
  but the per-prefix f64 Cholesky pays ~6-8% — measure before collapsing a fast
  path.
- **Bound-invariant policy**: the derived Q-band (e.g. `bound_sq`) is
  theorem-grade. If something looks like it needs the bound *widened* to pass,
  the bug is elsewhere — never widen a sound bound to absorb a symptom.
- **Test discipline**: iterate at coarse ε (1e-2/1e-3, k≤8); deep ε (≤1e-7) is
  slow (seconds-to-minutes per target) — reserve for milestones and **background
  long runs**, never block on a >30s foreground run. `#[ignore]` tests are
  diagnostic *probes* (in `*/probes.rs` or inline); they're not part of the
  default suite. "OK to miss a solution at lde = k if it's found at k+1."
- **Env vars** (`CYCLOSYNTH_*`) exist as A/B kill-switches; several (e.g.
  `CYCLOSYNTH_BOUND_SQ`) affect search *results* and are read per-call by tests —
  don't "optimize" them into one-time caches without checking the tests.
- `docs/`, `scripts/`, `bench_logs/`, `target/` are gitignored (planning notes /
  internal bench tooling). User-facing examples live in `examples/`.

## When unsure

Prefer measuring over asserting; prefer the existing parallel-backend symmetry;
keep the public Python surface small (`Synthesizer` + `SynthResult` only).
