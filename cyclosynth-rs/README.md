# cyclosynth-rs

Pure-Rust implementation of optimal-T-count Clifford+T synthesis for
single-qubit unitaries.

Given a 2×2 target unitary `V` and a tolerance `ε`, the synthesizer finds a
Clifford+T circuit `U` with `d_diamond(U, V) < ε` and the smallest possible
T-count for that ε. This implements Algorithm 3.14 of
[Mosca et al., arXiv:2510.05816](https://arxiv.org/abs/2510.05816), which
itself rests on the 8-dimensional integer enumeration of Algorithm 3.6 plus
the divide-and-conquer split of Algorithm 3.11.

## Quick example

```rust
use cyclosynth::synthesis::Synthesizer;
use num_complex::Complex;

// Rz(0.3) — a non-Clifford rotation
let theta = 0.3_f64;
let target = [
    [Complex::from_polar(1.0, -theta / 2.0), Complex::new(0.0, 0.0)],
    [Complex::new(0.0, 0.0), Complex::from_polar(1.0, theta / 2.0)],
];

let synth = Synthesizer::new(1e-5);
let result = synth.synthesize(target).unwrap();
println!("T-count = {}, gates = {}", result.lde, result.gates.unwrap());
```

The `time_synthesis` binary runs a benchmark suite across a range of
`(target, ε)` pairs:

```sh
cargo run --release --bin time_synthesis -- --threads 8 --trials 3
```

## How synthesis works

```
Synthesizer::synthesize(target)
        │
        ▼
  for t in 0, 1, 2, ...                       (T-count budget)
        ├── if t ≤ direct_limit:                 → search::aligned_search
        │     brute-force shell ‖x‖² = 2^t with Cauchy–Schwarz pruning
        │     across 24 Clifford prefixes × {even, T, T†} branches.
        │
        └── else:                                 → dc_search (divide & conquer)
              split: t' = max(t − direct_limit,
                              ⌈t − 5/2·log₂(1/ε)⌉)
              for each Matsumoto-Amano left prefix U_L ∈ L_{t'}:
                  ├── try U_R = LLL_aligned_search(target · U_L†)        (even branch)
                  └── try U_R = LLL_aligned_search(target · U_L† · T†)   (odd branch)
              the search runs in parallel via rayon (per-prefix scratch
              reused per worker; first-found wins).
```

`LLL_aligned_search` is the workhorse for non-trivial T-counts: it builds a
Q-metric lattice from the target, reduces it with LLL, and walks integer
lattice points within the cap × ball intersection (Schnorr-Euchner
enumeration) until it finds an `x ∈ ℤ⁸` whose reconstruction satisfies the
unitarity constraints.

## File layout

```
src/
├── lib.rs
├── matrix/             U2, SO3 matrix types and basic operations
├── rings/              Cyclotomic integer rings Z[ω], Z[ζ] and Float aliases
└── synthesis/
    ├── cliffords.rs        24-element Clifford table for the outer search
    ├── decomposer.rs       Output-side: Z[ω]² unitary → Clifford+T gate string
    ├── diag.rs             Optional CYCLOSYNTH_TRACE=1 counters
    ├── search.rs           Direct enumeration over small norm shells
    ├── synthesizer.rs      Top-level Synthesizer, T-count loop, dc_search,
    │                       parallel dispatch
    └── lenstra/
        ├── mod.rs          Dispatch: ε ≥ 1e-4 picks `light`, else `integer`
        ├── light.rs        TwoFloat (~104-bit) LLL for moderate ε
        ├── integer.rs      L²-LLL with i256 Gram + f64 GS for tight ε
        └── se.rs           Schnorr-Euchner walk + post-LLL helpers
                            (det check, Euclidean Cholesky, lattice-point
                            reconstruction, bilinear-form check)
```

## The two lattice-enumeration paths

`Synthesizer` builds an 8-dimensional integer lattice from the target's
alignment vector `y` and a norm shell `2^k` (the "lde" — log denominator
exponent). The lattice carries an anisotropic inner product `Q` that encodes
the `cap × ball` body whose interior contains valid `(u₁, u₂) ∈ Z[ω]²`
candidates.

The reduction-and-enumeration backend is picked per call, in
[`lenstra::LenstraScratch::new`](src/synthesis/lenstra/mod.rs):

| Backend | Range | What it uses |
|---|---|---|
| **`light`** | `ε ≥ 1e-4` | TwoFloat (`f64+f64`, ~104-bit mantissa). Stack-allocated, `Copy`-friendly arithmetic. Fast at moderate ε where `κ(Q)` fits in TwoFloat's margin. |
| **`integer`** | `ε < 1e-4` | L²-LLL (Nguyen–Stehlé 2009): exact integer Gram in `i256`, Gram-Schmidt coefficients in `f64`, INSERT semantics + lazy size-reduction. Stable down to `ε = 1e-10`. |

Both paths share the same Schnorr-Euchner enumeration and post-SE candidate
validation in [`lenstra::se`](src/synthesis/lenstra/se.rs).

### Why two paths?

`κ(Q) ≈ 16/ε⁴` grows fast. TwoFloat's ~104-bit mantissa is enough through
`ε ≈ 1e-4`. Below that, Gram-Schmidt cancellation eats the precision, the
basis is no longer L³-reduced after LLL, and the algorithm produces wrong
results. The `integer` path replaces the floating Gram with an exact
integer one; the GS coefficients then only need enough precision to make
correct Lovász decisions — Theorem 2 of Nguyen–Stehlé proves `f64` is
sufficient for any `d ≤ 11` at `(δ=0.75, η=0.55)`, comfortably covering
our `d=8`.

## Threading model

[`Synthesizer::synthesize`] uses [`rayon`] to parallelise the
prefix loop inside `dc_search`. Per-worker scratch is allocated once via
`rayon::map_init` and reused across all prefixes that worker handles, so
the LLL inner loop has zero per-prefix heap allocation. The
[`rayon::find_any`] combinator short-circuits as soon as any thread
finds a valid solution.

Threading efficiency at moderate ε is ~96% (CPU-summed time vs
wall-time × n_threads). Threading is not the bottleneck.

## Diagnostic tracing

Setting `CYCLOSYNTH_TRACE=1` in the environment dumps per-lde counters
to stderr:

```
[trace] lde=49 pass1 t'=8 prefixes=4952 mat_uv_rej=0 se_cb=56680 budget=0 3690.2ms result=none
[trace]            phase_ms (cpu-summed) build=295.3 lll=8415.1 chol=152.7 lu=289.2 se=26.0 sum=9178.3
[trace]            lll_iters total=1690284 avg=341 max=231 at_cap=0 (cap=10000)
```

Counters live in [`crate::synthesis::diag`].

## Building

The crate links against system `gmp` and `mpfr` via [`gmp-mpfr-sys`].
On macOS:

```sh
brew install gmp mpfr
cargo build --release
```

On Debian/Ubuntu:

```sh
apt-get install libgmp-dev libmpfr-dev
cargo build --release
```

## Testing

```sh
cargo test --release             # ~130 unit tests
cargo test --release -- --ignored   # add deeper-ε round-trip tests (~minutes)
```

Verification tests at `verify_correctness_at_1e_X_*` round-trip a
synthesized circuit through the Bloch decomposer and assert
`d_diamond(rebuilt, target) < ε`.

## Performance

Approximate wall-clock on Apple M-series, 8 threads (`time_synthesis
--trials 2`):

| Target | ε | T-count | Time |
|---|---|---|---|
| `Rz(π/7)` | 1e-3 | 28 | < 5 ms |
| `Rz(π/7)` | 1e-5 | 51 | ~250 ms |
| `Rz(π/7)` | 1e-7 | 70 | ~2.5 s |
| `Rz(0.30)` | 1e-8 | 82 | ~50 s |

Some target angles (notably `π/n` for small `n`) consistently need higher
T-count than generic angles at the same ε; this is a property of the
underlying Diophantine approximation problem, not an implementation
quirk.
