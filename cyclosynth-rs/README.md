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

Wall-clock minimums from `time_synthesis --trials 3` on Apple M-series,
8 threads. The `lde` column is the inner T-count budget at which a
solution was found (≈ T-count + small adjustment for the search
convention).

| Target                | ε      | lde | Time (ms) |
| --------------------- | ------ | --- | ----------|
| `identity`            | 1e-2   | 18  |     1.3   |
| `H`                   | 1e-2   | 18  |     1.3   |
| `T`                   | 1e-2   | 18  |     1.6   |
| `Rz(0.30)`            | 1e-2   | 18  |     2.0   |
| `Rz(1.34)`            | 1e-2   | 18  |     1.9   |
| `Rz(π/7)`             | 1e-2   | 20  |     9.4   |
| `Ry(0.50)`            | 1e-2   | 18  |     1.1   |
| `U3(0.3,0.7,1.2)`     | 1e-2   | 18  |     0.8   |
| `U3(1.1,0.4,2.3)`     | 1e-2   | 18  |     1.0   |
| `Rz(0.30)`            | 1e-3   | 28  |     2.7   |
| `Rz(1.34)`            | 1e-3   | 28  |     2.4   |
| `Rz(π/7)`             | 1e-3   | 28  |     1.6   |
| `Ry(0.50)`            | 1e-3   | 28  |     1.7   |
| `U3(0.3,0.7,1.2)`     | 1e-3   | 28  |     1.7   |
| `Rz(0.30)`            | 1e-4   | 37  |     2.2   |
| `Ry(π/7)`             | 1e-4   | 37  |     3.3   |
| `U3(0.3,0.7,1.2)`     | 1e-4   | 37  |     1.9   |
| `U3(4.3,1.8,0.2)`     | 1e-4   | 37  |     2.1   |
| `U3(6.1,3.4,3.3)`     | 1e-4   | 37  |     2.1   |
| `Rz(0.30)`            | 1e-5   | 49       45     |
| `Ry(π/7)`             | 1e-5   | 51  |   219     |
| `U3(0.3,0.7,1.2)`     | 1e-5   | 47  |     1.6   |
| `U3(4.3,1.8,0.2)`     | 1e-5   | 47  |     1.6   |
| `U3(6.1,3.4,3.3)`     | 1e-5   | 47  |    30     |
| `Rz(0.30)`            | 1e-6   | 59  |   223     |
| `Ry(π/7)`             | 1e-6   | 55  |     4.7   |
| `U3(0.3,0.7,1.2)`     | 1e-6   | 57  |    61     |
| `U3(4.3,1.8,0.2)`     | 1e-6   | 55  |     7.8   |
| `U3(6.1,3.4,3.3)`     | 1e-6   | 59  |   279     |
| `Rz(0.30)`            | 1e-7   | 66  |    22     |
| `Ry(π/7)`             | 1e-7   | 70  |  1220     |
| `U3(0.3,0.7,1.2)`     | 1e-7   | 66  |    23     |
| `U3(4.3,1.8,0.2)`     | 1e-7   | 66  |    32     |
| `U3(6.1,3.4,3.3)`     | 1e-7   | 68  |   280     |
| `Rz(0.30)`            | 1e-8   | 82  | 22100     |

Total (sum of minimums across all 36 cases): **24.6 s**.

Per-target time can be highly non-monotonic in ε. The dominant cost is
the number of MA prefixes processed before a valid `(u₁, u₂)` candidate
turns up, which depends on where the target sits in the modular
fundamental domain — some `(target, ε)` combinations land on a "lucky"
prefix early in the search order. `Rz(0.30)_1e-7` (22 ms) finishing
faster than `Rz(0.30)_1e-6` (223 ms) is normal: the lde=66 search
happens to find a valid lattice point at a low-numbered prefix, while
the lde=59 search needs to walk further.

Some angles also need a higher T-count than others at the same ε.
`π/n` for small `n` (and similar small-period rationals) tend to fall
in sparse regions of the cyclotomic-integer approximation lattice, so
their resolved T-count is larger and their search runs deeper than
generic-irrational angles. This is an algorithmic property of the
underlying Diophantine approximation problem, not an implementation
detail.
