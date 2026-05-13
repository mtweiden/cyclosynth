# cyclosynth-rs

Pure-Rust implementation of optimal-T-count Clifford+T synthesis for
single-qubit unitaries.

Given a 2×2 target unitary `V` and a tolerance `ε`, the synthesizer finds a
Clifford+T circuit `U` with `d_diamond(U, V) < ε` and the smallest possible
T-count for that ε. This implements Algorithm 3.14 of
[Mosca et al., arXiv:2510.05816](https://arxiv.org/abs/2510.05816), which
itself rests on the 8-dimensional integer enumeration of Algorithm 3.6 plus
the divide-and-conquer split of Algorithm 3.11.

## Quick start (Python)

Install via [maturin](https://www.maturin.rs/) — this builds the Rust
extension and installs `cyclosynth` into the active environment:

```sh
pip install maturin
maturin develop --release
```

Then synthesize a random single-qubit unitary:

```python
import numpy as np
import cyclosynth

# Build a single-qubit unitary as U3(α, β, γ) = Rz(α) · Ry(β) · Rz(γ).
# Angles fixed for reproducibility (originally drawn from uniform(0, 2π)).
def rz(t):
    return np.array([[np.exp(-1j * t / 2), 0],
                     [0,                    np.exp(1j * t / 2)]],
                    dtype=np.complex128)

def ry(t):
    c, s = np.cos(t / 2), np.sin(t / 2)
    return np.array([[c, -s],
                     [s,  c]], dtype=np.complex128)

alpha, beta, gamma = 4.863069, 2.757718, 5.394728
target = rz(alpha) @ ry(beta) @ rz(gamma)

# Approximate to within ε = 1e-5 in diamond distance.
synth = cyclosynth.Synthesizer(epsilon=1e-5)
result = synth.synthesize(target)

print(f"gates    = {result.gates}")      # Clifford+T sequence over {H, S, T, X, Y, Z}
print(f"lde      = {result.lde}")        # ≈ T-count + small offset
print(f"distance = {result.distance:e}") # < epsilon
```

Round-tripping the gate string back to a unitary recovers the target.
The composition convention is *leftmost gate is the leftmost matrix
factor* — so for a gate string `"ABC"`, the resulting unitary is `A·B·C`:

```python
inv2 = 1 / np.sqrt(2)
GATES = {
    "H": np.array([[inv2, inv2], [inv2, -inv2]],              dtype=np.complex128),
    "S": np.array([[1,    0],    [0,    1j]],                 dtype=np.complex128),
    "T": np.array([[1,    0],    [0,    np.exp(1j*np.pi/4)]], dtype=np.complex128),
    "X": np.array([[0,    1],    [1,    0]],                  dtype=np.complex128),
    "Y": np.array([[0,    -1j],  [1j,   0]],                  dtype=np.complex128),
    "Z": np.array([[1,    0],    [0,    -1]],                 dtype=np.complex128),
}

U = np.eye(2, dtype=np.complex128)
for g in result.gates:
    U = U @ GATES[g]

tr = np.trace(U @ target.conj().T)
recovered_distance = np.sqrt(max(0.0, 1.0 - abs(tr) ** 2 / 4.0))
assert recovered_distance < 1e-5
```

The synthesizer accepts any 2×2 `np.complex128` ndarray (contiguous or
strided). Optional keyword arguments override the defaults:
`Synthesizer(epsilon, *, max_lde=None, min_lde=None, direct_limit=None)`.

## Quick start (Rust)

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
        ├── mod.rs          LenstraScratch dispatch + phase1 entry point
        ├── integer.rs      L²-LLL with i256 Gram + f64 GS for all ε
        └── se.rs           Schnorr-Euchner walk + post-LLL helpers
                            (det check, Euclidean Cholesky, lattice-point
                            reconstruction, bilinear-form check)
```

## The lattice-enumeration pipeline

`Synthesizer` builds an 8-dimensional integer lattice from the target's
alignment vector `y` and a norm shell `2^k` (the "lde" — log denominator
exponent). The lattice carries an anisotropic inner product `Q` that encodes
the `cap × ball` body whose interior contains valid `(u₁, u₂) ∈ Z[ω]²`
candidates.

[`lenstra::integer`](src/synthesis/lenstra/integer.rs) runs L²-LLL
(Nguyen–Stehlé 2009): exact integer Gram in `i256`, Gram-Schmidt
coefficients in `f64`, INSERT semantics + lazy size-reduction. Stable
down to `ε = 1e-10`. Theorem 2 of the paper proves `f64` is sufficient for
any `d ≤ 11` at `(δ=0.75, η=0.55)`, comfortably covering our `d=8`.

The post-LLL phase runs:
- f64 Cholesky on the natural-scale Gram (LLL invariant bounds κ ≤ 16
  for d=8, so f64 is provably sufficient at the SE bound check).
- MPFR LU at scaled precision (`compute_lu_prec(eps) ≈ 6·log₂(1/ε)`)
  for the cap-center solve.
- Schnorr-Euchner enumeration in MPFR-128 in
  [`lenstra::se`](src/synthesis/lenstra/se.rs).

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
8 threads, against 10 random SU(2) targets per ε from a fixed
xorshift64 seed. The `lde` column is the inner T-count budget at which
a solution was found (≈ T-count + small adjustment for the search
convention).

| ε    | typical lde | min (ms) | median (ms) | max (ms) | mean (ms) |
| ---- | ----------- | -------- | ----------- | -------- | --------- |
| 1e-2 | 18          | 0.4      | 0.6         | 0.9      | 0.6       |
| 1e-3 | 26          | 0.5      | 0.7         | 1.0      | 0.7       |
| 1e-4 | 35–39       | 0.5      | 2.0         | 14.3     | 3.1       |
| 1e-5 | 43–49       | 0.6      | 5.5         | 77       | 18.3      |
| 1e-6 | 55–59       | 3.6      | 65          | 223      | 81        |
| 1e-7 | 66          | 1.1      | 25          | 89       | 32        |
| 1e-8 | 74–80       | 4        | 2,492       | 9,841    | 3,160     |

Total (sum of minimums across all 70 runs): **32.9 s**.

Per-target time can be highly non-monotonic in ε and varies by orders
of magnitude across the 10-target sweep at deeper ε. The dominant cost
is the number of MA prefixes processed before a valid `(u₁, u₂)`
candidate turns up, which depends on where the target sits in the
modular fundamental domain — some `(target, ε)` combinations land on a
"lucky" prefix early in the search order. The mean-vs-median gap at
ε ≤ 1e-5 reflects this: a few hard outliers per sweep dominate the
mean while the median stays modest. The same phenomenon explains why
ε=1e-7's mean (32 ms) is *smaller* than ε=1e-6's (81 ms) here — the
ε=1e-6 sweep happened to land on more high-prefix-index targets.

Some angles also need a higher T-count than others at the same ε.
`π/n` for small `n` (and similar small-period rationals) tend to fall
in sparse regions of the cyclotomic-integer approximation lattice, so
their resolved T-count is larger and their search runs deeper than
generic-irrational angles. This is an algorithmic property of the
underlying Diophantine approximation problem, not an implementation
detail.
