# cyclosynth

Syntheize general single-qubit unitaries into Clifford+T and Clifford+√T gates.

Given a target gate and a tolerance ε, cyclosynth returns a circuit within diamond distance ε of the target:

- **Clifford+T** — gates `{H, S, T, X, Y, Z}`, with the *minimal T-count* at that ε.
- **Clifford+√T** — adds the Q = √T gate; typically ~20% cheaper than the Clifford+T circuit for the same target, and never costlier.

This code base implements the algorithm of [Morisaki et al.](https://arxiv.org/abs/2510.05816), extended to Clifford+T.
Typical runtimes are about a half second at ε=`1e-10` (Clifford+T).

## Install

Requires system `gmp`/`mpfr` and [maturin](https://www.maturin.rs/), which builds the Rust extension into the active Python environment:

```sh
brew install gmp mpfr     # macOS  (Debian: apt-get install libgmp-dev libmpfr-dev)
pip install maturin
maturin develop --release
```

## Usage

`synthesize_u3(theta, phi, lam, epsilon)` synthesizes a general single-qubit gate (the qiskit `U3` convention):

```python
import cyclosynth

r = cyclosynth.synthesize_u3(1.0472, 2.7577, 5.3947, 1e-10)

if r:                    # None if nothing was found within epsilon
    print(r.gates)       # gate string, e.g. "HTSHt..." — lowercase = dagger (t = T†)
    print(r.t_count)     # also .q_count, .cost, .lde
    print(r.distance)    # diamond distance to the target, < epsilon
```

Pass `sqrt_t=True` to synthesize over Clifford+√T instead:

```python
r = cyclosynth.synthesize_u3(1.0472, 2.7577, 5.3947, 1e-5, sqrt_t=True)
```

Notes:

- **Angles, not matrices.** Targets are given by rotation angles — floats (radians) or exact-π strings (`"pi/64"`, `"3*pi/4"`, `"-2pi/3"`). Deep-ε synthesis needs more precision than an `f64` may carry.
- **Gate order.** In the gate string, the leftmost gate is the leftmost matrix factor: `"ABC"` means `A·B·C`.
- `synthesize_u1(lam, epsilon)` and `synthesize_u2(phi, lam, epsilon)` cover the rest of the qiskit U-gate family; `synthesize_rz/rx/ry(theta, epsilon)` cover the axis rotations (Rx and Ry are the Rz circuit wrapped in exact Cliffords, so they share Rz's range and guarantees).
- **Supported ε:** z-rotation targets (`u1`/`rz`/`rx`/`ry`, or `u3` with θ = 0) run a native Ross–Selinger-style route on BOTH gate sets: milliseconds-to-seconds, exact-verified, down to the hard floor of `1e-48`. General targets: the u-gate family (`synthesize_u1/u2/u3`) is jointly optimized by the lattice pipelines — Clifford+T validated to `1e-10` (warns and proceeds below; slow past `1e-12`), Clifford+√T to `1e-8`. `synthesize_zyz(alpha, beta, gamma, epsilon)` instead builds the rotation product through the native gridsynth routes (each rotation at ε/3): the full `1e-48`-class range on both gate sets, seconds-fast at any depth, at ~2.5–3× the jointly-optimized cost.
- **Cost-first √T:** the Clifford+√T z-rotation route optimizes the weighted gate cost (T + 3·√T) over hundreds of candidates and all legal assembly phases, preferring cheaper circuits over faster calls; it averages ~0.75–0.79× the T-only cost of the same angle at the same ε.

For repeated calls, construct one reusable instance:

```python
synth = cyclosynth.Synthesizer(epsilon=1e-5, sqrt_t=True)
r = synth.synthesize_u3(1.0472, 2.7577, 5.3947)
```

The `Synthesizer` constructor also exposes tuning knobs (`deadline_ms`, `q_cost`, …) for trading a little circuit cost for speed at deep ε; the defaults already minimize cost as far as my testing[...]

Runnable demos are in [`examples/`](examples/).

## Attribution

Algorithms follow arXiv:2510.05816 (core lattice synthesis) and
Ross–Selinger arXiv:1403.2975 (z-rotations). Portions of the Clifford+T
z-rotation route were derived from the MIT-licensed
[pygridsynth](https://github.com/quantum-programming/pygridsynth)
(see [THIRD_PARTY.md](THIRD_PARTY.md)).

## Usage (Rust)

The same API is available directly from the crate:

```rust
use cyclosynth::synthesis::{Synthesizer, angle::Angle};

let synth = Synthesizer::new(1e-5, /* sqrt_t = */ false);
let result = synth.synthesize_u3(Angle::Rad(1.0472), Angle::Rad(2.7577), Angle::Rad(5.3947)).unwrap();
println!("{}", result.gates.unwrap());
```

## Reproducing the paper

The data and numbers in arXiv:2609.16659 come from the scripts in
[`scripts/`](scripts/). One command rebuilds the library and regenerates
them (many hours at 500 targets per precision; resumable):

```sh
scripts/reproduce_paper.sh                 # build, gather, print numbers
SKIP_GATHER=1 scripts/reproduce_paper.sh   # recompute from existing CSVs
```

It needs `maturin`, Python with `numpy` and `mpmath`, and the
Ross–Selinger `gridsynth` binary (`cabal install newsynth`, or set
`GRIDSYNTH_BIN`). Each step can also be run on its own:

| Step | Script | Produces | Paper |
| --- | --- | --- | --- |
| General targets | `gather_u3.py` | `scripts/data/u3.csv` | Tables 2, 3 and the cost fits |
| z-rotations | `gather_rz.py` | `scripts/data/rz.csv` | Sec. 5.3 (gridsynth check), Sec. 6 |
| Quoted numbers | `paper_numbers.py` | `paper_numbers.txt` | Tables 2, 3 and the text |

The gathers draw seeded targets (the same 500 per precision on every run),
time each call in a fresh process with the whole machine, re-verify every
circuit's diamond distance in 40-digit arithmetic, and record the host and
commit in a `.meta.json` sidecar. Clifford+√T runs use the library's default
cost-search deadline; they turn off the near-Clifford speed escape
(`near_clifford_escape=False`), which can add T gates, and z-rotations use
`native_rz=False` so they exercise the general lattice search. Each call also
has a hard timeout of 5× the slowest completed call at that precision in the
2026-10-17 reference run (at least 30 s; table in `scripts/_gather.py`); a Clifford+√T call that hits it reports its Clifford+T circuit and
is counted by `paper_numbers.py`. Small run-to-run differences are expected: the parallel search
may return a different circuit of the same or nearly the same cost.
