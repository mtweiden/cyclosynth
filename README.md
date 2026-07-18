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
- `synthesize_u1(lam, epsilon)` and `synthesize_u2(phi, lam, epsilon)` cover the rest of the qiskit U-gate family.
- **Supported ε:** Clifford+T is validated to `1e-10` (below that it warns and proceeds); Clifford+√T requires `ε ≥ 1e-8`.

For repeated calls, construct one reusable instance:

```python
synth = cyclosynth.Synthesizer(epsilon=1e-5, sqrt_t=True)
r = synth.synthesize_u3(1.0472, 2.7577, 5.3947)
```

The `Synthesizer` constructor also exposes tuning knobs (`deadline_ms`, `q_cost`, …) for trading a little circuit cost for speed at deep ε; the defaults already minimize cost as far as my testing[...]

Runnable demos are in [`examples/`](examples/).

## Usage (Rust)

The same API is available directly from the crate:

```rust
use cyclosynth::synthesis::{Synthesizer, angle::Angle};

let synth = Synthesizer::new(1e-5, /* sqrt_t = */ false);
let result = synth.synthesize_u3(Angle::Rad(1.0472), Angle::Rad(2.7577), Angle::Rad(5.3947)).unwrap();
println!("{}", result.gates.unwrap());
```
