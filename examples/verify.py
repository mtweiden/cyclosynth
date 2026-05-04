"""Independent verification that synthesized gate strings approximate the target.

For each random U3 target we:
  1. Synthesize via cyclosynth (records `result.distance`).
  2. Rebuild the unitary by multiplying H/S/T/X/Y/Z left-to-right — first in
     plain NumPy f64, then in mpmath at 60-decimal precision (so the rebuild
     result is ground-truth, free of float64 round-off).
  3. Compute the diamond distance from each rebuild to the target.
  4. Assert the high-precision rebuild gives a distance below ε, and that it
     matches `result.distance` to within a tight algebraic tolerance.

Why two precisions: at deep ε (~1e-7) the gate string is hundreds of gates
long. Plain f64 accumulates O(n · ε_machine) round-off — visible as noise on
the distance number when ε itself is near the noise floor — but the gate
string is still *mathematically* exact. The mpmath rebuild proves this by
re-evaluating the same product at far higher precision.
"""
import numpy as np
import cyclosynth
import mpmath as mp
from random import random, seed

# ── f64 (NumPy) gate matrices ─────────────────────────────────────────────────
H = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
S = np.array([[1, 0], [0, 1j]], dtype=np.complex128)
T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=np.complex128)
X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
I = np.eye(2, dtype=np.complex128)
GATES_F64 = {"H": H, "S": S, "T": T, "X": X, "Y": Y, "Z": Z, "I": I}


def _gates_mp():
    """High-precision gate matrices (mpmath, current mp.mps precision)."""
    j = mp.mpc(0, 1)
    one = mp.mpf(1)
    half_sqrt2 = one / mp.sqrt(2)
    h = mp.matrix([[half_sqrt2, half_sqrt2], [half_sqrt2, -half_sqrt2]])
    s = mp.matrix([[one, 0], [0, j]])
    t = mp.matrix([[one, 0], [0, mp.exp(j * mp.pi / 4)]])
    x = mp.matrix([[0, one], [one, 0]])
    y = mp.matrix([[0, -j], [j, 0]])
    z = mp.matrix([[one, 0], [0, -one]])
    eye = mp.matrix([[one, 0], [0, one]])
    return {"H": h, "S": s, "T": t, "X": x, "Y": y, "Z": z, "I": eye}


def rebuild_f64(gate_str):
    """Multiply gate matrices left-to-right in NumPy complex128."""
    U = np.eye(2, dtype=np.complex128)
    for c in gate_str:
        U = U @ GATES_F64[c]
    return U


def rebuild_mp(gate_str, gates):
    """Multiply gate matrices left-to-right in mpmath (arbitrary precision)."""
    U = mp.matrix([[1, 0], [0, 1]])
    for c in gate_str:
        U = U * gates[c]
    return U


def diamond_f64(A, B):
    """√max(0, 1 − |tr(A·B†)|²/4) — same formula as diamond_distance_float."""
    tr = np.trace(A @ B.conj().T)
    return np.sqrt(max(0.0, 1.0 - abs(tr) ** 2 / 4.0))


def diamond_mp(A, B):
    """High-precision diamond distance. A, B are mpmath 2×2 matrices."""
    # tr(A·B†) = sum_{i,k} A[i,k] · conj(B[i,k])
    tr = mp.mpc(0)
    for i in range(2):
        for k in range(2):
            tr += A[i, k] * mp.conj(B[i, k])
    val = mp.mpf(1) - mp.fabs(tr) ** 2 / 4
    if val < 0:
        val = mp.mpf(0)
    return mp.sqrt(val)


def target_to_mp(target):
    """np.complex128 2x2 → mpmath 2x2 (exact reinterpretation of the f64 bits)."""
    return mp.matrix(
        [[mp.mpc(target[i, k].real, target[i, k].imag) for k in range(2)]
         for i in range(2)]
    )


def rz(t):
    return np.array([[np.exp(-1j * t / 2), 0],
                     [0,                    np.exp(1j * t / 2)]],
                    dtype=np.complex128)


def ry(t):
    c, s = np.cos(t / 2), np.sin(t / 2)
    return np.array([[c, -s], [s, c]], dtype=np.complex128)


def main():
    seed(0)
    epsilon = 1e-7
    n_trials = 100
    synth = cyclosynth.Synthesizer(epsilon=epsilon)

    # 60 decimals: ~200 bits of mantissa. With ~250 matmuls per circuit,
    # accumulated error is ~250 · 10^-60 ≈ 3e-58 — many orders below ε.
    mp.mp.dps = 60
    GATES_MP = _gates_mp()

    max_mp_dist = 0.0
    max_disagreement = 0.0
    max_f64_drift = 0.0
    failures = 0

    for i in range(n_trials):
        alpha, beta, gamma = [2 * np.pi * random() for _ in range(3)]
        target = rz(alpha) @ ry(beta) @ rz(gamma)

        result = synth.synthesize(target)
        if result is None or result.gates is None:
            print(f"[{i:3d}] U3({alpha:.3f},{beta:.3f},{gamma:.3f}): synthesis returned None")
            failures += 1
            continue

        # f64 rebuild — accumulates ~n_gates · ε_machine round-off.
        rebuilt_f64 = rebuild_f64(result.gates)
        py_f64 = diamond_f64(rebuilt_f64, target)

        # mpmath rebuild — at 60-digit precision, round-off is negligible.
        # This is the mathematically true diamond distance of the gate string.
        rebuilt_mp = rebuild_mp(result.gates, GATES_MP)
        target_mp = target_to_mp(target)
        py_mp = float(diamond_mp(rebuilt_mp, target_mp))

        # Drift between f64 rebuild and mpmath rebuild (= pure float64 noise).
        f64_drift = abs(py_f64 - py_mp)
        # Disagreement vs Rust's reported distance (= round-off in u2t.to_float).
        disagreement = abs(py_mp - result.distance)

        # The mathematically true distance must be below ε. This is the
        # only "correctness" assertion — everything else is diagnostic.
        ok_below_eps = py_mp < epsilon
        # Rust's reported distance should agree with the mp truth to within
        # u2t.to_float() round-off (large k loses bits at 2^53 boundary).
        # ~ε/10 is a generous bound: Rust's `dist < ε` decision is reliable
        # as long as the noise stays well below ε itself.
        ok_consistent = disagreement < epsilon / 10

        max_mp_dist = max(max_mp_dist, py_mp)
        max_disagreement = max(max_disagreement, disagreement)
        max_f64_drift = max(max_f64_drift, f64_drift)

        status = "OK " if (ok_below_eps and ok_consistent) else "FAIL"
        print(
            f"[{i:3d}] {status}  rust={result.distance:.3e}  mp={py_mp:.3e}  "
            f"f64={py_f64:.3e}  |mp−rust|={disagreement:.1e}  "
            f"|f64−mp|={f64_drift:.1e}  T={result.gates.count('T')}  "
            f"len={len(result.gates)}"
        )
        if not (ok_below_eps and ok_consistent):
            failures += 1

    print("=" * 80)
    print(f"trials={n_trials}  ε={epsilon:.0e}")
    print(f"max  mp-diamond       (TRUE distance)         = {max_mp_dist:.3e}  (must be < ε)")
    print(f"max |mp − rust|       (BlochDecomposer check) = {max_disagreement:.3e}")
    print(f"max |f64 − mp|        (f64 round-off floor)   = {max_f64_drift:.3e}")
    print(f"failures: {failures}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
