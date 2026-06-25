import csv
import os
import sys
from time import perf_counter
from random import random

import cyclosynth
import mpmath as mp
import numpy as np
import trasyn

# Reuse verify.py's mpmath helpers so distance reporting matches the
# Rust verification path bit-for-bit.
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), os.pardir, "examples")
)
from verify import _gates_mp, diamond_mp, rebuild_mp, target_to_mp  # noqa: E402


# Build a single-qubit unitary as U3(α, β, γ) = Rz(α) · Ry(β) · Rz(γ).
def rz(t):
    return np.array([[np.exp(-1j * t / 2), 0],
                     [0,                    np.exp(1j * t / 2)]],
                    dtype=np.complex128)

def ry(t):
    c, s = np.cos(t / 2), np.sin(t / 2)
    return np.array([[c, -s],
                     [s,  c]], dtype=np.complex128)


EPSILONS = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
N_TRIALS = 200
CSV_PATH = "scripts/data/comparison_data.csv"

# 60-decimal mpmath precision — ~200 mantissa bits. With ~250 matmuls per
# circuit at deep ε, accumulated rebuild error is ~250·10⁻⁶⁰ ≈ 3e-58, far
# below any ε we synthesize at.
mp.mp.dps = 60
GATES_MP = _gates_mp()


def true_distance(gates: str, target_np: np.ndarray) -> float:
    """Diamond distance between target and the mpmath rebuild of `gates`,
    via the Frobenius reformulation `D² = q · (8 − q) / 16`. Backend-
    independent; matches the Rust `diamond_distance_u2t_float` to f64
    precision."""
    if not gates:
        return float("inf")
    rebuilt = rebuild_mp(gates, GATES_MP)
    target_mp = target_to_mp(target_np)
    return float(diamond_mp(rebuilt, target_mp))


def run_trasyn(target, epsilon):
    start = perf_counter()
    budget = 10 * int(np.log10(1 / epsilon)) + 2
    try:
        seq, _mat, _err = trasyn.synthesize(
            target, error_threshold=epsilon, nonclifford_budget=budget
        )
        duration = perf_counter() - start
        gates = seq.upper()
        # Don't trust `_err` — trasyn returns its own self-reported number,
        # likely via the trace formula `1 − |tr|²/4` which clamps to 0 at
        # ε ≲ √machine_eps. Compute distance ourselves at mpmath precision.
        return seq.count("t"), true_distance(gates, target), duration, gates
    except Exception:
        duration = perf_counter() - start
        return -1, float("inf"), duration, ""


def run_cyclosynth(synth, target):
    start = perf_counter()
    try:
        result = synth.synthesize(target)
        duration = perf_counter() - start
        gates = (result.gates or "").upper()
        t_count = gates.count("T")
        # Cyclosynth's `result.distance` already uses the Frobenius MPFR
        # path, but we recompute via mpmath here for symmetry with trasyn
        # — both columns then come from the same oracle.
        return t_count, true_distance(gates, target), duration, gates
    except Exception:
        duration = perf_counter() - start
        return -1, float("inf"), duration, ""


def main():
    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        # writer.writerow([
        #     "epsilon", "method", "trial", "alpha", "beta", "gamma",
        #     "t_count", "distance", "duration_ms", "success", "gates",
        # ])

        for epsilon in EPSILONS:
            print(f"\n=== epsilon = {epsilon:.0e} ===")
            synth = cyclosynth.Synthesizer(epsilon=epsilon)

            for trial in range(N_TRIALS):
                # Haar-random SU(2): alpha/gamma uniform, beta sine-weighted on [0, pi]
                alpha = 2 * np.pi * random()
                gamma = 2 * np.pi * random()
                beta = np.arccos(1.0 - 2.0 * random())
                target = rz(alpha) @ ry(beta) @ rz(gamma)

                results = [
                    # ("trasyn",     run_trasyn(target, epsilon)),
                    ("cyclosynth", run_cyclosynth(synth, target)),
                ]

                for name, (t_count, dist, dur, gates) in results:
                    success = dist <= epsilon
                    writer.writerow([
                        f"{epsilon:.0e}",
                        name,
                        trial,
                        alpha,
                        beta,
                        gamma,
                        t_count if t_count is not None else "",
                        f"{dist:.6e}",
                        f"{dur * 1000:.3f}",
                        success,
                        gates,
                    ])
                    f.flush()
                    tag = "OK" if success else "FAIL"
                    tc = f"{t_count:>3}" if t_count is not None else "  -"
                    print(f"  trial {trial + 1:>3}  {name:<12} "
                          f"T={tc}  d={dist:.3e}  {dur * 1000:>9.1f} ms  {tag}")
    print(f"\nwrote {CSV_PATH}")


if __name__ == "__main__":
    main()

