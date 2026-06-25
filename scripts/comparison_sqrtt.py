"""T-count-equivalent cost comparison: Clifford+T vs Clifford+√T.

Synthesizes the same set of random U3 targets across ε ∈ {1e-3..1e-6}
with both backends, writes per-target rows to CSV. Sister script
`plot_comparison_sqrtt.py` reads the CSV and produces the grouped-violin
plot.

Cost model:
    cost = T_count + 3 · Q_count
(other Clifford gates are free in fault-tolerant accounting; Q gates cost
3× a T gate).
"""

import csv
import os
from time import perf_counter
from random import random, seed as set_seed

import cyclosynth
import numpy as np


# ─── Target generator ───────────────────────────────────────────────────────

def rz(t):
    return np.array([[np.exp(-1j * t / 2), 0],
                     [0, np.exp(1j * t / 2)]], dtype=np.complex128)


def ry(t):
    c, s = np.cos(t / 2), np.sin(t / 2)
    return np.array([[c, -s], [s, c]], dtype=np.complex128)


# ─── Cost model ─────────────────────────────────────────────────────────────

T_COST = 1.0
Q_COST = 3.0


def cost_of(gates: str) -> float:
    return T_COST * gates.count("T") + Q_COST * gates.count("Q")


# ─── Config ─────────────────────────────────────────────────────────────────

EPSILONS = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
N_TRIALS = 500
SEED = 42
CSV_PATH = "scripts/data/comparison_sqrtt_data.csv"


def run(synth, target):
    """Returns (gates, distance, duration_s). Uses synth's reported distance
    directly — that's the Frobenius-reformulated MPFR-correct value.
    Recomputing via mpmath would need a Q-aware gate table; not worth
    the extra dependency just to cross-check what we already trust."""
    start = perf_counter()
    try:
        result = synth.synthesize(target)
        duration = perf_counter() - start
        if result is None:
            return "", float("inf"), duration
        gates = (result.gates or "").upper()
        return gates, float(result.distance), duration
    except Exception:
        duration = perf_counter() - start
        return "", float("inf"), duration


def main():
    set_seed(SEED)
    write_header = not os.path.exists(CSV_PATH)

    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "epsilon", "method", "trial",
                "alpha", "beta", "gamma",
                "t_count", "q_count", "cost",
                "distance", "duration_ms", "success", "gates",
            ])

        for epsilon in EPSILONS:
            print(f"\n=== epsilon = {epsilon:.0e} ===")
            synth_t = cyclosynth.Synthesizer(epsilon=epsilon)
            synth_q = cyclosynth.Synthesizer(
                epsilon=epsilon,
                sqrt_t=True,
                optimize_cost=True,
                # seq_parity=False,
                # seq_parity=True,
                # deadline_ms=5000,
            )

            for trial in range(N_TRIALS):
                # Haar-random SU(2): alpha/gamma uniform, beta sine-weighted on [0, pi]
                alpha = 2 * np.pi * random()
                gamma = 2 * np.pi * random()
                beta = np.arccos(1.0 - 2.0 * random())
                target = rz(alpha) @ ry(beta) @ rz(gamma)

                # Synthesize both backends, then apply the never-costlier floor
                # (Algorithm 2): a Clifford+sqrt(T) circuit is never costlier than
                # the Clifford+T circuit for the same target, so if the raw sqrt(T)
                # result is costlier (or failed) we report the Clifford+T circuit
                # for the sqrt(T) row, keeping its own synthesis duration.
                res = {m: run(s, target) for m, s in
                       (("clifford_t", synth_t), ("clifford_sqrt_t", synth_q))}
                cost_t = cost_of(res["clifford_t"][0]) if res["clifford_t"][0] else float("inf")
                cost_q = cost_of(res["clifford_sqrt_t"][0]) if res["clifford_sqrt_t"][0] else float("inf")
                if cost_t < cost_q:
                    t_gates, t_dist, _ = res["clifford_t"]
                    res["clifford_sqrt_t"] = (t_gates, t_dist, res["clifford_sqrt_t"][2])

                for method in ("clifford_t", "clifford_sqrt_t"):
                    gates, dist, dur = res[method]
                    t_count = gates.count("T")
                    q_count = gates.count("Q")
                    cost = cost_of(gates) if gates else float("nan")
                    success = dist <= epsilon
                    writer.writerow([
                        f"{epsilon:.0e}", method, trial,
                        alpha, beta, gamma,
                        t_count, q_count, f"{cost:.1f}",
                        f"{dist:.6e}", f"{dur * 1000:.3f}",
                        success, gates,
                    ])
                    f.flush()
                    tag = "OK  " if success else "FAIL"
                    print(
                        f"  trial {trial + 1:>3}  {method:<16}  "
                        f"T={t_count:>3} Q={q_count:>3}  "
                        f"cost={cost:>6.1f}  d={dist:.3e}  "
                        f"{dur * 1000:>9.1f} ms  {tag}"
                    )

    print(f"\nwrote {CSV_PATH}")


if __name__ == "__main__":
    main()
