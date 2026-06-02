"""Cost and timing comparison for n=4, n=6, and n=8 synthesis.

Synthesizes the same random U3 targets with:
    n=4: Clifford+T
    n=6: Clifford+Rz(pi/6)
    n=8: Clifford+sqrt(T)

Writes one CSV row per target/backend. Cost model:
    n=4 cost = T_count
    n=6 cost = O_COST * O_count, where O is the pi/6 non-Clifford gate R
    n=8 cost = T_count + Q_COST * Q_count

For now O_COST = 2.0, per the current working assumption.
"""

import csv
import os
from random import random, seed as set_seed
from time import perf_counter

import cyclosynth
import numpy as np


# Target generator: U3(alpha, beta, gamma) = Rz(alpha) Ry(beta) Rz(gamma).
def rz(t):
    return np.array(
        [[np.exp(-1j * t / 2), 0], [0, np.exp(1j * t / 2)]],
        dtype=np.complex128,
    )


def ry(t):
    c, s = np.cos(t / 2), np.sin(t / 2)
    return np.array([[c, -s], [s, c]], dtype=np.complex128)


# Cost model.
T_COST = 1.0
O_COST = 2.0
Q_COST = 2.5


def gate_cost(method, gates):
    if method == "n4_clifford_t":
        return T_COST * gates.count("T")
    if method == "n6_clifford_pi6":
        return O_COST * gates.count("R")
    if method == "n8_clifford_sqrt_t":
        return T_COST * gates.count("T") + Q_COST * gates.count("Q")
    raise ValueError(f"unknown method: {method}")


# Config.
EPSILONS = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
N_TRIALS = 20
SEED = 42
CSV_PATH = "comparison_468_data.csv"


def run(synth, target):
    """Return (gates, lde, distance, duration_s)."""
    start = perf_counter()
    try:
        result = synth.synthesize(target)
        duration = perf_counter() - start
        if result is None:
            return "", "", float("inf"), duration
        gates = (result.gates or "").upper()
        return gates, result.lde, float(result.distance), duration
    except Exception:
        duration = perf_counter() - start
        return "", "", float("inf"), duration


def main():
    set_seed(SEED)
    write_header = not os.path.exists(CSV_PATH)

    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(
                [
                    "epsilon",
                    "method",
                    "trial",
                    "alpha",
                    "beta",
                    "gamma",
                    "lde",
                    "t_count",
                    "o_count",
                    "q_count",
                    "cost",
                    "distance",
                    "duration_ms",
                    "success",
                    "gates",
                ]
            )

        for epsilon in EPSILONS:
            print(f"\n=== epsilon = {epsilon:.0e} ===")
            synths = [
                ("n4_clifford_t", cyclosynth.Synthesizer(epsilon=epsilon)),
                ("n6_clifford_pi6", cyclosynth.Synthesizer(epsilon=epsilon, pi6=True)),
                (
                    "n8_clifford_sqrt_t",
                    cyclosynth.Synthesizer(epsilon=epsilon, sqrt_t=True),
                ),
            ]

            for trial in range(N_TRIALS):
                alpha, beta, gamma = [2 * np.pi * random() for _ in range(3)]
                target = rz(alpha) @ ry(beta) @ rz(gamma)

                for method, synth in synths:
                    gates, lde, dist, dur = run(synth, target)
                    t_count = gates.count("T")
                    o_count = gates.count("R")
                    q_count = gates.count("Q")
                    cost = gate_cost(method, gates) if gates else float("nan")
                    success = dist <= epsilon

                    writer.writerow(
                        [
                            f"{epsilon:.0e}",
                            method,
                            trial,
                            alpha,
                            beta,
                            gamma,
                            lde,
                            t_count,
                            o_count,
                            q_count,
                            f"{cost:.1f}",
                            f"{dist:.6e}",
                            f"{dur * 1000:.3f}",
                            success,
                            gates,
                        ]
                    )
                    f.flush()

                    tag = "OK  " if success else "FAIL"
                    print(
                        f"  trial {trial + 1:>3}  {method:<20}  "
                        f"lde={str(lde):>3}  T={t_count:>3} O={o_count:>3} Q={q_count:>3}  "
                        f"cost={cost:>6.1f}  d={dist:.3e}  "
                        f"{dur * 1000:>9.1f} ms  {tag}"
                    )

    print(f"\nwrote {CSV_PATH}")


if __name__ == "__main__":
    main()
