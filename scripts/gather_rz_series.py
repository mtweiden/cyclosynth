"""Gather the Rz-panel data for the two-panel cost-vs-precision figure:
for a fixed set of random z-rotation angles, synthesize each with cyclosynth
Clifford+T, cyclosynth Clifford+sqrt(T) (optimize_cost), and gridsynth
(optimal Clifford+T), at every precision.

Cost: T-count for the Clifford+T methods; T + 3*sqrt(T) for cyclosynth sqrt(T).
Output: scripts/data/cost_vs_eps_rz.csv  (eps, theta, cyc_T, cyc_sqrtT, grid_T)

The U3 panel reuses scripts/data/cost_vs_eps_3way.csv (already gathered).
"""
import csv
import sys
import numpy as np
import cyclosynth
from comparison_gridsynth import grid_rz, rz

OUT_CSV = "scripts/data/cost_vs_eps_rz.csv"
EPSS = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
# Capped at deep eps where sqrt(T) + gridsynth are slow (~20 s/target).
N_BY_EPS = {1e-3: 120, 1e-4: 120, 1e-5: 120, 1e-6: 80, 1e-7: 60, 1e-8: 40}


def cyc_T(U, eps):
    r = cyclosynth.Synthesizer(epsilon=eps).synthesize(U)
    return r.t_count if r else None


def cyc_Q(U, eps):
    r = cyclosynth.Synthesizer(epsilon=eps, sqrt_t=True,
                               optimize_cost=True).synthesize(U)
    return r.cost if r else None


def main():
    rng = np.random.default_rng(0x5172A)  # "RzA"
    rowsout = []
    for eps in EPSS:
        n = N_BY_EPS[eps]
        angles = rng.uniform(0, 2 * np.pi, n)
        done = 0
        for th in angles:
            U = rz(th)
            ct = cyc_T(U, eps)
            qt = cyc_Q(U, eps)
            if ct is None or qt is None:
                continue
            gt = grid_rz(th, eps).count("T")
            rowsout.append((eps, th, ct, min(qt, ct), gt))
            done += 1
        print(f"eps={eps:.0e}: {done}/{n} targets", flush=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["eps", "theta", "cyc_T", "cyc_sqrtT", "grid_T"])
        w.writerows(rowsout)
    print(f"saved {OUT_CSV}")


if __name__ == "__main__":
    main()
