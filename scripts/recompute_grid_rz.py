"""Recompute only the gridsynth column of the Rz-panel CSV using the real
gridsynth binary (gridsynth_real), leaving the cached cyclosynth columns
(cyc_T, cyc_sqrtT) untouched. Avoids re-running the slow cyclosynth sqrt(T)
synthesis just to refresh the gridsynth baseline.

In:  scripts/data/cost_vs_eps_rz.csv  (eps, theta, cyc_T, cyc_sqrtT, grid_T)
Out: same file, grid_T recomputed with the real gridsynth.
"""
import csv
import sys
import gridsynth_real as gr

RZ_CSV = "scripts/data/cost_vs_eps_rz.csv"


def main():
    rows = list(csv.DictReader(open(RZ_CSV)))
    out = []
    n = len(rows)
    for i, r in enumerate(rows):
        eps = float(r["eps"])
        theta = float(r["theta"])
        gt = gr.grid_rz(theta, eps).count("T")
        r["grid_T"] = gt
        out.append(r)
        if (i + 1) % 50 == 0:
            print(f"{i+1}/{n}", flush=True)
    with open(RZ_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["eps", "theta", "cyc_T", "cyc_sqrtT", "grid_T"])
        w.writeheader()
        w.writerows(out)
    print(f"rewrote {RZ_CSV} ({n} rows) with real gridsynth")


if __name__ == "__main__":
    sys.path.insert(0, "scripts")
    main()
