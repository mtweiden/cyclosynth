"""Gather z-rotation synthesis data: cyclosynth Clifford+T, cyclosynth
Clifford+sqrt(T) (cost-optimal), and gridsynth (Ross-Selinger, the provably
T-optimal method for z-rotations) on the *same* random Rz(theta) targets.

One row per (target, epsilon, synthesizer); long format:

    epsilon, synthesizer, trial, theta,
    t_count, q_count, cost, distance, duration_ms, success, gates

with synthesizer in {cyclosynth_t, cyclosynth_sqrt_t, gridsynth}.

gridsynth is skipped automatically if its binary is not installed. The
Clifford+sqrt(T) row is floored at the Clifford+T cost. Cost = n_T + 3 n_sqrt(T).

Output: scripts/data/rz.csv
Smoke test: GATHER_N=2 GATHER_EPS=1e-3 python3 scripts/gather_rz.py
"""
import csv
import os
from time import perf_counter

import numpy as np
import cyclosynth

# ─── Config (env-overridable for smoke tests) ────────────────────────────────
SEED = 0x5172A          # "RzA"
Q_WEIGHT = 3
OUT = os.environ.get("GATHER_OUT", "scripts/data/rz.csv")
N_TRIALS = int(os.environ.get("GATHER_N", "500"))
_eps_env = os.environ.get("GATHER_EPS")
EPSILONS = ([float(x) for x in _eps_env.split(",")] if _eps_env
            else [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8])

# ─── gridsynth gate ──────────────────────────────────────────────────────────
try:
    from gridsynth_real import grid_rz, to_u, dphase, rz, _find_binary
    _find_binary()
    HAVE_GRID = True
except Exception as e:                                    # noqa: BLE001
    print(f"[gather_rz] gridsynth unavailable ({e}); skipping gridsynth rows.",
          flush=True)
    HAVE_GRID = False

    def rz(t):  # noqa: E306 -- still needed for cyclosynth targets
        return np.array([[np.exp(-1j * t / 2), 0],
                         [0, np.exp(1j * t / 2)]], dtype=np.complex128)


def cost_of(t_count, q_count):
    return t_count + Q_WEIGHT * q_count


def run_cyc(synth, target):
    t0 = perf_counter()
    try:
        r = synth.synthesize(target)
        dur = (perf_counter() - t0) * 1000.0
        if r is None:
            return dict(t_count=0, q_count=0, distance=float("inf"),
                        duration_ms=dur, gates="")
        g = (r.gates or "").upper()
        return dict(t_count=g.count("T"), q_count=g.count("Q"),
                    distance=float(r.distance), duration_ms=dur, gates=g)
    except Exception:                                     # noqa: BLE001
        return dict(t_count=0, q_count=0, distance=float("inf"),
                    duration_ms=(perf_counter() - t0) * 1000.0, gates="")


def run_grid(theta, eps):
    t0 = perf_counter()
    try:
        s = grid_rz(theta, eps)
        dist = dphase(to_u(s), rz(theta))
        return dict(t_count=s.count("T"), q_count=0, distance=float(dist),
                    duration_ms=(perf_counter() - t0) * 1000.0, gates=s.upper())
    except Exception:                                     # noqa: BLE001
        return dict(t_count=0, q_count=0, distance=float("inf"),
                    duration_ms=(perf_counter() - t0) * 1000.0, gates="")


def main():
    rng = np.random.default_rng(SEED)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epsilon", "synthesizer", "trial", "theta",
                    "t_count", "q_count", "cost", "distance", "duration_ms",
                    "success", "gates"])
        for eps in EPSILONS:
            print(f"\n=== eps = {eps:.0e} ===", flush=True)
            synth_t = cyclosynth.Synthesizer(epsilon=eps)
            synth_q = cyclosynth.Synthesizer(epsilon=eps, sqrt_t=True,
                                             optimize_cost=True)
            angles = rng.uniform(0.0, 2 * np.pi, N_TRIALS)
            for trial, theta in enumerate(angles):
                theta = float(theta)
                target = rz(theta)
                rows = {"cyclosynth_t": run_cyc(synth_t, target),
                        "cyclosynth_sqrt_t": run_cyc(synth_q, target)}
                if HAVE_GRID:
                    rows["gridsynth"] = run_grid(theta, eps)

                ct, cq = rows["cyclosynth_t"], rows["cyclosynth_sqrt_t"]
                c_t = cost_of(ct["t_count"], ct["q_count"]) if ct["gates"] else float("inf")
                c_q = cost_of(cq["t_count"], cq["q_count"]) if cq["gates"] else float("inf")
                if c_t < c_q:
                    rows["cyclosynth_sqrt_t"] = dict(ct, duration_ms=cq["duration_ms"])

                for name in ("cyclosynth_t", "cyclosynth_sqrt_t", "gridsynth"):
                    d = rows.get(name)
                    if d is None:
                        continue
                    cost = cost_of(d["t_count"], d["q_count"]) if d["gates"] else float("nan")
                    w.writerow([f"{eps:.0e}", name, trial, theta,
                                d["t_count"], d["q_count"], f"{cost:.1f}",
                                f"{d['distance']:.6e}", f"{d['duration_ms']:.3f}",
                                d["distance"] <= eps, d["gates"]])
                f.flush()
            print(f"  {N_TRIALS} targets done", flush=True)
    print(f"\nwrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
