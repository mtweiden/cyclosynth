"""Gather the z-rotation data behind the paper's gridsynth validation
(Sec. 5.3) and the Clifford+sqrt(T) z-rotation slope (Sec. 6).

For each epsilon, 500 random Rz(theta) targets are synthesized by:

    cyclosynth_t       Clifford+T through the GENERAL lattice search
                       (native_rz=False: the native Ross-Selinger route the
                       library normally takes for diagonal targets is off)
    cyclosynth_sqrt_t  Clifford+sqrt(T) through the general lattice search,
                       Algorithm 2 with the library's default deadline
    gridsynth          Ross-Selinger (`gridsynth` binary), tolerance tuned so
                       the achieved diamond distance is just within epsilon

Same CSV conventions as gather_u3.py (scripts/_gather.py): long format,
independently re-verified distances, per-process timing, resumable.

    python3 scripts/gather_rz.py
    GATHER_N=2 GATHER_EPS=1e-3 GATHER_OUT=/tmp/rz.csv python3 scripts/gather_rz.py

Output: scripts/data/rz.csv (+ rz.csv.meta.json)
"""
import hashlib
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _cost     # noqa: E402
import _gather   # noqa: E402

SEED = 0x5172A          # "RzA"
Q_WEIGHT = 3
OUT = os.environ.get("GATHER_OUT", "scripts/data/rz.csv")
N_TRIALS = int(os.environ.get("GATHER_N", "500"))
_eps_env = os.environ.get("GATHER_EPS")
EPSILONS = ([float(x) for x in _eps_env.split(",")] if _eps_env
            else [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8])

try:
    from gridsynth_real import grid_rz, _find_binary
    _find_binary()
    HAVE_GRID = True
except Exception as e:                                    # noqa: BLE001
    print(f"[gather_rz] gridsynth unavailable ({e}); skipping gridsynth rows.", flush=True)
    HAVE_GRID = False

FIELDS = ["epsilon", "trial", "theta"] + _gather.FIELDS_COMMON


def angles_for(eps, n):
    """The n random angles for one epsilon, seeded from (SEED, eps)."""
    key = int.from_bytes(hashlib.sha256(f"{eps:.0e}".encode()).digest()[:8], "big")
    return np.random.default_rng([SEED, key]).uniform(0.0, 2 * np.pi, n)


def row(eps, trial, theta, name, gates, target, duration_ms, timed_out=False):
    t, q = _gather.counts(gates)
    if gates:
        dist = _gather.diamond(_gather.unitary_of(gates), target)
        cost = _cost.syllable_cost(gates, Q_WEIGHT)
    else:
        dist, cost = float("inf"), float("nan")
    ok = bool(gates) and dist < eps
    return dict(epsilon=f"{eps:.0e}", trial=trial, theta=theta, synthesizer=name,
                t_count=t, q_count=q, cost=f"{cost:.1f}", distance=f"{dist:.6e}",
                verified=ok, duration_ms=f"{duration_ms:.3f}", timed_out=timed_out,
                success=ok, gates=gates)


def main():
    _gather.write_metadata(OUT, dict(epsilons=EPSILONS, n=N_TRIALS, seed=SEED,
                                     q_weight=Q_WEIGHT, sqrt_t_deadline="library default",
                                     native_rz=False, timeouts_s=_gather.TIMEOUTS_S["rz"]))
    done = _gather.load_done(OUT)
    f, w = _gather.open_writer(OUT, FIELDS)
    for eps in EPSILONS:
        print(f"\n=== eps = {eps:.0e} ===", flush=True)
        kw_t = dict(epsilon=eps, native_rz=False, near_clifford_escape=False)
        kw_q = dict(epsilon=eps, sqrt_t=True, native_rz=False,
                    near_clifford_escape=False)
        for trial, theta in enumerate(angles_for(eps, N_TRIALS)):
            if (f"{eps:.0e}", str(trial), "cyclosynth_sqrt_t") in done:
                continue
            theta = float(theta)
            target = _gather.rz(theta)
            # synthesize_u3(0, theta, 0) targets Rz(theta).
            args = (0.0, theta, 0.0)
            rt = _gather.run_cyc(kw_t, "synthesize_u3", args, _gather.timeout_for("rz", "cyclosynth_t", eps))
            rq = _gather.run_cyc(kw_q, "synthesize_u3", args, _gather.timeout_for("rz", "cyclosynth_sqrt_t", eps))
            out = [row(eps, trial, theta, "cyclosynth_t", rt["gates"], target,
                       rt["duration_ms"], rt.get("timed_out", False)),
                   row(eps, trial, theta, "cyclosynth_sqrt_t", _gather.sqrt_t_gates(rq, rt), target,
                       rq["duration_ms"], rq.get("timed_out", False))]
            if HAVE_GRID:
                out.insert(0, row(eps, trial, theta, "gridsynth", grid_rz(theta, eps),
                                  target, 0.0))
            for r in out:
                w.writerow(r)
            f.flush()
            print(f"  [{eps:.0e} #{trial}] " + "  ".join(
                f"{r['synthesizer']}={r['cost']}" for r in out), flush=True)
    f.close()
    print(f"\nwrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
