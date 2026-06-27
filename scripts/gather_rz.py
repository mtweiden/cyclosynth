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
import hashlib
import multiprocessing as mp
import os
from time import perf_counter

import sys

import numpy as np
import cyclosynth

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _cost  # block-model cost (sqrt(T)-class blocks, T^{3/2}=R)

# ─── Config (env-overridable for smoke tests) ────────────────────────────────
SEED = 0x5172A          # "RzA"
Q_WEIGHT = 3
OUT = os.environ.get("GATHER_OUT", "scripts/data/rz.csv")
N_TRIALS = int(os.environ.get("GATHER_N", "500"))
# Two-layer guard on the Clifford+sqrt(T) cost-optimal search, which is
# expensive at deep eps (~22s/target at 1e-8 unbounded) and can occasionally
# run away (observed >11h on one target).
#  1. deadline_ms: the README's recommended knob. 2000 was chosen from an N=25
#     sweep on these Rz 1e-8 angles: cost ratio vs Clifford+T plateaus at ~0.82x
#     for deadline >=2000 (2000/4000/6000 all tie), while time keeps climbing
#     (9.4s/13.3s/17.5s median) -- so extra budget buys time, not quality. 1000
#     was worse (0.86x) and floored 2/25. It only bites at deep eps; shallower
#     searches self-terminate below it. Tune per run via GATHER_SQRT_T_DEADLINE_MS
#     (see angles_for: a 1e-8-only rerun hits the same angles, so you can re-sweep
#     on identical targets). BUT it is only checked between search-depth
#     iterations, so a pathological target stuck inside one iteration sails past
#     it (verified: ran >240s past a 10s deadline) -- hence the second layer.
#  2. CYC_TIMEOUT_S: the hard guarantee. Each cyclosynth call runs in a child
#     process the parent SIGKILLs on overrun; a killed call floors to the
#     Clifford+T cost via existing logic. Set to 120s: a few 1e-8 angles whose
#     solution is a few lde levels above median are slow-but-terminating, not
#     true runaways -- measured up to 97s (both the Clifford+T and sqrt(T)
#     paths; deadline_ms is ineffective on them and they run to completion).
#     60s truncated 2/500 of those; 120s clears the observed ceiling with margin
#     while still bounding any genuine hang. (gridsynth solves the same targets
#     in ~100ms -- cyclosynth's per-lde search cost grows ~exponentially with
#     depth; see [[gather-rz-sqrt-t-hang]].)
SQRT_T_DEADLINE_MS = int(os.environ.get("GATHER_SQRT_T_DEADLINE_MS", "2000"))
CYC_TIMEOUT_S = float(os.environ.get("GATHER_CYC_TIMEOUT_S", "120"))
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


def angles_for(eps, n):
    """The n random angles for a given epsilon, seeded from (SEED, eps) so the
    sample is independent of which epsilons are run. A full sweep and a
    `GATHER_EPS=1e-8` rerun therefore hit the *same* 500 angles at 1e-8 — which
    lets us tune the 1e-8 deadline on identical targets and merge the result
    back without disturbing the other epsilons. (The old code drew all epsilons
    from one sequential RNG, so 1e-8's angles depended on the draw order.)"""
    key = int.from_bytes(hashlib.sha256(f"{eps:.0e}".encode()).digest()[:8], "big")
    return np.random.default_rng([SEED, key]).uniform(0.0, 2 * np.pi, n)


def cost_of(gates):
    """Block-model resource cost in T states (sqrt(T)-class blocks, T^{3/2}=R)."""
    return _cost.block_cost(gates, Q_WEIGHT)


def run_cyc(synth, target):
    t0 = perf_counter()
    try:
        r = synth.synthesize(target)
        dur = (perf_counter() - t0) * 1000.0
        if r is None:
            return dict(t_count=0, q_count=0, distance=float("inf"),
                        duration_ms=dur, gates="")
        # Keep case: lowercase q,t,s are the adjoints Q†,T†,S† (don't upper()).
        g = r.gates or ""
        return dict(t_count=g.count("T") + g.count("t"),
                    q_count=g.count("Q") + g.count("q"),
                    distance=float(r.distance), duration_ms=dur, gates=g)
    except Exception:                                     # noqa: BLE001
        return dict(t_count=0, q_count=0, distance=float("inf"),
                    duration_ms=(perf_counter() - t0) * 1000.0, gates="")


# Hard wall-clock guard for cyclosynth calls. "spawn" (not "fork") because
# cyclosynth runs a rayon threadpool and forking a multithreaded native process
# is unsafe; spawn gives each call a clean interpreter. The parent never runs a
# synthesize() itself, so it can always SIGKILL a runaway child.
_CTX = mp.get_context("spawn")


def _cyc_worker(q, synth_kwargs, target):
    synth = cyclosynth.Synthesizer(**synth_kwargs)
    q.put(run_cyc(synth, target))


def run_cyc_timeout(synth_kwargs, target, timeout_s):
    """run_cyc in a child process, SIGKILLed if it overruns timeout_s.
    A killed/failed call returns a failed (inf-distance, empty-gates) row,
    exactly as run_cyc does on error, so downstream handling is unchanged."""
    t0 = perf_counter()
    q = _CTX.Queue()
    p = _CTX.Process(target=_cyc_worker, args=(q, synth_kwargs, target))
    p.start()
    p.join(timeout_s)
    if p.is_alive():
        p.terminate()
        p.join(5)
        if p.is_alive():
            p.kill()
            p.join()
        return dict(t_count=0, q_count=0, distance=float("inf"),
                    duration_ms=(perf_counter() - t0) * 1000.0, gates="",
                    timed_out=True)
    try:
        return q.get(timeout=5)
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
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epsilon", "synthesizer", "trial", "theta",
                    "t_count", "q_count", "cost", "distance", "duration_ms",
                    "success", "gates"])
        for eps in EPSILONS:
            print(f"\n=== eps = {eps:.0e} ===", flush=True)
            kw_t = dict(epsilon=eps)
            kw_q = dict(epsilon=eps, sqrt_t=True, optimize_cost=True,
                        deadline_ms=SQRT_T_DEADLINE_MS)
            angles = angles_for(eps, N_TRIALS)
            for trial, theta in enumerate(angles):
                theta = float(theta)
                target = rz(theta)
                rows = {"cyclosynth_t": run_cyc_timeout(kw_t, target, CYC_TIMEOUT_S),
                        "cyclosynth_sqrt_t": run_cyc_timeout(kw_q, target, CYC_TIMEOUT_S)}
                if HAVE_GRID:
                    rows["gridsynth"] = run_grid(theta, eps)

                ct, cq = rows["cyclosynth_t"], rows["cyclosynth_sqrt_t"]
                c_t = cost_of(ct["gates"]) if ct["gates"] else float("inf")
                c_q = cost_of(cq["gates"]) if cq["gates"] else float("inf")
                if c_t < c_q:
                    rows["cyclosynth_sqrt_t"] = dict(ct, duration_ms=cq["duration_ms"])

                for name in ("cyclosynth_t", "cyclosynth_sqrt_t", "gridsynth"):
                    d = rows.get(name)
                    if d is None:
                        continue
                    cost = cost_of(d["gates"]) if d["gates"] else float("nan")
                    w.writerow([f"{eps:.0e}", name, trial, theta,
                                d["t_count"], d["q_count"], f"{cost:.1f}",
                                f"{d['distance']:.6e}", f"{d['duration_ms']:.3f}",
                                d["distance"] <= eps, d["gates"]])
                f.flush()
            print(f"  {N_TRIALS} targets done", flush=True)
    print(f"\nwrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
