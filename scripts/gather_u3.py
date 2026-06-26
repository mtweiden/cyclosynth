"""Gather U3 synthesis data: cyclosynth Clifford+T, cyclosynth Clifford+sqrt(T)
(cost-optimal), and gridsynth (Ross-Selinger Clifford+T, via the Euler
decomposition) on the *same* Haar-random U3 = Rz(a)Ry(b)Rz(g) targets.

One row per (target, epsilon, synthesizer); long format:

    epsilon, synthesizer, trial, alpha, beta, gamma,
    t_count, q_count, cost, distance, duration_ms, success, gates

with synthesizer in {cyclosynth_t, cyclosynth_sqrt_t, gridsynth}.

gridsynth is skipped automatically if its binary is not installed (the rest of
the data is still gathered). The Clifford+sqrt(T) row is floored at the
Clifford+T cost -- a sqrt(T) circuit is never costlier than the T circuit for
the same target -- while keeping its own synthesis duration.

Cost: n_T + 3 n_sqrt(T) (T states).  Output: scripts/data/u3.csv

Smoke test: GATHER_N=2 GATHER_EPS=1e-3 python3 scripts/gather_u3.py
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
SEED = 42
Q_WEIGHT = 3
OUT = os.environ.get("GATHER_OUT", "scripts/data/u3.csv")
N_TRIALS = int(os.environ.get("GATHER_N", "500"))
_eps_env = os.environ.get("GATHER_EPS")
EPSILONS = ([float(x) for x in _eps_env.split(",")] if _eps_env
            else [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8])
# Guard on the cyclosynth searches (esp. Clifford+sqrt(T) optimize_cost), which
# are expensive at deep eps and can occasionally take far longer than median on
# above-median-depth targets. See scripts/gather_rz.py for the full rationale.
#  1. deadline_ms (sqrt(T) only): README knob; 2000 chosen on the rz sweep
#     (cost plateaus, time keeps climbing past it). Only bites at deep eps.
#  2. CYC_TIMEOUT_S: hard guarantee. Each cyclosynth call runs in a spawn child
#     process the parent SIGKILLs on overrun (a killed call floors to the
#     Clifford+T cost via existing logic). 120s clears observed slow targets
#     with margin while bounding any genuine hang.
SQRT_T_DEADLINE_MS = int(os.environ.get("GATHER_SQRT_T_DEADLINE_MS", "2000"))
CYC_TIMEOUT_S = float(os.environ.get("GATHER_CYC_TIMEOUT_S", "120"))

# ─── gridsynth gate: skip its rows entirely if the binary isn't installed ────
try:
    from gridsynth_real import grid_u3, _find_binary
    _find_binary()
    HAVE_GRID = True
except Exception as e:                                    # noqa: BLE001
    print(f"[gather_u3] gridsynth unavailable ({e}); skipping gridsynth rows.",
          flush=True)
    HAVE_GRID = False


# ─── Target / cost helpers ───────────────────────────────────────────────────
def rz(t):
    return np.array([[np.exp(-1j * t / 2), 0],
                     [0, np.exp(1j * t / 2)]], dtype=np.complex128)


def ry(t):
    c, s = np.cos(t / 2), np.sin(t / 2)
    return np.array([[c, -s], [s, c]], dtype=np.complex128)


def angles_for(eps, n):
    """The n Haar-random (alpha, beta, gamma) Euler angles for a given epsilon,
    seeded from (SEED, eps) so the sample is independent of which epsilons run.
    A full sweep and a `GATHER_EPS=1e-8` rerun therefore hit the *same* targets
    at 1e-8 (lets us re-run one epsilon on identical targets and merge). alpha,
    gamma uniform; beta sine-weighted via arccos(1-2u) for Haar SU(2)."""
    key = int.from_bytes(hashlib.sha256(f"{eps:.0e}".encode()).digest()[:8], "big")
    rng = np.random.default_rng([SEED, key])
    alpha = rng.uniform(0.0, 2 * np.pi, n)
    gamma = rng.uniform(0.0, 2 * np.pi, n)
    beta = np.arccos(1.0 - 2.0 * rng.uniform(0.0, 1.0, n))
    return alpha, beta, gamma


def cost_of(gates):
    """Block-model resource cost in T states (sqrt(T)-class blocks, T^{3/2}=R)."""
    return _cost.block_cost(gates, Q_WEIGHT)


def run_cyc(synth, target):
    """-> dict(t_count, q_count, distance, duration_ms, gates)."""
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


# Hard wall-clock guard for cyclosynth calls. "spawn" (not "fork") because
# cyclosynth runs a rayon threadpool and forking a multithreaded native process
# is unsafe; spawn gives each call a clean interpreter.
_CTX = mp.get_context("spawn")


def _cyc_worker(q, synth_kwargs, target):
    synth = cyclosynth.Synthesizer(**synth_kwargs)
    q.put(run_cyc(synth, target))


def run_cyc_timeout(synth_kwargs, target, timeout_s):
    """run_cyc in a child process, SIGKILLed if it overruns timeout_s. A killed
    call returns a failed (inf-distance, empty-gates) row, as run_cyc does on
    error, so downstream handling is unchanged."""
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


def run_grid(a, b, g, eps):
    t0 = perf_counter()
    try:
        gates, tc, dist = grid_u3(a, b, g, eps)
        return dict(t_count=int(tc), q_count=0, distance=float(dist),
                    duration_ms=(perf_counter() - t0) * 1000.0,
                    gates=(gates or "").upper())
    except Exception:                                     # noqa: BLE001
        return dict(t_count=0, q_count=0, distance=float("inf"),
                    duration_ms=(perf_counter() - t0) * 1000.0, gates="")


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epsilon", "synthesizer", "trial", "alpha", "beta", "gamma",
                    "t_count", "q_count", "cost", "distance", "duration_ms",
                    "success", "gates"])
        for eps in EPSILONS:
            print(f"\n=== eps = {eps:.0e} ===", flush=True)
            kw_t = dict(epsilon=eps)
            kw_q = dict(epsilon=eps, sqrt_t=True, optimize_cost=True,
                        deadline_ms=SQRT_T_DEADLINE_MS)
            alphas, betas, gammas = angles_for(eps, N_TRIALS)
            for trial in range(N_TRIALS):
                alpha, beta, gamma = float(alphas[trial]), float(betas[trial]), float(gammas[trial])
                target = rz(alpha) @ ry(beta) @ rz(gamma)

                rows = {"cyclosynth_t": run_cyc_timeout(kw_t, target, CYC_TIMEOUT_S),
                        "cyclosynth_sqrt_t": run_cyc_timeout(kw_q, target, CYC_TIMEOUT_S)}
                if HAVE_GRID:
                    rows["gridsynth"] = run_grid(alpha, beta, gamma, eps)

                # Never-costlier floor: if raw sqrt(T) is costlier than (or
                # failed relative to) Clifford+T, report the T circuit for the
                # sqrt(T) row, keeping the sqrt(T) synthesis duration.
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
                    w.writerow([f"{eps:.0e}", name, trial, alpha, beta, gamma,
                                d["t_count"], d["q_count"], f"{cost:.1f}",
                                f"{d['distance']:.6e}", f"{d['duration_ms']:.3f}",
                                d["distance"] <= eps, d["gates"]])
                f.flush()
            print(f"  {N_TRIALS} targets done", flush=True)
    print(f"\nwrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
