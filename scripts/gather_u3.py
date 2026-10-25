"""Gather the general-unitary data behind the paper's Fig. 1, Fig. 5 (violin),
Fig. 6 (cost-weight sweep), Table 2 (cost) and Table 3 (runtime).

For each epsilon, 500 Haar-random targets V = Rz(alpha) Ry(beta) Rz(gamma)
are synthesized by:

    cyclosynth_t       Clifford+T, joint lattice search (Algorithm 1)
    cyclosynth_sqrt_t  Clifford+sqrt(T), Algorithm 2 with the library's default
                       cost-search deadline. The library runs the Clifford+T
                       floor inside this call, so its result and runtime
                       already include it.
    gridsynth          Ross-Selinger on the three Euler rotations (needs the
                       `gridsynth` binary; skipped if absent)

Long-format CSV, one row per (epsilon, trial, synthesizer). Cost is
n_T + 3 n_sqrt(T) over syllables (scripts/_cost.py). Each cyclosynth call is
timed in its own process (scripts/_gather.py); the two cyclosynth calls run
one after the other, each with the whole machine. Resumable: rerunning skips
rows already in the CSV.

    python3 scripts/gather_u3.py                       # full run
    GATHER_N=2 GATHER_EPS=1e-3 GATHER_OUT=/tmp/u3.csv python3 scripts/gather_u3.py

Output: scripts/data/u3.csv (+ u3.csv.meta.json)
"""
import hashlib
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _cost     # noqa: E402
import _gather   # noqa: E402

SEED = 42
Q_WEIGHT = 3
OUT = os.environ.get("GATHER_OUT", "scripts/data/u3.csv")
N_TRIALS = int(os.environ.get("GATHER_N", "500"))
_eps_env = os.environ.get("GATHER_EPS")
EPSILONS = ([float(x) for x in _eps_env.split(",")] if _eps_env
            else [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8])

try:
    from gridsynth_real import grid_u3, _find_binary
    _find_binary()
    HAVE_GRID = True
except Exception as e:                                    # noqa: BLE001
    print(f"[gather_u3] gridsynth unavailable ({e}); skipping gridsynth rows.", flush=True)
    HAVE_GRID = False

FIELDS = ["epsilon", "trial", "alpha", "beta", "gamma"] + _gather.FIELDS_COMMON


def angles_for(eps, n):
    """The n Haar-random (alpha, beta, gamma) for one epsilon, seeded from
    (SEED, eps) so each epsilon's sample does not depend on which others run.
    alpha, gamma uniform; beta = arccos(1 - 2u) for Haar SU(2)."""
    key = int.from_bytes(hashlib.sha256(f"{eps:.0e}".encode()).digest()[:8], "big")
    rng = np.random.default_rng([SEED, key])
    alpha = rng.uniform(0.0, 2 * np.pi, n)
    gamma = rng.uniform(0.0, 2 * np.pi, n)
    beta = np.arccos(1.0 - 2.0 * rng.uniform(0.0, 1.0, n))
    return alpha, beta, gamma


def row(eps, trial, ang, name, gates, target, duration_ms, timed_out=False):
    t, q = _gather.counts(gates)
    if gates:
        dist = _gather.diamond(_gather.unitary_of(gates), target)
        cost = _cost.syllable_cost(gates, Q_WEIGHT)
    else:
        dist, cost = float("inf"), float("nan")
    ok = bool(gates) and dist < eps
    return dict(epsilon=f"{eps:.0e}", trial=trial, alpha=ang[0], beta=ang[1], gamma=ang[2],
                synthesizer=name, t_count=t, q_count=q, cost=f"{cost:.1f}",
                distance=f"{dist:.6e}", verified=ok, duration_ms=f"{duration_ms:.3f}",
                timed_out=timed_out, success=ok, gates=gates)


def main():
    _gather.write_metadata(OUT, dict(epsilons=EPSILONS, n=N_TRIALS, seed=SEED,
                                     q_weight=Q_WEIGHT, sqrt_t_deadline="library default",
                                     timeouts_s=_gather.TIMEOUTS_S["u3"]))
    done = _gather.load_done(OUT)
    f, w = _gather.open_writer(OUT, FIELDS)
    for eps in EPSILONS:
        print(f"\n=== eps = {eps:.0e} ===", flush=True)
        alphas, betas, gammas = angles_for(eps, N_TRIALS)
        kw_t = dict(epsilon=eps, near_clifford_escape=False)
        kw_q = dict(epsilon=eps, sqrt_t=True, near_clifford_escape=False)
        for trial in range(N_TRIALS):
            if (f"{eps:.0e}", str(trial), "cyclosynth_sqrt_t") in done:
                continue
            a, b, g = float(alphas[trial]), float(betas[trial]), float(gammas[trial])
            ang = (a, b, g)
            target = _gather.rz(a) * _gather.ry(b) * _gather.rz(g)
            # synthesize_u3(theta, phi, lam) targets Rz(phi) Ry(theta) Rz(lam).
            u3_args = (b, a, g)
            rt = _gather.run_cyc(kw_t, "synthesize_u3", u3_args, _gather.timeout_for("u3", "cyclosynth_t", eps))
            rq = _gather.run_cyc(kw_q, "synthesize_u3", u3_args, _gather.timeout_for("u3", "cyclosynth_sqrt_t", eps))
            r_t = row(eps, trial, ang, "cyclosynth_t", rt["gates"], target,
                      rt["duration_ms"], rt.get("timed_out", False))
            r_q = row(eps, trial, ang, "cyclosynth_sqrt_t", _gather.sqrt_t_gates(rq, rt), target,
                      rq["duration_ms"], rq.get("timed_out", False))
            out = [r_t, r_q]
            if HAVE_GRID:
                gates, _, _ = grid_u3(a, b, g, eps)
                # gridsynth first: the sqrt(T) row is the resume marker.
                out.insert(0, row(eps, trial, ang, "gridsynth", gates, target, 0.0))
            for r in out:
                w.writerow(r)
            f.flush()
            print(f"  [{eps:.0e} #{trial}] T={r_t['cost']} ({float(r_t['duration_ms'])/1e3:.1f}s)"
                  f"  sqrtT={r_q['cost']} ({float(r_q['duration_ms'])/1e3:.1f}s)", flush=True)
    f.close()
    print(f"\nwrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
