"""Hyperparameter sweep to pick Clifford+√T defaults at ε = 1e-7 and 1e-8.

For each ε we synthesize a FIXED set of seeded random U3 targets (same
targets across every config so cost ratios are paired) and record, per
(ε, config, target):

    T_count, Q_count, cost = T + 3.5·Q,
    ref_cost   = T_count of the SAME target in Clifford+T (gate-cost ref),
    ratio      = cost / ref_cost,
    wall_ms, distance, success.

Sister script `analyze_sweep_deep_eps.py` reads the CSV and builds the
cost-ratio-vs-runtime Pareto frontier and recommends a default per ε.

Robustness for unattended runs:
  * every synthesize() runs in a child process with a hard timeout, so a
    pathological target is recorded as a timeout instead of hanging the
    sweep;
  * results are flushed per row and the run is resume-safe — re-running
    skips (ε, config, trial) rows already present in the CSV.

Cost model: cost = T_count + 3.5·Q_count (other Cliffords free; Q = √T).
"""

import csv
import multiprocessing as mp
import os
from time import perf_counter

import numpy as np

# ─── Cost model ──────────────────────────────────────────────────────────────

Q_COST = 3.5


def cost_of(t_count: int, q_count: int) -> float:
    return t_count + Q_COST * q_count


# ─── Target generator (fixed, seeded) ────────────────────────────────────────

def rz(t):
    return np.array([[np.exp(-1j * t / 2), 0],
                     [0, np.exp(1j * t / 2)]], dtype=np.complex128)


def ry(t):
    c, s = np.cos(t / 2), np.sin(t / 2)
    return np.array([[c, -s], [s, c]], dtype=np.complex128)


def make_targets(n: int, seed: int):
    """n Haar-random U3 = Rz(α)·Ry(β)·Rz(γ), deterministic from `seed`.

    Haar measure on SU(2) in ZYZ angles is ∝ sin(β) dα dβ dγ: α, γ are
    uniform on [0, 2π), but β must be sine-weighted on [0, π].
    """
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        a, g = rng.uniform(0.0, 2 * np.pi, size=2)
        b = np.arccos(1.0 - 2.0 * rng.uniform(0.0, 1.0))
        out.append(((float(a), float(b), float(g)), rz(a) @ ry(b) @ rz(g)))
    return out


# ─── Config grid ─────────────────────────────────────────────────────────────
# Each config is (name, kwargs) where kwargs are the √T-only knobs passed to
# cyclosynth.Synthesizer alongside epsilon and sqrt_t=True. The Clifford+T
# reference ("ref") uses no knobs.
#
# Axes swept: optimize_cost {first-hit vs anytime min-cost}, lde_window
# {1,2,3} (cost-quality lever, optimize_cost only), deadline_ms {None,
# 3000, 1500, 750} (the runtime/Pareto axis), and a seq_parity probe at the
# deep end. q_cost stays at the canonical 3.5.

def configs_for(epsilon: float):
    if epsilon > 1e-8:
        # 1e-7: affordable — sweep the full lde_window axis uncapped, plus a
        # deadline ladder on the prior-best window=2 to trace the frontier.
        return [
            ("firsthit", {"optimize_cost": False}),
            ("opt_w1", {"optimize_cost": True, "lde_window": 1}),
            ("opt_w2", {"optimize_cost": True, "lde_window": 2}),
            ("opt_w3", {"optimize_cost": True, "lde_window": 3}),
            ("opt_w2_dl3000", {"optimize_cost": True, "lde_window": 2, "deadline_ms": 3000}),
            ("opt_w2_dl1500", {"optimize_cost": True, "lde_window": 2, "deadline_ms": 1500}),
            ("opt_w2_dl750", {"optimize_cost": True, "lde_window": 2, "deadline_ms": 750}),
        ]
    # 1e-8: uncapped optimize_cost is too slow to run across every window, so
    # lean on the deadline ladder (which *is* the Pareto axis here). Keep one
    # uncapped window=2 as the best-achievable-cost anchor, capped window=1/3
    # for the window comparison, and a seq_parity probe (matters < 2.5e-8).
    return [
        ("firsthit", {"optimize_cost": False}),
        ("opt_w2", {"optimize_cost": True, "lde_window": 2}),
        ("opt_w1_dl4000", {"optimize_cost": True, "lde_window": 1, "deadline_ms": 4000}),
        ("opt_w3_dl4000", {"optimize_cost": True, "lde_window": 3, "deadline_ms": 4000}),
        ("opt_w2_dl6000", {"optimize_cost": True, "lde_window": 2, "deadline_ms": 6000}),
        ("opt_w2_dl4000", {"optimize_cost": True, "lde_window": 2, "deadline_ms": 4000}),
        ("opt_w2_dl2000", {"optimize_cost": True, "lde_window": 2, "deadline_ms": 2000}),
        ("opt_w2_seq_dl4000", {"optimize_cost": True, "lde_window": 2, "deadline_ms": 4000, "seq_parity": True}),
    ]


# Order configs cheap→expensive so the fast data lands first.
def config_sort_key(item):
    _, kw = item
    if not kw.get("optimize_cost", False):
        return (0, 0)
    dl = kw.get("deadline_ms")
    return (1, 0 if dl is None else -dl)  # capped (small dl) before uncapped


# ─── Per-call worker (child process with hard timeout) ───────────────────────

def _worker(epsilon, sqrt_t, kwargs, theta, phi, lam, q):
    import cyclosynth  # imported in child to keep parent light

    synth = cyclosynth.Synthesizer(epsilon=epsilon, sqrt_t=sqrt_t, **kwargs)
    t0 = perf_counter()
    try:
        r = synth.synthesize_zyz(theta, phi, lam)
        dur = perf_counter() - t0
        if r is None:
            q.put(("none", dur, 0, 0, float("inf"), 0))
        else:
            q.put(("ok", dur, int(r.t_count), int(r.q_count),
                   float(r.distance), int(r.lde)))
    except Exception as e:  # noqa: BLE001 - record, don't crash the sweep
        dur = perf_counter() - t0
        q.put(("err", dur, 0, 0, float("inf"), 0, repr(e)))


def synth_with_timeout(epsilon, sqrt_t, kwargs, angles, timeout_s):
    """Run one synthesize() in a child; kill + report on timeout."""
    theta, phi, lam = angles
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_worker,
                    args=(epsilon, sqrt_t, kwargs, theta, phi, lam, q))
    t0 = perf_counter()
    p.start()
    p.join(timeout_s)
    if p.is_alive():
        p.terminate()
        p.join()
        return {"status": "timeout", "wall_ms": timeout_s * 1000.0,
                "t": 0, "q": 0, "dist": float("inf"), "lde": 0}
    try:
        msg = q.get_nowait()
    except Exception:
        return {"status": "crash", "wall_ms": (perf_counter() - t0) * 1000.0,
                "t": 0, "q": 0, "dist": float("inf"), "lde": 0}
    status, dur, t, qc, dist, lde = msg[:6]
    return {"status": status, "wall_ms": dur * 1000.0,
            "t": t, "q": qc, "dist": dist, "lde": lde}


# ─── Run config ──────────────────────────────────────────────────────────────

EPSILONS = [1e-7, 1e-8]
N_TARGETS = 30
SEED = 0xC0FFEE
TIMEOUT_S = {1e-7: 90.0, 1e-8: 180.0}
CSV_PATH = "scripts/data/sweep_deep_eps.csv"

HEADER = [
    "epsilon", "config", "trial",
    "alpha", "beta", "gamma",
    "t_count", "q_count", "cost",
    "ref_cost", "ratio",
    "distance", "wall_ms", "status",
]


def load_done(path):
    """Set of (epsilon_str, config, trial) already recorded."""
    done = set()
    if not os.path.exists(path):
        return done
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            done.add((row["epsilon"], row["config"], int(row["trial"])))
    return done


def main():
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    done = load_done(CSV_PATH)
    write_header = not os.path.exists(CSV_PATH)

    with open(CSV_PATH, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(HEADER)

        for epsilon in EPSILONS:
            eps_s = f"{epsilon:.0e}"
            timeout_s = TIMEOUT_S[epsilon]
            targets = make_targets(N_TARGETS, SEED)
            print(f"\n=== ε = {eps_s}  ({N_TARGETS} targets, "
                  f"timeout {timeout_s:.0f}s) ===", flush=True)

            # 1) Clifford+T reference cost per target (= ref T_count).
            ref_cost = {}
            print("  [ref] Clifford+T reference ...", flush=True)
            for i, (angles, _) in enumerate(targets):
                r = synth_with_timeout(epsilon, False, {}, angles, timeout_s)
                ref_cost[i] = cost_of(r["t"], r["q"]) if r["status"] == "ok" else float("nan")
                if ("ref", "ref", i) not in done:
                    w.writerow([
                        eps_s, "ref", i, *angles,
                        r["t"], r["q"], f"{cost_of(r['t'], r['q']):.1f}",
                        f"{ref_cost[i]:.1f}", "1.0",
                        f"{r['dist']:.6e}", f"{r['wall_ms']:.1f}", r["status"],
                    ])
                    f.flush()
                print(f"    ref {i:>2}  T={r['t']:>3}  "
                      f"d={r['dist']:.2e}  {r['wall_ms']:>8.0f}ms  {r['status']}",
                      flush=True)

            # 2) √T configs.
            for name, kwargs in sorted(configs_for(epsilon), key=config_sort_key):
                if all((eps_s, name, i) in done for i in range(N_TARGETS)):
                    print(f"  [{name}] already complete, skipping", flush=True)
                    continue
                print(f"  [{name}] {kwargs}", flush=True)
                for i, (angles, _) in enumerate(targets):
                    if (eps_s, name, i) in done:
                        continue
                    r = synth_with_timeout(epsilon, True, kwargs, angles, timeout_s)
                    cost = cost_of(r["t"], r["q"]) if r["status"] == "ok" else float("nan")
                    rc = ref_cost.get(i, float("nan"))
                    ratio = cost / rc if (rc and rc == rc and r["status"] == "ok") else float("nan")
                    w.writerow([
                        eps_s, name, i, *angles,
                        r["t"], r["q"], f"{cost:.1f}",
                        f"{rc:.1f}", f"{ratio:.4f}",
                        f"{r['dist']:.6e}", f"{r['wall_ms']:.1f}", r["status"],
                    ])
                    f.flush()
                    print(f"    {name} {i:>2}  T={r['t']:>3} Q={r['q']:>3}  "
                          f"cost={cost:>6.1f}  ratio={ratio:>6.3f}  "
                          f"{r['wall_ms']:>8.0f}ms  {r['status']}", flush=True)

    print(f"\nwrote {CSV_PATH}", flush=True)


if __name__ == "__main__":
    main()
