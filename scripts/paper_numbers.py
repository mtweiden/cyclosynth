"""Every number the paper quotes from the experiments, computed from the
gather CSVs. Run after gather_u3.py and gather_rz.py:

    python3 scripts/paper_numbers.py            # PAPER_STAT=mean for means

Sections map to the paper: Table 2 (cost), Fig. 1 fits, Clifford-gate counts,
the gridsynth comparison (Sec. 5.3), Table 3 (runtime), the c* crossover
(Sec. 6), the catalyst break-even (Sec. 2.3) and the z-rotation sqrt(T)
slope (Sec. 6). It refuses to report from data with failed or unverified
rows, and prints how many calls hit the per-call timeout.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _cost    # noqa: E402
import _stats   # noqa: E402

U3 = os.environ.get("U3_CSV", "scripts/data/u3.csv")
RZ = os.environ.get("RZ_CSV", "scripts/data/rz.csv")
C_REF = 3
CLIFFORDS = set("HSXYZs")


def check(data, name):
    """Abort on any failed or unverified row; print sample sizes and how many
    calls hit the per-call timeout (sqrt(T) ones report the Clifford+T floor)."""
    bad = []
    for eps, trials in sorted(data.items(), reverse=True):
        per, tout = {}, {}
        for trial, d in trials.items():
            for synth, r in d.items():
                per[synth] = per.get(synth, 0) + 1
                tout[synth] = tout.get(synth, 0) + (r["timed_out"] == "True")
                if r["success"] != "True" or r["verified"] != "True":
                    bad.append((eps, trial, synth))
        print(f"  [{name}] {eps:.0e}: " + ", ".join(f"{k}={v}" for k, v in sorted(per.items()))
              + "  timed out: " + ", ".join(f"{k}={v}" for k, v in sorted(tout.items()) if v)
              if any(tout.values()) else
              f"  [{name}] {eps:.0e}: " + ", ".join(f"{k}={v}" for k, v in sorted(per.items())))
    if bad:
        print(f"  [{name}] {len(bad)} bad rows, e.g. {bad[:5]}")
        sys.exit(1)


def meta(path):
    p = path + ".meta.json"
    if os.path.exists(p):
        m = json.load(open(p))
        print(f"  {os.path.basename(path)}: {m['cpu']} ({m['logical_cpus']} logical CPUs, "
              f"rayon={m['rayon_num_threads']}), commit {m['git_commit'][:7]}"
              f"{' (dirty src)' if m['git_dirty'] else ''}")


def pairs(data, eps, a="cyclosynth_t", b="cyclosynth_sqrt_t"):
    return [(d[a], d[b]) for d in data[eps].values() if a in d and b in d]


def section(title):
    print(f"\n== {title} ==")


def main():
    u3 = _stats.load(U3)
    rz = _stats.load(RZ) if os.path.exists(RZ) else None
    epss = sorted(u3, reverse=True)
    print(f"statistic for rho / c*: {_stats.STAT}")
    section("data")
    meta(U3)
    if rz:
        meta(RZ)
    check(u3, "u3")
    if rz:
        check(rz, "rz")

    section("Table 2: cost, Clifford+sqrt(T) vs Clifford+T (u3)")
    print("  eps    rho    %cheaper  worse  med cost_T  med n_T  med n_sqrtT")
    rho_c, medT, nT_q, nR_q = {}, {}, {}, {}
    for e in epss:
        P = pairs(u3, e)
        ct = np.array([float(t["cost"]) for t, _ in P])
        cq = np.array([float(q["cost"]) for _, q in P])
        cls = np.array([_cost.syllable_classes(q["gates"]) for _, q in P])
        rho_c[e] = _stats.center(cq / ct)
        medT[e] = float(np.median(ct))
        nT_q[e], nR_q[e] = float(np.median(cls[:, 0])), float(np.median(cls[:, 1]))
        print(f"  {e:.0e}  {rho_c[e]:.3f}  {100*np.mean(cq < ct):6.1f}%  {int(np.sum(cq > ct)):5d}"
              f"  {medT[e]:10.1f}  {nT_q[e]:7.0f}  {nR_q[e]:11.0f}")

    section("Fig. 1: least-squares fits, mean cost = C log2(1/eps) + b (u3)")
    fits = {}
    for s in ("gridsynth", "cyclosynth_t", "cyclosynth_sqrt_t"):
        es = [e for e in epss if _stats.costs(u3, e, s)]
        if not es:
            continue
        fits[s] = _stats.fit_log2(es, [np.mean(_stats.costs(u3, e, s)) for e in es])
        print(f"  {s:18s} C = {fits[s][0]:.3f}  b = {fits[s][1]:+.2f}")
    Ct, Cq = fits["cyclosynth_t"][0], fits["cyclosynth_sqrt_t"][0]
    print(f"  slope reduction sqrt(T) vs T: {100*(1 - Cq/Ct):.1f}%")

    section("Clifford-gate counts (median, u3)")
    for e in (epss[0], epss[-1]):
        P = pairs(u3, e)
        nq = np.median([sum(c in CLIFFORDS for c in q["gates"]) for _, q in P])
        nt = np.median([sum(c in CLIFFORDS for c in t["gates"]) for t, _ in P])
        print(f"  {e:.0e}: sqrt(T) {nq:.0f} vs T {nt:.0f}")

    if "gridsynth" in fits:
        section("gridsynth on general targets: mean-cost ratios (u3)")
        rt, rq = [], []
        for e in epss:
            g = np.mean(_stats.costs(u3, e, "gridsynth"))
            rt.append(g / np.mean(_stats.costs(u3, e, "cyclosynth_t")))
            rq.append(g / np.mean(_stats.costs(u3, e, "cyclosynth_sqrt_t")))
        print(f"  Clifford+T cheaper by {min(rt):.2f}-{max(rt):.2f}x; "
              f"Clifford+sqrt(T) by {min(rq):.2f}-{max(rq):.2f}x")

    section("Table 3: median runtime (u3)")
    print("  eps    T(ms)  sqrtT(ms)  sqrtT max(ms)")
    for e in epss:
        P = pairs(u3, e)
        dt = np.array([float(t["duration_ms"]) for t, _ in P])
        dq = np.array([float(q["duration_ms"]) for _, q in P])
        print(f"  {e:.0e}  {np.median(dt):7.1f}  {np.median(dq):9.1f}  {dq.max():13.1f}")

    section(f"Sec. 6: crossover c* ({_stats.STAT} rho(c) = 1, u3)")
    cs = {}
    for e in epss:
        P = pairs(u3, e)
        cls = [(_cost.syllable_classes(t["gates"]), _cost.syllable_classes(q["gates"])) for t, q in P]

        def rho(c):
            return _stats.center([(nq[0] + c * nq[1]) / (nt[0] + c * nt[1]) for nt, nq in cls])
        lo, hi = 1.0, 20.0
        for _ in range(60):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if rho(mid) < 1 else (lo, mid)
        cs[e] = lo
        print(f"  {e:.0e}: c* = {lo:.2f}")
    print(f"  range {min(cs.values()):.1f}-{max(cs.values()):.1f}")
    x = np.log10(1 / np.array(epss))
    gT = np.polyfit(x, [medT[e] for e in epss], 1)[0]
    gt = np.polyfit(x, [nT_q[e] for e in epss], 1)[0]
    gr = np.polyfit(x, [nR_q[e] for e in epss], 1)[0]
    print(f"  per-decade growth of medians: n_T(U_T) {gT:.2f}, n_T(U_sqrtT) {gt:.2f}, "
          f"n_sqrtT(U_sqrtT) {gr:.2f} -> c* limit {(gT - gt)/gr:.2f}")

    section("Sec. 2.3: catalyst break-even at the deepest eps")
    e = epss[-1]
    startup = 2 * C_REF * np.log2(1 / e)         # 3 log2(1/eps^2)
    saving = (1 - rho_c[e]) * medT[e]
    print(f"  {e:.0e}: startup {startup:.1f} T states, saving {100*(1-rho_c[e]):.0f}% "
          f"= {saving:.1f} per op -> break-even after {startup/saving:.1f} ops")

    if rz:
        section("Sec. 5.3: Clifford+T (lattice) vs gridsynth on z-rotations (rz)")
        for e in sorted(rz, reverse=True):
            P = pairs(rz, e, "gridsynth", "cyclosynth_t")
            g = np.array([float(a["cost"]) for a, _ in P])
            t = np.array([float(b["cost"]) for _, b in P])
            print(f"  {e:.0e}: n={len(P)}  worse {int(np.sum(t > g))}  better {int(np.sum(t < g))}"
                  f"  equal {int(np.sum(t == g))}  median ratio {np.median(t/g):.3f}"
                  f"  mean diff {np.mean(t - g):+.2f}  max worse {np.max(t - g):+.0f}")
        section("Sec. 6: Clifford+sqrt(T) slope on z-rotations (rz)")
        for s in ("gridsynth", "cyclosynth_t", "cyclosynth_sqrt_t"):
            es = sorted(rz, reverse=True)
            C, b = _stats.fit_log2(es, [np.mean(_stats.costs(rz, e, s)) for e in es])
            print(f"  {s:18s} C = {C:.3f}  b = {b:+.2f}")


if __name__ == "__main__":
    main()
