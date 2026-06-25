"""Two-panel T-cost comparison vs synthesis precision.

Left panel  (a): z-rotations Rz(theta) -- gridsynth's native, provably
                 T-optimal case. Validates that cyclosynth Clifford+T matches
                 gridsynth, and that Clifford+sqrt(T) beats both.
Right panel (b): general unitaries U3 = Rz Ry Rz -- gridsynth must
                 Euler-decompose into three z-rotations (via the Rz-sqrt(X)
                 form), while cyclosynth synthesizes the unitary directly.

X = synthesis epsilon (log). Y = cost in T states (T-count for the
Clifford+T methods; T + 3*sqrt(T) for cyclosynth sqrt(T)). Each method is drawn
as a mean line with a +/- 1 standard-deviation band.

Data:
  Rz panel -- scripts/data/cost_vs_eps_rz.csv   (gather_rz_series.py)
  U3 panel -- scripts/data/cost_vs_eps_3way.csv (this script, U3 path)
The U3 cyclosynth costs are read from the main paired benchmark CSV (same Haar
U3 targets); gridsynth is computed here on those same targets.
"""
import csv
import collections
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _plotstyle                                  # noqa: E402
_plotstyle.apply()

MAIN_CSV = "scripts/data/comparison_sqrtt_data.csv"
RZ_CSV = "scripts/data/cost_vs_eps_rz.csv"
OUT_PNG = "scripts/data/cost_vs_eps_3way.pdf"  # vector; .svg also written
OUT_CSV = "scripts/data/cost_vs_eps_3way.csv"
EPSS = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
# With the real gridsynth (fast) we can use all available cyclosynth targets.
N_BY_EPS = {1e-3: 500, 1e-4: 500, 1e-5: 500, 1e-6: 500, 1e-7: 500, 1e-8: 500}

# Shared per-algorithm palette (see _plotstyle): cyclosynth T = blue,
# cyclosynth sqrt(T) = reddish-purple, gridsynth = orange.
STYLES = {
    "grid_T":    ("gridsynth (Clifford+$T$)",         _plotstyle.GRIDSYNTH,      "o"),
    "cyc_T":     (r"cyclosynth (Clifford+$T$)",        _plotstyle.CLIFFORD_T,     "s"),
    "cyc_sqrtT": (r"cyclosynth (Clifford+$\sqrt{T}$)", _plotstyle.CLIFFORD_SQRT_T, "^"),
}


def load_cyclosynth():
    """data[eps][(a,b,g)] = (cyc_T_cost, cyc_sqrtT_cost) for the U3 panel."""
    rows = list(csv.DictReader(open(MAIN_CSV)))
    by = collections.defaultdict(dict)
    for r in rows:
        if r["success"] != "True":
            continue
        key = (float(r["epsilon"]), r["alpha"], r["beta"], r["gamma"])
        by[key][r["method"]] = r
    out = collections.defaultdict(dict)
    for (eps, a, b, g), d in by.items():
        if "clifford_t" in d and "clifford_sqrt_t" in d:
            ct = int(d["clifford_t"]["t_count"])
            qt = (int(d["clifford_sqrt_t"]["t_count"])
                  + 3 * int(d["clifford_sqrt_t"]["q_count"]))
            out[eps][(float(a), float(b), float(g))] = (ct, min(qt, ct))
    return out


def compute_u3_series():
    from gridsynth_real import grid_u3
    cyc = load_cyclosynth()
    series = {k: collections.defaultdict(list)
              for k in ("cyc_T", "cyc_sqrtT", "grid_T")}
    rowsout = []
    for eps in EPSS:
        targets = list(cyc[eps].keys())[:N_BY_EPS[eps]]
        print(f"U3 eps={eps:.0e}: {len(targets)} targets")
        for (a, b, g) in targets:
            ct, qt = cyc[eps][(a, b, g)]
            gt = grid_u3(a, b, g, eps)[1]
            series["cyc_T"][eps].append(ct)
            series["cyc_sqrtT"][eps].append(qt)
            series["grid_T"][eps].append(gt)
            rowsout.append((eps, a, b, g, ct, qt, gt))
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["eps", "alpha", "beta", "gamma", "cyc_T", "cyc_sqrtT", "grid_T"])
        w.writerows(rowsout)
    return series


def load_u3_series():
    """Re-plot the U3 panel from the saved per-target CSV (no recompute)."""
    series = {k: collections.defaultdict(list)
              for k in ("cyc_T", "cyc_sqrtT", "grid_T")}
    for r in csv.DictReader(open(OUT_CSV)):
        eps = float(r["eps"])
        series["cyc_T"][eps].append(float(r["cyc_T"]))
        series["cyc_sqrtT"][eps].append(float(r["cyc_sqrtT"]))
        series["grid_T"][eps].append(float(r["grid_T"]))
    return series


def load_rz_series():
    series = {k: collections.defaultdict(list)
              for k in ("cyc_T", "cyc_sqrtT", "grid_T")}
    for r in csv.DictReader(open(RZ_CSV)):
        eps = float(r["eps"])
        series["cyc_T"][eps].append(float(r["cyc_T"]))
        series["cyc_sqrtT"][eps].append(float(r["cyc_sqrtT"]))
        series["grid_T"][eps].append(float(r["grid_T"]))
    return series


LOG2_10 = np.log2(10.0)  # convert C*log2(1/eps) to plot slope over x=-log10(eps)


def add_ref_lines(ax, specs, x0=3.0, x1=8.0):
    """Draw dashed C*log2(1/eps) reference lines with on-line, rotated labels.

    specs: list of (C, text, label_x, perp_pts). perp_pts offsets the label
    perpendicular to the line (+ pushes it above/left, - below/right) so the
    text sits in whitespace rather than on the plotted data. Called after the
    final layout so the data->display angle is correct; limits are frozen first.
    """
    ax.set_autoscale_on(False)
    for C, text, lx, perp in specs:
        y0, y1 = C * LOG2_10 * x0, C * LOG2_10 * x1
        ax.plot([x0, x1], [y0, y1], ls=(0, (6, 3)), color="0.45",
                lw=1.2, zorder=1.5)
        p0 = ax.transData.transform((x0, y0))
        p1 = ax.transData.transform((x1, y1))
        ang = np.degrees(np.arctan2(p1[1] - p0[1], p1[0] - p0[0]))
        ar = np.radians(ang)
        off = (-np.sin(ar) * perp, np.cos(ar) * perp)
        ax.annotate(text, xy=(lx, C * LOG2_10 * lx), xytext=off,
                    textcoords="offset points", rotation=ang,
                    rotation_mode="anchor", ha="center", va="center",
                    fontsize=11, color="0.30", zorder=6)


def draw(ax, series, title):
    epss = [e for e in EPSS if series["cyc_T"].get(e)]
    xs = np.array([-np.log10(e) for e in epss])
    handles = []
    for key, (label, color, marker) in STYLES.items():
        mean = np.array([np.mean(series[key][e]) for e in epss])
        std = np.array([np.std(series[key][e]) for e in epss])
        ax.fill_between(xs, mean - std, mean + std, color=color, alpha=0.18, lw=0)
        (h,) = ax.plot(xs, mean, "-", marker=marker, color=color, lw=2, ms=5.5, label=label)
        handles.append(h)
    ax.set_xticks(xs)
    ax.set_xticklabels([rf"$10^{{-{int(x)}}}$" for x in xs])
    ax.set_xlabel(r"Precision  $\varepsilon$")
    ax.set_title(title, fontsize=12)
    ax.yaxis.set_major_locator(MultipleLocator(40))
    ax.yaxis.set_minor_locator(MultipleLocator(10))
    ax.grid(alpha=0.25)
    ax.tick_params(axis="y", which="minor", length=2.5, color="0.6")
    return handles


def main():
    replot = "replot" in sys.argv and os.path.exists(OUT_CSV)
    u3 = load_u3_series() if replot else compute_u3_series()
    rz = load_rz_series() if os.path.exists(RZ_CSV) else None

    if rz is None:
        print(f"WARNING: {RZ_CSV} missing -- run gather_rz_series.py; "
              "drawing U3 panel only.")
        fig, ax2 = plt.subplots(figsize=(6.4, 4.6))
        handles = draw(ax2, u3, "general unitaries $U$")
        ax2.set_ylabel(r"Cost ($T$ states)")
        axes_for_legend = ax2
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.4, 3.6),
                                       sharey=False)
        handles = draw(ax1, rz, r"(a)  $z$-rotations $R_z(\theta)$")
        draw(ax2, u3, r"(b)  general unitaries $U$")
        ax1.set_ylabel(r"Cost ($T$ states)")
        axes_for_legend = ax1

    fig.legend(handles, [h.get_label() for h in handles],
               loc="upper center", ncol=3, fontsize=10.5,
               framealpha=0.95, handlelength=1.6, columnspacing=1.4,
               bbox_to_anchor=(0.5, 1.005))
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    # Dashed C*log2(1/eps) reference slopes with on-line labels (after layout
    # so the rotation angle matches the final aspect; not in the legend).
    if rz is not None:
        add_ref_lines(ax1, [
            (3.0, r"$3\log_2(1/\varepsilon)$", 5.4, 11),
            (2.6, r"$2.6\log_2(1/\varepsilon)$", 5.4, -13),
        ])
    add_ref_lines(ax2, [
        (9.0, r"$9\log_2(1/\varepsilon)$", 4.6, 12),
        (3.0, r"$3\log_2(1/\varepsilon)$", 6.6, 12),
        (2.6, r"$2.6\log_2(1/\varepsilon)$", 7.3, -19),
    ])
    fig.savefig(OUT_PNG, bbox_inches="tight")
    fig.savefig(OUT_PNG.replace(".pdf", ".svg"), bbox_inches="tight")
    print(f"saved {OUT_PNG} (+ .svg)")
    for panel, s in (("Rz", rz), ("U3", u3)):
        if s is None:
            continue
        print(f"  [{panel}]")
        for e in EPSS:
            if not s["cyc_T"].get(e):
                continue
            print(f"    {e:.0e}: grid {np.mean(s['grid_T'][e]):.1f}  "
                  f"cycT {np.mean(s['cyc_T'][e]):.1f}  "
                  f"cycQ {np.mean(s['cyc_sqrtT'][e]):.1f}")


if __name__ == "__main__":
    main()
