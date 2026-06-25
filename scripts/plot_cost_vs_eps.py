"""Two-panel T-cost comparison vs synthesis precision (Fig 3).

(a) z-rotations Rz(theta) -- gridsynth's native, provably T-optimal case.
(b) general unitaries U = Rz Ry Rz -- gridsynth must Euler-decompose into three
    z-rotations, while cyclosynth synthesizes the unitary directly.

Three methods per panel: gridsynth (Clifford+T), cyclosynth Clifford+T,
cyclosynth Clifford+sqrt(T). Each is a mean line with a +/-1 std band, plus
dashed C*log2(1/eps) reference slopes. Plot-only: reads the long-format gather
CSVs (synthesizer field), computes nothing.

Data:
  (a) scripts/data/rz.csv  (gather_rz.py)
  (b) scripts/data/u3.csv  (gather_u3.py)
Cost is the `cost` column (n_T for the Clifford+T methods; n_T + 3 n_sqrt(T),
floored, for Clifford+sqrt(T)).
Output: scripts/data/cost_vs_eps_3way.{pdf,svg}
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

U3_CSV = "scripts/data/u3.csv"
RZ_CSV = "scripts/data/rz.csv"
OUT_PNG = "scripts/data/cost_vs_eps_3way.pdf"  # vector; .svg also written
EPSS = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]

# Shared per-algorithm palette (see _plotstyle): cyclosynth T = blue,
# cyclosynth sqrt(T) = reddish-purple, gridsynth = orange.
STYLES = {
    "grid_T":    ("gridsynth (Clifford+$T$)",         _plotstyle.GRIDSYNTH,      "o"),
    "cyc_T":     (r"cyclosynth (Clifford+$T$)",        _plotstyle.CLIFFORD_T,     "s"),
    "cyc_sqrtT": (r"cyclosynth (Clifford+$\sqrt{T}$)", _plotstyle.CLIFFORD_SQRT_T, "^"),
}
# long-format `synthesizer` value -> series key
SYNTH = {"gridsynth": "grid_T", "cyclosynth_t": "cyc_T",
         "cyclosynth_sqrt_t": "cyc_sqrtT"}


def load_series(path):
    """series[key][eps] = list of per-target costs (T states), or None if the
    file is missing."""
    if not os.path.exists(path):
        return None
    series = {k: collections.defaultdict(list) for k in STYLES}
    for r in csv.DictReader(open(path)):
        if r["success"] != "True":
            continue
        key = SYNTH.get(r["synthesizer"])
        if key is None:
            continue
        series[key][float(r["epsilon"])].append(float(r["cost"]))
    return series


LOG2_10 = np.log2(10.0)  # convert C*log2(1/eps) to plot slope over x=-log10(eps)


def add_ref_lines(ax, specs, x0=3.0, x1=8.0):
    """Draw dashed C*log2(1/eps) reference lines with on-line, rotated labels.

    specs: list of (C, text, label_x, perp_pts). perp_pts offsets the label
    perpendicular to the line so the text sits in whitespace. Called after the
    final layout so the data->display angle is correct; limits frozen first.
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
        if not any(series[key].get(e) for e in epss):
            continue
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
    u3 = load_series(U3_CSV)
    rz = load_series(RZ_CSV)
    if u3 is None:
        raise SystemExit(f"missing {U3_CSV} -- run gather_u3.py first")

    if rz is None:
        print(f"WARNING: {RZ_CSV} missing -- run gather_rz.py; "
              "drawing U3 panel only.")
        fig, ax2 = plt.subplots(figsize=(6.4, 4.6))
        handles = draw(ax2, u3, "general unitaries $U$")
        ax2.set_ylabel(r"Cost ($T$ states)")
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.4, 3.6), sharey=False)
        handles = draw(ax1, rz, r"(a)  $z$-rotations $R_z(\theta)$")
        draw(ax2, u3, r"(b)  general unitaries $U$")
        ax1.set_ylabel(r"Cost ($T$ states)")

    fig.legend(handles, [h.get_label() for h in handles],
               loc="upper center", ncol=3, fontsize=10.5,
               framealpha=0.95, handlelength=1.6, columnspacing=1.4,
               bbox_to_anchor=(0.5, 1.005))
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    # Dashed C*log2(1/eps) reference slopes (after layout so the rotation angle
    # matches the final aspect; not in the legend).
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
            g = np.mean(s["grid_T"][e]) if s["grid_T"].get(e) else float("nan")
            print(f"    {e:.0e}: grid {g:.1f}  "
                  f"cycT {np.mean(s['cyc_T'][e]):.1f}  "
                  f"cycQ {np.mean(s['cyc_sqrtT'][e]):.1f}")


if __name__ == "__main__":
    main()
