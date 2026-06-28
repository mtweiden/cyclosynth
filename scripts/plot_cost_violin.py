"""Grouped-violin cost comparison, Clifford+T vs Clifford+sqrt(T) (Fig 2).

For each precision epsilon, two full violins show the distribution of per-target
cost = T_count + 3*sqrt(T)_count over the Haar-random U3 targets at that
epsilon: Clifford+T (blue, left) and Clifford+sqrt(T) (reddish-purple, right).
The sqrt(T) cost is floored at the Clifford+T cost, so every sqrt(T) value is
<= its paired T value.

Reads the long-format gather (scripts/data/u3.csv), filtering to the two
cyclosynth synthesizers. Output: scripts/data/cost_violin.{pdf,svg} (vector).
"""
import csv
import os
import sys
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.transforms import blended_transform_factory

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _plotstyle  # noqa: E402
import _cost       # noqa: E402  syllable-model cost (sqrt(T)-class syllables, T^{3/2}=R)
_plotstyle.apply()

CSV_PATH = "scripts/data/u3.csv"
OUT_PATH = "scripts/data/cost_violin.pdf"  # vector; .svg also written

Q_WEIGHT = 3  # sqrt(T) price in T states (syllable-model sqrt(T)-class cost)
COLOR_T = _plotstyle.CLIFFORD_T        # blue (shared)
COLOR_Q = _plotstyle.CLIFFORD_SQRT_T   # reddish-purple (shared)
MEAN_COLOR = "0.95"                    # light grey / almost white
MEAN_FX = [pe.withStroke(linewidth=3.4, foreground="0.4")]  # faint halo
OFFSET = 0.23                          # half-gap between the paired violins
WIDTH = 0.40                           # full-violin width


def load(path):
    """data[eps] = list of (cost_T, cost_Q_floored) over paired trials."""
    by = defaultdict(lambda: defaultdict(dict))
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row["success"] != "True":
                continue
            cost = _cost.syllable_cost(row["gates"], Q_WEIGHT)
            by[float(row["epsilon"])][int(row["trial"])][row["synthesizer"]] = cost
    out = {}
    for eps, trials in by.items():
        pairs = []
        for d in trials.values():
            if "cyclosynth_t" in d and "cyclosynth_sqrt_t" in d:
                t = d["cyclosynth_t"]
                q = min(d["cyclosynth_sqrt_t"], t)   # floor guarantee
                pairs.append((t, q))
        out[eps] = pairs
    return out


def violin(ax, data, center, color):
    """Draw one full (two-sided) violin with a light-grey mean marker."""
    parts = ax.violinplot([data], positions=[center], widths=WIDTH,
                          showextrema=False, showmeans=True)
    for b in parts["bodies"]:
        b.set_facecolor(color)
        b.set_edgecolor(color)
        b.set_alpha(0.6)
        b.set_linewidth(0.8)
    cm = parts["cmeans"]
    cm.set_color(MEAN_COLOR)
    cm.set_linewidth(2.2)
    cm.set_zorder(5)
    cm.set_path_effects(MEAN_FX)


def main():
    data = load(CSV_PATH)
    epsilons = sorted(data.keys(), reverse=True)   # 1e-3 leftmost

    fig, ax = plt.subplots(figsize=(9.4, 3.5))
    gmin = min(min(q for _, q in data[e]) for e in epsilons)
    gmax = max(max(t for t, _ in data[e]) for e in epsilons)

    for j, eps in enumerate(epsilons):
        pairs = data[eps]
        ts = np.array([p[0] for p in pairs])
        qs = np.array([p[1] for p in pairs])
        violin(ax, ts, j - OFFSET, COLOR_T)
        violin(ax, qs, j + OFFSET, COLOR_Q)

    # Annotations below the violins (median reduction + strictly-cheaper count),
    # in axes-fraction y so they never collide with the violins.
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    for j, eps in enumerate(epsilons):
        pairs = data[eps]
        ts = np.array([p[0] for p in pairs])
        qs = np.array([p[1] for p in pairs])
        rho = np.median(qs / ts)
        ax.text(j, 0.015, f"$-{(1-rho)*100:.0f}\\%$\n{int(np.sum(qs<ts))}/{len(pairs)}",
                transform=trans, ha="center", va="bottom", fontsize=10.5,
                color="0.2")

    ax.set_xticks(range(len(epsilons)))
    ax.set_xticklabels([rf"$10^{{{int(round(np.log10(e)))}}}$" for e in epsilons])
    ax.set_xlabel(r"Precision  $\varepsilon$")
    ax.set_ylabel(r"Cost ($T$ states)")
    ax.grid(axis="y", alpha=0.25)
    ax.set_xlim(-0.6, len(epsilons) - 0.4)
    ax.set_ylim(gmin - 0.16 * (gmax - gmin), gmax + 0.04 * (gmax - gmin))

    handles = [
        plt.Line2D([0], [0], marker="s", ls="", color=COLOR_T, alpha=0.7,
                   ms=11, label="Clifford+$T$"),
        plt.Line2D([0], [0], marker="s", ls="", color=COLOR_Q, alpha=0.7,
                   ms=11, label=r"Clifford+$\sqrt{T}$"),
        plt.Line2D([0], [0], color=MEAN_COLOR, lw=2.2,
                   path_effects=MEAN_FX, label="mean"),
    ]
    leg = ax.legend(handles=handles, loc="upper left", framealpha=0.95,
                    fontsize=11, handlelength=1.4)
    leg.get_frame().set_edgecolor("0.7")

    fig.tight_layout()
    fig.savefig(OUT_PATH, bbox_inches="tight")
    fig.savefig(OUT_PATH.replace(".pdf", ".svg"), bbox_inches="tight")
    print(f"saved {OUT_PATH} (+ .svg)")
    for eps in epsilons:
        pairs = data[eps]
        ts = np.array([p[0] for p in pairs]); qs = np.array([p[1] for p in pairs])
        print(f"  {eps:.0e}: median rho={np.median(qs/ts):.3f}  "
              f"mean_T={ts.mean():.1f} mean_Q={qs.mean():.1f}  "
              f"cheaper {int(np.sum(qs<ts))}/{len(pairs)}")


if __name__ == "__main__":
    main()
