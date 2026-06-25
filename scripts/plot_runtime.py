"""Synthesis wall-clock comparison, Clifford+T vs Clifford+sqrt(T) (auxiliary).

Grouped violins of per-target synthesis time (one pair per precision epsilon,
log-scaled y since runtimes span orders of magnitude). Plot-only: reads the
`duration_ms` column of the long-format U3 gather (scripts/data/u3.csv),
filtering to the two cyclosynth synthesizers. Not used in the paper -- a quick
look at the cost-optimizing sqrt(T) overhead.

Output: scripts/data/runtime_violin.{pdf,svg}
"""
import csv
import os
import sys
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _plotstyle  # noqa: E402
_plotstyle.apply()

CSV_PATH = "scripts/data/u3.csv"
OUT_PATH = "scripts/data/runtime_violin.pdf"  # vector; .svg also written

METHODS = ["cyclosynth_t", "cyclosynth_sqrt_t"]
LABELS = {"cyclosynth_t": "Clifford+$T$",
          "cyclosynth_sqrt_t": r"Clifford+$\sqrt{T}$"}
COLORS = {"cyclosynth_t": _plotstyle.CLIFFORD_T,
          "cyclosynth_sqrt_t": _plotstyle.CLIFFORD_SQRT_T}


def load(path):
    """data[eps][synthesizer] = list of successful-run durations (ms)."""
    data = defaultdict(lambda: defaultdict(list))
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row["success"] != "True" or row["synthesizer"] not in METHODS:
                continue
            data[float(row["epsilon"])][row["synthesizer"]].append(
                float(row["duration_ms"]))
    return data


def grouped_violin(ax, data, epsilons, slot_width=0.8):
    n = len(METHODS)
    width = slot_width / n
    for i, method in enumerate(METHODS):
        positions, datasets = [], []
        for j, eps in enumerate(epsilons):
            vals = [np.log10(max(v, 1e-3)) for v in data[eps][method]]
            if vals:
                positions.append(j + (i - (n - 1) / 2) * width)
                datasets.append(vals)
        if not datasets:
            continue
        parts = ax.violinplot(datasets, positions=positions,
                              widths=width * 0.9, showmeans=True,
                              showextrema=True)
        for body in parts["bodies"]:
            body.set_facecolor(COLORS[method])
            body.set_edgecolor(COLORS[method])
            body.set_alpha(0.6)
        for lk in ("cmeans", "cmaxes", "cmins", "cbars"):
            if lk in parts:
                parts[lk].set_edgecolor(COLORS[method])
                parts[lk].set_linewidth(0.9)


def main():
    data = load(CSV_PATH)
    if not data:
        raise SystemExit(f"no data in {CSV_PATH} -- run gather_u3.py first")
    epsilons = sorted(data.keys(), reverse=True)   # 1e-3 leftmost

    fig, ax = plt.subplots(figsize=(8, 4.6))
    grouped_violin(ax, data, epsilons)

    ax.set_xticks(range(len(epsilons)))
    ax.set_xticklabels([rf"$10^{{{int(round(np.log10(e)))}}}$" for e in epsilons])
    ax.set_xlabel(r"Precision  $\varepsilon$")
    ax.set_ylabel("Wall-clock time")
    ymin, ymax = ax.get_ylim()
    ticks = list(range(int(np.floor(ymin)), int(np.ceil(ymax)) + 1))
    ax.set_yticks(ticks)
    # data is log10(ms); label as seconds (ms -> s is -3 in log10).
    ax.set_yticklabels([rf"$10^{{{t - 3}}}$ s" for t in ticks])

    handles = [plt.Line2D([0], [0], marker="s", ls="", color=COLORS[m],
                          alpha=0.7, ms=11, label=LABELS[m]) for m in METHODS]
    leg = ax.legend(handles=handles, loc="upper left", framealpha=0.95,
                    fontsize=11, handlelength=1.4)
    leg.get_frame().set_edgecolor("0.7")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_PATH, bbox_inches="tight")
    fig.savefig(OUT_PATH.replace(".pdf", ".svg"), bbox_inches="tight")
    print(f"saved {OUT_PATH} (+ .svg)")
    for eps in epsilons:
        msg = "  ".join(
            f"{m.replace('cyclosynth_', '')}: median {np.median(data[eps][m]):.1f} ms"
            for m in METHODS if data[eps][m])
        print(f"  {eps:.0e}: {msg}")


if __name__ == "__main__":
    main()
