"""Paired-slope plot of cost = T + 3·Q, Clifford+√T vs Clifford+T.

Single-axis layout (mirrors the grouped-violin style of
`plot_comparison_sqrtt.py`): x-axis is ε with two sub-positions per
bucket (Clifford+T on the left, Clifford+√T on the right). Each trial
draws a single piecewise polyline through all 10 sub-positions —
within-bucket segments show the Clifford+T vs Clifford+√T comparison
at that ε, while cross-bucket segments show the monotone increase
in cost as ε tightens. Same (α, β, γ) is used for trial t across
all ε buckets (the binary re-seeds the RNG per ε), so every line is
a real per-target trajectory.

Subset relationship: every Clifford+T circuit is also a Clifford+√T
circuit (since T = Q²), so the *theoretical* Clifford+√T optimum is
bounded above by the Clifford+T cost. If the synthesiser produced a
Clifford+√T circuit with cost > Clifford+T cost on a given target,
we cap the Clifford+√T plotted value at the Clifford+T cost — that
is, we plot what an ideal synthesiser would do (fall back to the
Clifford+T decomposition). The count of capped trials per ε is
shown above each bucket.

Cost weights match the synthesiser's `gates_cost`: T = 1, Q = 3.
H, S, X, Y, Z are free.
"""

import csv
import glob
import os
from collections import defaultdict
import matplotlib
# matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

# White background (default style).

# Latin Modern. Default path: register lmroman10 for body text and use
# bundled Computer Modern for mathtext (visually equivalent to Latin
# Modern Math; mathtext can't consume OpenType MATH tables).
# Set CYCLOSYNTH_USETEX=1 to delegate rendering to LaTeX with `lmodern`
# for true Latin Modern Math glyphs — requires `latex` and `dvipng` on
# PATH (TeX Live 2026basic is missing dvipng; install via tlmgr).
USETEX = os.environ.get("CYCLOSYNTH_USETEX") == "1"
if USETEX:
    mpl.rcParams["text.usetex"] = True
    mpl.rcParams["text.latex.preamble"] = (
        r"\usepackage[T1]{fontenc}\usepackage{lmodern}"
    )
else:
    for _f in glob.glob(
        "/usr/local/texlive/*/texmf-dist/fonts/opentype/public/lm/lmroman10-*.otf"
    ):
        fm.fontManager.addfont(_f)
    mpl.rcParams["font.family"] = "serif"
    mpl.rcParams["font.serif"] = ["Latin Modern Roman", "DejaVu Serif"]
    mpl.rcParams["mathtext.fontset"] = "cm"


CSV_PATH = "scripts/data/comparison_sqrtt_data.csv"
OUT_PATH = "scripts/data/comparison_sqrtt_slope.pdf"  # vector; .svg also written

# Synthesizer's optimization weights — must match `gates_cost` in
# `src/synthesis/clifford_sqrt_t.rs`.
T_WEIGHT = 1
Q_WEIGHT = 3

# Half-distance between the Clifford+√T and Clifford+T sub-positions
# inside each ε bucket on the x-axis.
SUB_OFFSET = 0.15
# Stride between ε bucket centers. <1 packs the buckets closer.
X_STRIDE = 0.6
# Global font scale.
FS_TICK = 14
FS_AXIS = 16
FS_ANNOT = 14
FS_LEGEND = 13


def load(path):
    """Return data[eps][trial] = {"cost_T": float, "cost_Q": float}."""
    data = defaultdict(lambda: defaultdict(dict))
    with open(path) as f:
        for row in csv.DictReader(f):
            eps = float(row["epsilon"])
            trial = int(row["trial"])
            method = row["method"]
            t = int(row["t_count"])
            q = int(row["q_count"])
            cost = T_WEIGHT * t + Q_WEIGHT * q
            if method == "clifford_t":
                data[eps][trial]["cost_T"] = cost
            elif method == "clifford_sqrt_t":
                data[eps][trial]["cost_Q"] = cost
    return data


def main():
    data = load(CSV_PATH)
    # Sort ε descending so 1e-3 is leftmost (least precision).
    epsilons = sorted(data.keys(), reverse=True)
    if not epsilons:
        raise SystemExit(f"no data in {CSV_PATH}")

    fig, ax = plt.subplots(figsize=(1.6 * len(epsilons) + 2, 6))

    # Color for the Clifford+T (Q=0) endpoint scatter.
    color_T = plt.cm.tab10(0)
    color_Q = plt.cm.tab10(4)  # purple

    cap_annot_y = []   # save (x, text) for top-of-axis annotations

    # Pass 0: per-trial polyline across all ε buckets. Each polyline
    # goes  t@ε0 → q@ε0 → t@ε1 → q@ε1 → … visiting both sub-positions
    # in every ε (Clifford+T on the left, Clifford+√T on the right).
    # Drawn in light gray behind the per-bucket colored segments so
    # the within-bucket lines (drawn next) remain visible. This is what
    # gives the cross-bucket trajectory: each line is the *same target*
    # tightened across precisions.
    all_trials = sorted({t for eps in epsilons for t in data[eps]})
    for trial in all_trials:
        xs, ys = [], []
        for j, eps in enumerate(epsilons):
            r = data[eps].get(trial, {})
            if "cost_T" in r and "cost_Q" in r:
                q_cap = min(r["cost_Q"], r["cost_T"])
                xs.append(j * X_STRIDE - SUB_OFFSET); ys.append(r["cost_T"])
                xs.append(j * X_STRIDE + SUB_OFFSET); ys.append(q_cap)
        if len(xs) >= 2:
            ax.plot(xs, ys, "-", alpha=0.25, color="lightgray",
                    linewidth=0.6, zorder=1)

    for j, eps in enumerate(epsilons):
        x_t = j * X_STRIDE - SUB_OFFSET  # Clifford+T on the left
        x_q = j * X_STRIDE + SUB_OFFSET  # Clifford+√T on the right

        pairs = []
        for trial in sorted(data[eps].keys()):
            r = data[eps][trial]
            if "cost_T" in r and "cost_Q" in r:
                pairs.append((r["cost_Q"], r["cost_T"]))

        n = len(pairs)
        n_below_raw = sum(1 for q, t in pairs if q < t)
        n_equal_raw = sum(1 for q, t in pairs if q == t)
        n_capped = sum(1 for q, t in pairs if q > t)

        # Apply subset cap.
        plotted = [(min(q, t), t) for q, t in pairs]

        # Lines.
        for q_c, t_c in plotted:
            color = color_Q if q_c < t_c else "0.6"
            ax.plot([x_q, x_t], [q_c, t_c], "-",
                    alpha=0.35, color=color, linewidth=0.9)

        # Endpoint scatter to highlight discrete cost values.
        if plotted:
            qs = [p[0] for p in plotted]
            ts = [p[1] for p in plotted]
            ax.scatter([x_q] * n, qs, s=18, alpha=0.35, color=color_Q,
                       edgecolors="none", zorder=3)
            ax.scatter([x_t] * n, ts, s=18, alpha=0.35, color=color_T,
                       edgecolors="none", zorder=3)

            # Mean overlay.
            mean_q = np.mean(qs)
            mean_t = np.mean(ts)
            ax.plot([x_q, x_t], [mean_q, mean_t], "-",
                    color="black", linewidth=2.2, alpha=0.95, zorder=4)
            ax.scatter([x_q, x_t], [mean_q, mean_t], s=42, color="black",
                       zorder=5, edgecolors="white", linewidth=0.8)

        # Bucket label: win count + mean savings.
        # Savings = cost_T − cost_√T_capped per trial (= slope of each
        # within-bucket segment); we report the mean across trials in
        # both absolute and percentage terms.
        if plotted:
            saves = [t - q for (q, t) in plotted]
            mean_save = float(np.mean(saves))
            mean_t_cost = float(np.mean([t for (_, t) in plotted]))
            pct = 100.0 * mean_save / mean_t_cost if mean_t_cost else 0.0
            pct_sym = r"\%" if USETEX else "%"
            label = (
                f"{n_below_raw}/{n}\n"
                rf"$\bar\Delta = {mean_save:.1f}$"
                f"  ({pct:.0f}{pct_sym})"
            )
            bucket_top = max(t for (_, t) in plotted)
            bucket_bottom = min(q for (q, _) in plotted)
        else:
            label = f"{n_below_raw}/{n}"
            bucket_top = bucket_bottom = 0.0
        cap_annot_y.append((j * X_STRIDE, eps, bucket_top, bucket_bottom, label))

    # Axis cosmetics. LaTeX-formatted ε ticks render as 10^{-k}.
    ax.set_xticks([j * X_STRIDE for j in range(len(epsilons))])
    ax.set_xticklabels(
        [rf"$10^{{{int(round(np.log10(e)))}}}$" for e in epsilons],
        fontsize=FS_TICK,
    )
    ax.tick_params(axis="y", labelsize=FS_TICK)
    ax.set_xlabel(r"Precision  $\varepsilon$", fontsize=FS_AXIS)
    ax.set_ylabel(rf"Cost  $n_T + {Q_WEIGHT} $" + r"$\cdot n_{\sqrt{T}}$", fontsize=FS_AXIS)
    ax.grid(axis="y", alpha=0.3)
    ax.set_xlim(-X_STRIDE * 0.5, (len(epsilons) - 0.5) * X_STRIDE)

    # Per-bucket annotation block placed inside the axes. Above the
    # data for most buckets; below for the deepest ε so the label
    # doesn't get clipped by the axis top.
    for x, eps, ymax, ymin, text in cap_annot_y:
        below = eps <= 1e-7
        ax.annotate(
            text,
            xy=(x, ymin if below else ymax), xycoords="data",
            xytext=(0, -6 if below else 6), textcoords="offset points",
            ha="center", va="top" if below else "bottom",
            fontsize=FS_ANNOT, color="black",
        )

    # Legend ordered to match sub-position layout: T on the left,
    # √T on the right.
    legend_handles = [
        plt.Line2D([0], [0], marker="o", linestyle="",
                   color=color_T, alpha=0.5, markersize=8,
                   label="Clifford+T"),
        plt.Line2D([0], [0], marker="o", linestyle="",
                   color=color_Q, alpha=0.5, markersize=8,
                   label=r"Clifford+$\sqrt{T}$"),
        plt.Line2D([0], [0], color="black", linewidth=2.2,
                   label="mean across trials"),
    ]
    ax.legend(handles=legend_handles, loc="upper left",
              framealpha=0.9, fontsize=FS_LEGEND)

    plt.tight_layout()
    plt.savefig(OUT_PATH, bbox_inches="tight")
    plt.savefig(OUT_PATH.replace(".pdf", ".svg"), bbox_inches="tight")
    print(f"saved {OUT_PATH} (+ .svg)")


if __name__ == "__main__":
    main()
