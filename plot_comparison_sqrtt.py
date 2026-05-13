"""Plot Clifford+T vs Clifford+√T cost comparison from
`comparison_sqrtt_data.csv`. Mirrors `plot_comparison.py`'s grouped-violin
style; y-axis is the T-count-equivalent cost (T_count + 2.5·Q_count).
"""

import csv
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np


CSV_PATH = "comparison_sqrtt_data.csv"
OUT_PATH = "comparison_sqrtt_violin.png"


def load(path):
    data = defaultdict(lambda: defaultdict(
        lambda: {"cost": [], "t": [], "q": [], "s": [], "fail": 0, "total": 0}))
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            eps = float(row["epsilon"])
            method = row["method"]
            success = row["success"] == "True"
            d = data[eps][method]
            d["total"] += 1
            if success:
                d["cost"].append(float(row["cost"]))
                d["t"].append(int(row["t_count"]))
                d["q"].append(int(row["q_count"]))
                d["s"].append(float(row["duration_ms"]) / 1000)
            else:
                d["fail"] += 1
    return data


def grouped_violin(ax, data, epsilons, methods, key, colors, slot_width=0.8):
    n = len(methods)
    width = slot_width / n
    for i, method in enumerate(methods):
        positions, datasets = [], []
        for j, eps in enumerate(epsilons):
            vals = data[eps][method][key]
            if vals:
                positions.append(j + (i - (n - 1) / 2) * width)
                datasets.append(vals)
        if not datasets:
            continue
        parts = ax.violinplot(
            datasets, positions=positions, widths=width * 0.9,
            showmeans=True, showextrema=True,
        )
        for body in parts["bodies"]:
            body.set_facecolor(colors[i])
            body.set_edgecolor("black")
            body.set_alpha(0.6)
        for line_key in ("cmeans", "cmaxes", "cmins", "cbars"):
            if line_key in parts:
                parts[line_key].set_edgecolor("black")
                parts[line_key].set_linewidth(0.8)


def annotate_failures(ax, data, epsilons, methods, colors,
                      where="above", slot_width=0.8):
    n = len(methods)
    width = slot_width / n
    y_frac = 1.0 if where == "above" else 0.0
    y_offset = -4 if where == "above" else 4
    va = "top" if where == "above" else "bottom"
    for i, method in enumerate(methods):
        for j, eps in enumerate(epsilons):
            fails = data[eps][method]["fail"]
            total = data[eps][method]["total"]
            if fails == 0:
                continue
            pos = j + (i - (n - 1) / 2) * width
            ax.annotate(
                f"{fails}/{total}\nfail",
                xy=(pos, y_frac), xycoords=("data", "axes fraction"),
                xytext=(0, y_offset), textcoords="offset points",
                ha="center", va=va, fontsize=7, color=colors[i],
            )


def main():
    data = load(CSV_PATH)
    epsilons = sorted(data.keys(), reverse=True)  # 1e-3 first → 1e-6 last
    methods = sorted({m for eps in data for m in data[eps]})

    # Pretty labels.
    label_map = {
        "clifford_t": "Clifford+T",
        "clifford_sqrt_t": "Clifford+√T",
    }
    pretty = [label_map.get(m, m) for m in methods]
    # Clifford+T → tab:blue (tab10[0]); Clifford+√T → tab:purple (tab10[4],
    # the next categorical color after tab:orange).
    color_map = {
        "clifford_t": plt.cm.tab10(0),
        "clifford_sqrt_t": plt.cm.tab10(4),
    }
    colors = [color_map.get(m, plt.cm.tab10(i)) for i, m in enumerate(methods)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    grouped_violin(axes[0], data, epsilons, methods, "cost", colors)
    axes[0].set_ylabel("T-count-equivalent cost (T + 2.5·Q)")
    axes[0].set_title("Cost distribution")
    bound_xs = range(len(epsilons))
    bound_ys = [3 * np.log2(1 / eps) for eps in epsilons]
    axes[0].plot(bound_xs, bound_ys, linestyle=":", color="black",
                 linewidth=1.0, label=r"$3\log_2(1/\varepsilon)$")

    grouped_violin(axes[1], data, epsilons, methods, "s", colors)
    axes[1].set_ylabel("Time (s)")
    axes[1].set_yscale("log")
    axes[1].set_title("Compilation time")

    for ax in axes:
        ax.set_xticks(range(len(epsilons)))
        ax.set_xticklabels([f"{e:.0e}" for e in epsilons])
        ax.set_xlabel("epsilon")
        ax.grid(axis="y", alpha=0.3)

    annotate_failures(axes[0], data, epsilons, methods, colors, where="above")
    annotate_failures(axes[1], data, epsilons, methods, colors, where="below")

    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[i], alpha=0.6)
               for i in range(len(methods))]
    bound_handle = plt.Line2D([0], [0], linestyle=":", color="black",
                              linewidth=1.0)
    axes[0].legend(handles + [bound_handle],
                   pretty + [r"$3\log_2(1/\varepsilon)$"], loc="best")

    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=150)
    plt.show()
    print(f"saved {OUT_PATH}")


if __name__ == "__main__":
    main()
