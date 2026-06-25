"""Runtime comparison: Clifford+T vs Clifford+√T synthesis (violin plot).

By default it renders a grouped-violin plot of synthesis wall-clock (one
violin pair per ε, log-scaled y-axis since runtimes span orders of
magnitude) from an existing cost-comparison CSV — which already records
per-target `duration_ms` for both backends.

With --generate it instead synthesizes a fresh seeded dataset and plots
that, writing to a *separate* CSV so the shared file is never overwritten.
In that mode both backends do the same job — find a circuit within ε — so
the √T side uses first-hit (`optimize_cost=False`); flip
SQRT_T_OPTIMIZE_COST to time the √T cost-optimizing default instead.

Usage:
    python scripts/runtime_comparison_sqrtt.py             # plot existing CSV
    python scripts/runtime_comparison_sqrtt.py --generate  # fresh data + plot
"""

import csv
import glob
import os
import sys
from collections import defaultdict
from time import perf_counter

import numpy as np

# ─── Config ──────────────────────────────────────────────────────────────────

EPSILONS = [1e-3, 1e-4, 1e-5, 1e-6]
N_TARGETS = 20
SEED = 0xC0FFEE
SQRT_T_OPTIMIZE_COST = False  # first-hit = same job as Clifford+T (fair runtime)

# Default input: the existing cost-comparison dataset, which already records
# per-target duration_ms for both backends. --generate writes a fresh runtime
# dataset to GEN_CSV instead, so the shared file is never overwritten.
CSV_PATH = "scripts/data/comparison_sqrtt_data.csv"
GEN_CSV = "scripts/data/runtime_comparison_sqrtt.csv"
OUT_PATH = "scripts/data/comparison_sqrtt_runtime_violin.png"


# ─── Target generator (fixed, seeded) ────────────────────────────────────────

def rz(t):
    return np.array([[np.exp(-1j * t / 2), 0],
                     [0, np.exp(1j * t / 2)]], dtype=np.complex128)


def ry(t):
    c, s = np.cos(t / 2), np.sin(t / 2)
    return np.array([[c, -s], [s, c]], dtype=np.complex128)


def make_targets(n, seed):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        # Haar-random SU(2): a/g uniform, b sine-weighted on [0, pi]
        a, g = rng.uniform(0.0, 2 * np.pi, size=2)
        b = np.arccos(1.0 - 2.0 * rng.uniform(0.0, 1.0))
        out.append(((float(a), float(b), float(g)), rz(a) @ ry(b) @ rz(g)))
    return out


# ─── Data generation ─────────────────────────────────────────────────────────

def time_one(synth, target):
    """Return (duration_ms, distance, success)."""
    t0 = perf_counter()
    try:
        r = synth.synthesize(target)
        dur = (perf_counter() - t0) * 1000.0
        if r is None:
            return dur, float("inf"), False
        return dur, float(r.distance), True
    except Exception:
        return (perf_counter() - t0) * 1000.0, float("inf"), False


def generate():
    import cyclosynth

    os.makedirs(os.path.dirname(GEN_CSV), exist_ok=True)
    targets = make_targets(N_TARGETS, SEED)

    # Warm up each backend once (discarded): the first call pays one-time
    # rayon-pool init + code-path warmup, which would otherwise show up as
    # a single fat outlier in the violins.
    _, warm = make_targets(1, SEED + 1)[0]
    cyclosynth.Synthesizer(epsilon=1e-3).synthesize(warm)
    cyclosynth.Synthesizer(epsilon=1e-3, sqrt_t=True,
                           optimize_cost=SQRT_T_OPTIMIZE_COST).synthesize(warm)

    with open(GEN_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epsilon", "method", "trial", "alpha", "beta", "gamma",
                    "duration_ms", "distance", "success"])
        for eps in EPSILONS:
            print(f"\n=== ε = {eps:.0e} ===", flush=True)
            synth_t = cyclosynth.Synthesizer(epsilon=eps)
            synth_q = cyclosynth.Synthesizer(
                epsilon=eps, sqrt_t=True, optimize_cost=SQRT_T_OPTIMIZE_COST)
            for i, (angles, target) in enumerate(targets):
                for method, synth in (("clifford_t", synth_t),
                                      ("clifford_sqrt_t", synth_q)):
                    dur, dist, ok = time_one(synth, target)
                    w.writerow([f"{eps:.0e}", method, i, *angles,
                                f"{dur:.3f}", f"{dist:.6e}", ok])
                    f.flush()
                    print(f"  {i:>2} {method:<16} {dur:>9.1f} ms "
                          f"{'ok' if ok else 'FAIL'}", flush=True)
    print(f"\nwrote {GEN_CSV}", flush=True)


# ─── Plot ────────────────────────────────────────────────────────────────────

def _setup_style():
    import matplotlib as mpl
    import matplotlib.font_manager as fm
    import matplotlib.pyplot as plt

    plt.style.use("dark_background")
    if os.environ.get("CYCLOSYNTH_USETEX") == "1":
        mpl.rcParams["text.usetex"] = True
        mpl.rcParams["text.latex.preamble"] = (
            r"\usepackage[T1]{fontenc}\usepackage{lmodern}")
    else:
        for _f in glob.glob(
            "/usr/local/texlive/*/texmf-dist/fonts/opentype/public/lm/lmroman10-*.otf"
        ):
            fm.fontManager.addfont(_f)
        mpl.rcParams["font.family"] = "serif"
        mpl.rcParams["font.serif"] = ["Latin Modern Roman", "DejaVu Serif"]
        mpl.rcParams["mathtext.fontset"] = "cm"
    return plt


def load(path):
    data = defaultdict(lambda: defaultdict(
        lambda: {"ms": [], "fail": 0, "total": 0}))
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            eps = float(row["epsilon"])
            d = data[eps][row["method"]]
            d["total"] += 1
            if row["success"] == "True":
                d["ms"].append(float(row["duration_ms"]))
            else:
                d["fail"] += 1
    return data


def grouped_violin(ax, data, epsilons, methods, colors, slot_width=0.8):
    n = len(methods)
    width = slot_width / n
    for i, method in enumerate(methods):
        positions, datasets = [], []
        for j, eps in enumerate(epsilons):
            vals = data[eps][method]["ms"]
            if vals:
                positions.append(j + (i - (n - 1) / 2) * width)
                datasets.append(vals)
        if not datasets:
            continue
        parts = ax.violinplot(datasets, positions=positions,
                              widths=width * 0.9, showmeans=True,
                              showextrema=True)
        for body in parts["bodies"]:
            body.set_facecolor(colors[i])
            body.set_edgecolor("white")
            body.set_alpha(0.6)
        for lk in ("cmeans", "cmaxes", "cmins", "cbars"):
            if lk in parts:
                parts[lk].set_edgecolor("white")
                parts[lk].set_linewidth(0.8)


def plot(csv_path):
    plt = _setup_style()
    data = load(csv_path)
    epsilons = sorted(data.keys(), reverse=True)  # 1e-3 first
    methods = ["clifford_t", "clifford_sqrt_t"]
    labels = {"clifford_t": "Clifford+T",
              "clifford_sqrt_t": "Clifford+√T"}
    colors = ["#4c9be8", "#e8794c"]

    fig, ax = plt.subplots(figsize=(8, 5))
    # Violins on a log axis: plot log10(ms) and relabel ticks, so the
    # KDE is computed in the space the data actually spreads over.
    logdata = defaultdict(lambda: defaultdict(lambda: {"ms": []}))
    for eps in epsilons:
        for m in methods:
            logdata[eps][m]["ms"] = [np.log10(max(v, 1e-3))
                                     for v in data[eps][m]["ms"]]
    grouped_violin(ax, logdata, epsilons, methods, colors)

    ax.set_xticks(range(len(epsilons)))
    ax.set_xticklabels([f"$10^{{{int(np.log10(e))}}}$" for e in epsilons])
    ax.set_xlabel("Synthesis $\\varepsilon$")
    ax.set_ylabel("Wall-Clock Time (s)")
    ymin, ymax = ax.get_ylim()
    ticks = range(int(np.floor(ymin)), int(np.ceil(ymax)) + 1)
    ax.set_yticks(list(ticks))
    ax.set_yticklabels([f"$10^{{{t-3}}}$" for t in ticks])

    # npts = max((data[e][m]["total"] for e in epsilons for m in methods),
    #            default=0)
    # ax.set_title("Synthesis runtime: Clifford+T vs Clifford+$\\sqrt{T}$  "
    #              f"(N={npts} per $\\varepsilon$, {os.path.basename(csv_path)})")
    handles = [plt.Line2D([0], [0], color=colors[i], lw=8, alpha=0.6)
               for i in range(len(methods))]
    ax.legend(handles, [labels[m] for m in methods], loc="upper left",
              frameon=False)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150)
    print(f"wrote {OUT_PATH}")


def main():
    if "--generate" in sys.argv:
        generate()
        plot(GEN_CSV)
    else:
        plot(CSV_PATH)


if __name__ == "__main__":
    main()
