"""Cost-weight sensitivity: median cost ratio rho vs the sqrt(T) price c.

Under the block cost model a circuit's cost in T states is
n_Tclass + c * n_sqrtTclass (T-class blocks cost 1, sqrt(T)-class blocks cost
the sqrt(T) price c; see scripts/_cost.py). Per paired target,
rho(c) = (n_Tclass^Q + c*n_R^Q)/(n_Tclass^T + c*n_R^T), with n_R^T = 0 for the
Clifford+T circuit so its denominator is just its T-class count. We plot the
median over 500 paired Haar targets per precision, UNFLOORED (no min with T),
so the crossover c* where median rho=1 is the honest sqrt(T) price above which
the raw sqrt(T) circuit stops being cheaper.

Output: scripts/data/cost_weight_sweep.pdf
"""
import csv
import collections
import os
import statistics
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _plotstyle  # noqa: E402
import _cost       # noqa: E402  block-model cost (sqrt(T)-class blocks, T^{3/2}=R)
_plotstyle.apply()
plt.rcParams.update({"axes.labelsize": 16, "xtick.labelsize": 13, "ytick.labelsize": 13})

CSV = "scripts/data/u3.csv"
OUT = "scripts/data/cost_weight_sweep.pdf"  # vector; .svg also written
C_REF = 3.0

rows = list(csv.DictReader(open(CSV)))
byk = collections.defaultdict(dict)
for r in rows:
    if r["success"] != "True":
        continue
    byk[(r["epsilon"], r["alpha"], r["beta"], r["gamma"])][r["synthesizer"]] = r

epsilons = sorted({k[0] for k in byk}, key=lambda e: -float(e))


def med_rho(eps, c):
    rr = []
    for k, d in byk.items():
        if k[0] != eps or "cyclosynth_t" not in d or "cyclosynth_sqrt_t" not in d:
            continue
        ntT, nrT = _cost.block_classes(d["cyclosynth_t"]["gates"])
        ntQ, nrQ = _cost.block_classes(d["cyclosynth_sqrt_t"]["gates"])
        den = ntT + c * nrT
        if den > 0:
            rr.append((ntQ + c * nrQ) / den)
    return statistics.median(rr)


def cstar(eps):
    lo, hi = 1.0, 9.0
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if med_rho(eps, mid) < 1.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


cs = np.linspace(1.0, 6.0, 51)
cstars = [cstar(e) for e in epsilons]
# Single-column figure: a larger canvas keeps the plot un-squished (the
# fixed-size fonts then occupy a smaller fraction of the axes). On-page text
# is necessarily a bit smaller than the full-width figures placed at the same
# scale -- the usual single- vs double-column trade-off.
fig, ax = plt.subplots(figsize=(5.6, 4.3))

# Win/lose shading relative to the break-even line rho = 1.
ax.axhspan(0.0, 1.0, color="tab:green", alpha=0.07, zorder=0)
ax.axhspan(1.0, 2.0, color="tab:red", alpha=0.07, zorder=0)
# Crossover band: the range of c* across precisions.
ax.axvspan(min(cstars), max(cstars), color="0.5", alpha=0.12, zorder=0)
# Extrapolated asymptote of c* as eps -> 0 under the block cost model
# [ per-decade n_Tclass^T~10, n_Tclass^Q~2, n_R^Q~2 -> (10-2)/2 ~ 4.0; fit 4.15 ].
C_ASYMP = 4.0
ax.axvline(C_ASYMP, color="darkgreen", lw=1.5, ls=(0, (5, 2)), zorder=1)

cmap = plt.cm.plasma(np.linspace(0.05, 0.85, len(epsilons)))
ends = []  # (color, y at c=6, exponent) for direct line labels
for col, eps, cstar_e in zip(cmap, epsilons, cstars):
    ys = [med_rho(eps, c) for c in cs]
    e = int(round(np.log10(float(eps))))
    ax.plot(cs, ys, "-", color=col, lw=1.8)
    ax.plot([cstar_e], [1.0], "o", color=col, ms=6, mec="k", mew=0.6, zorder=5)
    ends.append((col, ys[-1], e))

ax.axhline(1.0, color="0.25", lw=1.3, ls="--")           # break-even
ax.axvline(C_REF, color="black", lw=1.4, ls=":")          # our weight
ax.set_ylim(0.5, 1.45)
ax.set_xlim(1, 6)
ax.set_xticks(range(1, 7))
# In-plot annotations placed right beside the rho=1 break-even line, on the
# left where the curves are far below 1 (so they never overlap the curves).
ax.text(1.75, 0.965, r"$\sqrt{T}$ cheaper", color="tab:green",
        fontsize=14, va="top", ha="center", ma="center", fontweight="bold")
ax.text(1.75, 1.035, r"$\sqrt{T}$ more" + "\nexpensive", color="tab:red",
        fontsize=14, va="bottom", ha="center", ma="center", fontweight="bold")
# Short vertical-line labels at the top (details in the caption).
ax.text(C_REF - 0.06, 1.43, r"$c=3$", color="black",
        fontsize=14, rotation=90, va="top", ha="right")
ax.text(C_ASYMP + 0.06, 1.43, r"$c^\star\!\to\!4.0$",
        color="darkgreen", fontsize=14, rotation=90, va="top", ha="left")
ax.text(np.mean(cstars), 0.53, "measured\ncrossover\nrange", color="0.3",
        fontsize=14, ha="center", va="bottom", ma="center")
ax.set_xlabel(r"$c$ in $\mathrm{cost}(U_{\sqrt{T}}) = n_T + c\,n_{\sqrt{T}}$")
ax.set_ylabel(r"$\rho = \mathrm{cost}(U_{\sqrt{T}})/\mathrm{cost}(U_T)$")
def dark_text(col, cap=0.45):
    """Scale a color toward black so its luminance <= cap (readable on white)."""
    r, g, b = mcolors.to_rgb(col)
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    s = min(1.0, cap / max(lum, 1e-3))
    return (r * s, g * s, b * s)


# Direct labels at each line's right end (decluttered, with leaders) -- no legend.
order = sorted(range(len(ends)), key=lambda i: ends[i][1])
ylab = [ends[i][1] for i in order]
GAP = 0.060
for j in range(1, len(ylab)):
    if ylab[j] - ylab[j - 1] < GAP:
        ylab[j] = ylab[j - 1] + GAP
ylab = [y + (np.mean([ends[i][1] for i in order]) - np.mean(ylab)) for y in ylab]
for rank, i in enumerate(order):
    col, y_end, e = ends[i]
    ax.annotate(rf"$\varepsilon\,{{=}}\,10^{{{e}}}$", xy=(6.0, y_end),
                xytext=(6.25, ylab[rank]), color=dark_text(col), fontsize=12,
                va="center", ha="left", annotation_clip=False, fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=col, lw=0.7, shrinkA=0, shrinkB=2))
ax.grid(alpha=0.2)
fig.tight_layout()
fig.savefig(OUT, bbox_inches="tight")
fig.savefig(OUT.replace(".pdf", ".svg"), bbox_inches="tight")
print(f"saved {OUT} (+ .svg)")
print("crossover c* per eps:")
for eps in epsilons:
    print(f"  {eps}: c*={cstar(eps):.2f}  rho(3)={med_rho(eps,3):.3f}")
