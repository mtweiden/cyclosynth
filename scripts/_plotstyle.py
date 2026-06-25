"""Shared matplotlib style for paper figures: Latin Modern serif + cm mathtext,
matching figure 1 (the cost-slope plot). White background.

Usage:
    import _plotstyle; _plotstyle.apply()
"""
import glob
import os
import matplotlib as mpl
import matplotlib.font_manager as fm

# Canonical per-algorithm colors, shared across all paper figures
# (colorblind-friendly Wong palette). Use these everywhere so each
# synthesis method has one consistent color.
CLIFFORD_T = "#0072B2"       # cyclosynth Clifford+T   (blue)
CLIFFORD_SQRT_T = "#CC79A7"  # cyclosynth Clifford+sqrt(T)  (reddish-purple)
GRIDSYNTH = "#E69F00"        # gridsynth Clifford+T    (orange)


def apply():
    if os.environ.get("CYCLOSYNTH_USETEX") == "1":
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
    # Font sizes shared across all paper figures (kept consistent here).
    mpl.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 13.5,
        "xtick.labelsize": 11.5,
        "ytick.labelsize": 11.5,
        "legend.fontsize": 11,
        "legend.title_fontsize": 11,
    })
