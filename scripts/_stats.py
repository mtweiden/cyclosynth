"""Shared loading, summary statistic and slope fit for the paper's plots and
tables, so every figure and every number in the text is computed one way.

PAPER_STAT selects the per-precision summary of the relative cost rho and of
the crossover c* (median by default; PAPER_STAT=mean for means).
"""
import collections
import csv
import os

import numpy as np

STAT = os.environ.get("PAPER_STAT", "median")
assert STAT in ("median", "mean"), STAT


def center(xs):
    return float(np.median(xs) if STAT == "median" else np.mean(xs))


def load(path):
    """{eps(float): {trial(int): {synthesizer: row}}} from a long-format CSV."""
    out = collections.defaultdict(lambda: collections.defaultdict(dict))
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            out[float(r["epsilon"])][int(r["trial"])][r["synthesizer"]] = r
    return out


def ok(r):
    return r is not None and r["success"] == "True"


def costs(data, eps, synth):
    return [float(d[synth]["cost"]) for d in data[eps].values() if ok(d.get(synth))]


def fit_log2(epss, means):
    """Least-squares fit mean_cost = C * log2(1/eps) + b. Returns (C, b)."""
    x = np.log2(1.0 / np.asarray(epss, dtype=float))
    C, b = np.polyfit(x, np.asarray(means, dtype=float), 1)
    return float(C), float(b)
