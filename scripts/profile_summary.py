#!/usr/bin/env python3
"""Parse a probe_omega_vs_zeta log into one campaign-scoreboard CSV row.

Aggregates the per-target `[profile]` lines (emitted when
CYCLOSYNTH_TRACE=1) plus the probe's aggregate block, so a step's
verdict can cite per-subsystem deltas, not just wall/cost.

Usage:
  python3 scripts/profile_summary.py --step S1 --config "bkz_on_d2500" \
      --log bench_logs/campaign_2026_06/s1_bkz_on.log \
      [--scoreboard bench_logs/campaign_2026_06/scoreboard.csv]

Appends one row per invocation; creates the CSV (with header) if absent.
Multiple runs in one log file (replicates) are aggregated together —
pass per-replicate logs separately if per-replicate rows are wanted.
"""
import argparse
import csv
import os
import re
import sys
from collections import defaultdict

PROFILE_RE = re.compile(r"^\[profile\] i=(\d+) synth=(\w) wall_s=([\d.]+) (.*)$")
KV_RE = re.compile(r"(\w+)=([-\d.]+)")

# Scoreboard columns derived from summed Q-side [profile] lines.
Q_KEYS = [
    "screen_ms", "frontier_ms", "baseline_ms",
    "build_ms", "lll_ms", "chol_ms", "lu_ms", "se_ms",
    "lll_iters", "lll_at_cap", "f64_escal",
    "se_nodes", "se_cb", "search_calls", "prefixes",
    "prune_fires", "verify_fires", "verify_corrected",
    "pred_trunc", "budget_exhaust", "sols",
]


def parse_log(path):
    sums = {"T": defaultdict(float), "Q": defaultdict(float)}
    counts = {"T": 0, "Q": 0}
    agg = {}
    with open(path, errors="replace") as f:
        for line in f:
            m = PROFILE_RE.match(line.strip())
            if m:
                synth = m.group(2)
                if synth in sums:
                    counts[synth] += 1
                    sums[synth]["wall_s"] += float(m.group(3))
                    for k, v in KV_RE.findall(m.group(4)):
                        sums[synth][k] += float(v)
                continue
            m = re.search(r"Clifford\+T : total cost = +([\d.]+), total wall = +([\d.]+)s", line)
            if m:
                agg["t_cost"] = agg.get("t_cost", 0.0) + float(m.group(1))
                agg["t_wall"] = agg.get("t_wall", 0.0) + float(m.group(2))
                agg["n_runs"] = agg.get("n_runs", 0) + 1
                continue
            m = re.search(r"Clifford\+√T: total cost = +([\d.]+), total wall = +([\d.]+)s", line)
            if m:
                agg["q_cost"] = agg.get("q_cost", 0.0) + float(m.group(1))
                agg["q_wall"] = agg.get("q_wall", 0.0) + float(m.group(2))
                continue
    return sums, counts, agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--scoreboard",
                    default="bench_logs/campaign_2026_06/scoreboard.csv")
    args = ap.parse_args()

    sums, counts, agg = parse_log(args.log)
    n_runs = agg.get("n_runs", 0)
    if n_runs == 0:
        sys.exit(f"no aggregate block found in {args.log}")

    row = {
        "step": args.step,
        "config": args.config,
        "log": args.log,
        "n_runs": n_runs,
        "q_cost_mean": round(agg.get("q_cost", 0.0) / n_runs, 1),
        "t_cost_mean": round(agg.get("t_cost", 0.0) / n_runs, 1),
        "ratio": round(agg.get("q_cost", 0.0) / agg["t_cost"], 4)
                 if agg.get("t_cost") else "",
        "q_wall_mean_s": round(agg.get("q_wall", 0.0) / n_runs, 2),
        "n_profile_lines": counts["Q"],
    }
    q = sums["Q"]
    for k in Q_KEYS:
        row[f"q_{k}"] = round(q.get(k, 0.0), 1)
    # Subsystem shares of the trace-phase total, for at-a-glance attribution.
    phase_total = sum(q.get(k, 0.0) for k in
                      ("build_ms", "lll_ms", "chol_ms", "lu_ms", "se_ms"))
    row["phase_total_ms"] = round(phase_total, 1)
    for k in ("lll_ms", "se_ms"):
        row[f"{k}_share"] = round(q.get(k, 0.0) / phase_total, 3) if phase_total else ""

    os.makedirs(os.path.dirname(args.scoreboard), exist_ok=True)
    new = not os.path.exists(args.scoreboard)
    with open(args.scoreboard, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)

    print(f"[scoreboard] {args.step}/{args.config}: "
          f"cost={row['q_cost_mean']} ratio={row['ratio']} "
          f"wall={row['q_wall_mean_s']}s "
          f"lll_share={row['lll_ms_share']} se_share={row['se_ms_share']} "
          f"-> {args.scoreboard}")


if __name__ == "__main__":
    main()
