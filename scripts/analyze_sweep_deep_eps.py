"""Build the cost-ratio-vs-runtime Pareto frontier from sweep_deep_eps.csv.

For each (ε, config) aggregates over the target set:
  * ratio   = cost / Clifford+T ref cost  → median + mean (lower = cheaper)
  * wall_ms → median + p90               (lower = faster)
  * success rate (status == ok and distance < ε)

Then marks the Pareto-optimal configs (no other config is both cheaper in
median ratio AND faster in median wall) and prints a recommendation per ε:
the cheapest config whose runtime is "acceptable", with the frontier shown
so the speed/cost trade-off is explicit.
"""

import csv
import sys
from collections import defaultdict
from statistics import median, mean

CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else "scripts/data/sweep_deep_eps.csv"


def pctl(xs, p):
    if not xs:
        return float("nan")
    s = sorted(xs)
    k = max(0, min(len(s) - 1, int(round(p * (len(s) - 1)))))
    return s[k]


def main():
    # rows[(eps, config)] = list of dicts
    rows = defaultdict(list)
    eps_order = []
    with open(CSV_PATH, newline="") as f:
        for r in csv.DictReader(f):
            eps = r["epsilon"]
            if eps not in eps_order:
                eps_order.append(eps)
            rows[(eps, r["config"])].append(r)

    for eps in eps_order:
        print(f"\n{'='*78}\nε = {eps}\n{'='*78}")
        configs = sorted({c for (e, c) in rows if e == eps})
        agg = {}
        for cfg in configs:
            data = rows[(eps, cfg)]
            eps_f = float(eps)
            ok = [d for d in data if d["status"] == "ok"
                  and float(d["distance"]) < eps_f and d["ratio"] not in ("nan", "")]
            ratios = [float(d["ratio"]) for d in ok]
            walls = [float(d["wall_ms"]) for d in data]  # incl. timeouts
            agg[cfg] = {
                "n": len(data),
                "n_ok": len(ok),
                "succ": len(ok) / len(data) if data else 0.0,
                "ratio_med": median(ratios) if ratios else float("nan"),
                "ratio_mean": mean(ratios) if ratios else float("nan"),
                "wall_med": median(walls) if walls else float("nan"),
                "wall_p90": pctl(walls, 0.90),
            }

        # Pareto frontier over (ratio_med ↓, wall_med ↓), excluding the ref.
        cand = [c for c in configs if c != "ref"
                and agg[c]["ratio_med"] == agg[c]["ratio_med"]]  # not NaN

        def dominated(c):
            for o in cand:
                if o == c:
                    continue
                a, b = agg[o], agg[c]
                if (a["ratio_med"] <= b["ratio_med"] and a["wall_med"] <= b["wall_med"]
                        and (a["ratio_med"] < b["ratio_med"] or a["wall_med"] < b["wall_med"])):
                    return True
            return False

        frontier = {c for c in cand if not dominated(c)}

        print(f"{'config':<22}{'succ':>6}{'ratio_med':>11}{'ratio_mean':>11}"
              f"{'wall_med_ms':>13}{'wall_p90_ms':>13}  pareto")
        for cfg in sorted(configs, key=lambda c: (agg[c]["ratio_med"]
                                                  if agg[c]["ratio_med"] == agg[c]["ratio_med"]
                                                  else 9e9)):
            a = agg[cfg]
            star = "  ★" if cfg in frontier else ""
            print(f"{cfg:<22}{a['succ']*100:>5.0f}%{a['ratio_med']:>11.3f}"
                  f"{a['ratio_mean']:>11.3f}{a['wall_med']:>13.0f}"
                  f"{a['wall_p90']:>13.0f}{star}")

        # Recommendation: cheapest frontier config with full success and a
        # "reasonable" median wall. We report the frontier and call out the
        # min-cost point + the best cost-under-2s point.
        full = [c for c in frontier if agg[c]["succ"] >= 0.999]
        pool = full or list(frontier)
        if pool:
            min_cost = min(pool, key=lambda c: agg[c]["ratio_med"])
            fast = [c for c in pool if agg[c]["wall_med"] <= 2000]
            best_fast = min(fast, key=lambda c: agg[c]["ratio_med"]) if fast else None
            print(f"\n  min-cost (frontier): {min_cost}  "
                  f"ratio={agg[min_cost]['ratio_med']:.3f}  "
                  f"wall_med={agg[min_cost]['wall_med']:.0f}ms")
            if best_fast:
                print(f"  best cost ≤2s wall:  {best_fast}  "
                      f"ratio={agg[best_fast]['ratio_med']:.3f}  "
                      f"wall_med={agg[best_fast]['wall_med']:.0f}ms")


if __name__ == "__main__":
    main()
