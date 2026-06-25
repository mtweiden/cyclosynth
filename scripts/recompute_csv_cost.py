"""Recompute the `cost` column of `comparison_sqrtt_data.csv` (or a
similarly-formatted CSV) as `T + 3·Q` from the `t_count` and `q_count`
columns. Leaves all other columns untouched.

Use case: an older binary was emitting `cost = T + 2.5·Q` for backward
compatibility with `comparison_sqrtt.py`, but the actual synthesizer
optimisation target is `T + 3·Q`. The synthesised gate strings are
already T+3Q-optimal; we just need to re-display the cost field.
"""

import argparse
import csv
import shutil
import sys
import tempfile
from pathlib import Path


def recompute(in_path: Path, out_path: Path, q_weight: float = 3.0):
    with in_path.open(newline="") as src, out_path.open("w", newline="") as dst:
        reader = csv.DictReader(src)
        cols = reader.fieldnames
        if "cost" not in cols or "t_count" not in cols or "q_count" not in cols:
            sys.exit(f"missing required columns; have {cols}")
        writer = csv.DictWriter(dst, fieldnames=cols)
        writer.writeheader()
        n = 0
        for row in reader:
            try:
                t = int(row["t_count"])
                q = int(row["q_count"])
            except ValueError:
                writer.writerow(row)
                continue
            if t == 0 and q == 0:
                # Failed synthesis row; leave cost as-is.
                writer.writerow(row)
                continue
            new_cost = t + q_weight * q
            row["cost"] = f"{new_cost:.1f}"
            writer.writerow(row)
            n += 1
        print(f"recomputed cost on {n} rows (Q weight = {q_weight})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("csv", type=Path)
    p.add_argument("--in-place", action="store_true",
                   help="overwrite input file (default: write to <input>.t3q.csv)")
    p.add_argument("--q-weight", type=float, default=3.0)
    args = p.parse_args()

    in_path = args.csv
    if not in_path.exists():
        sys.exit(f"not found: {in_path}")
    if args.in_place:
        with tempfile.NamedTemporaryFile(
            "w", newline="", delete=False,
            dir=in_path.parent, suffix=".tmp",
        ) as tmp:
            tmp_path = Path(tmp.name)
        try:
            recompute(in_path, tmp_path, q_weight=args.q_weight)
            shutil.move(tmp_path, in_path)
            print(f"wrote {in_path} (in-place)")
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise
    else:
        out_path = in_path.with_suffix(".t3q.csv")
        recompute(in_path, out_path, q_weight=args.q_weight)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
