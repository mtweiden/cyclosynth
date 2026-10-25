#!/usr/bin/env bash
# Regenerate every number in the paper from scratch.
#
#   scripts/reproduce_paper.sh                 # build, gather, print numbers
#   SKIP_GATHER=1 scripts/reproduce_paper.sh   # recompute from existing CSVs
#
# Requirements: Rust toolchain, maturin, Python 3 with numpy and mpmath, and
# the `gridsynth` binary (Haskell newsynth: `cabal install newsynth`, or set
# GRIDSYNTH_BIN). Run from the repository root. The gathers take many hours at
# 500 targets per precision; they are resumable, so an interrupted run picks up
# where it stopped. See scripts/README.md.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p scripts/data

maturin develop --release

if [[ -z "${SKIP_GATHER:-}" ]]; then
  python3 scripts/gather_rz.py 2>&1 | tee scripts/data/gather_rz.log
  python3 scripts/gather_u3.py 2>&1 | tee scripts/data/gather_u3.log
fi

python3 scripts/paper_numbers.py | tee scripts/data/paper_numbers.txt
