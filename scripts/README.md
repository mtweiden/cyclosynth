# scripts/

The experiment scripts behind the numbers reported in the paper. Everything
here is plain Python 3 driving the installed `cyclosynth` module; the library
itself is in `src/`.

Run everything from the **repository root**, not from this directory — paths
are repo-relative throughout.

## Quick start

```sh
scripts/reproduce_paper.sh                 # build, gather, print numbers
SKIP_GATHER=1 scripts/reproduce_paper.sh   # recompute from existing CSVs
```

The gathers synthesize 500 targets per precision per method and **take many
hours**. They are resumable: rows already present for an
`(epsilon, trial, synthesizer)` triple are kept and skipped, so an interrupted
run picks up where it stopped. `SKIP_GATHER=1` skips them entirely and
recomputes the reported numbers from the existing CSVs.

## Requirements

- Rust toolchain and [`maturin`](https://github.com/PyO3/maturin) — the script
  runs `maturin develop --release` to build and install the module.
- Python 3 with `numpy` and `mpmath`.
- The `gridsynth` binary (Haskell `newsynth`: `cabal install newsynth`) for the
  Ross–Selinger baseline. Located via `$GRIDSYNTH_BIN`, then `PATH`, then the
  usual cabal/ghcup install dirs. The gather skips the `gridsynth` rows if it
  is absent; the cyclosynth rows are unaffected.

## What runs what

| script | reads | writes |
| --- | --- | --- |
| `gather_u3.py` | — | `data/u3.csv` — general-unitary targets |
| `gather_rz.py` | — | `data/rz.csv` — z-rotation targets |
| `paper_numbers.py` | `data/{rz,u3}.csv` | stdout, `data/paper_numbers.txt` |

`paper_numbers.py` computes every number the paper quotes from the
experiments: the cost and runtime tables, the fitted cost scalings, the
Clifford-gate counts, the gridsynth comparison, the crossover $c^\star$, the
catalyst break-even, and the z-rotation $\sqrt{T}$ slope.

Shared modules, not run directly: `_gather.py` (gather plumbing), `_cost.py`
(syllable cost model, mirrors `gates_cost` in
`src/synthesis/clifford_sqrt_t/mod.rs`), `_stats.py` (summary statistic and
slope fit), `gridsynth_real.py` (wrapper around the `gridsynth` binary).

## Data files

`data/` holds the gather output. It is gitignored except for `rz.csv` and
`u3.csv`, the two CSVs the reported numbers are computed from; everything else
there is a local run artifact.

Both CSVs are long-format, one row per `(epsilon, trial, synthesizer)`:

```
epsilon,trial,alpha,beta,gamma,synthesizer,t_count,q_count,cost,distance,
verified,duration_ms,timed_out,success,gates
```

`rz.csv` carries `theta` in place of `alpha,beta,gamma`. `synthesizer` is one
of `cyclosynth_t`, `cyclosynth_sqrt_t`, `gridsynth`. `cost` is the syllable
model of `_cost.py` (T-class syllable 1, √T-class syllable 3). `gates` is the
gate string in matrix order — rightmost gate applied first.

Each CSV has a `<name>.csv.meta.json` sidecar recording host, CPU, thread
count, Python version, cyclosynth commit and the gather config. The sidecars
are local-only.

### Verification

Every circuit is re-verified **independently of the library**: the gate string
is multiplied out in `mpmath` at 40 digits and the diamond distance to the
target recomputed. That check is the `verified` column, and `success` requires
it. `paper_numbers.py` refuses to report from data containing failed or
unverified rows, and prints how many calls hit the per-call timeout.

Each cyclosynth call runs in its own spawned process and is timed there, so
`duration_ms` is one `synthesize()` call with the whole machine available,
excluding interpreter start-up.

### Timeouts

`_gather.py` carries a `TIMEOUTS_S` table of per-call timeouts keyed by
`(family, synthesizer, epsilon)`, calibrated from run history so that it
catches pathological calls only — every other call in that history finished
well below the limit. A √T call that hits it reports its Clifford+T circuit
instead. There is no fallback default: gathering at a precision outside the
table exits with an error, so set `GATHER_TIMEOUT_S` explicitly when extending
the ε range.

## Environment variables

| variable | default | effect |
| --- | --- | --- |
| `SKIP_GATHER` | unset | set to any value to skip the gathers and recompute from existing CSVs |
| `GATHER_N` | 500 | targets per precision |
| `GATHER_EPS` | the paper's ladder | comma-separated ε list to gather instead |
| `GATHER_OUT` | `scripts/data/{rz,u3}.csv` | output CSV path |
| `GATHER_TIMEOUT_S` | calibrated table | per-call timeout in seconds; see [Timeouts](#timeouts) |
| `GRIDSYNTH_BIN` | search `PATH` | path to the `gridsynth` binary |
| `GRIDSYNTH_SEED` | `1` | `-r` seed passed to `gridsynth` |
| `PAPER_STAT` | `median` | `mean` to report means instead |
| `RZ_CSV` | `scripts/data/rz.csv` | alternate z-rotation CSV for `paper_numbers.py` |
| `RAYON_NUM_THREADS` | rayon default | thread count for the library during a gather |
