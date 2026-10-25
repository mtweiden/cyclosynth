"""Shared plumbing for the paper's gather scripts (gather_u3.py, gather_rz.py).

- Every cyclosynth call runs in its own spawned process and is timed there, so
  the measured wall clock is one synthesize() call with the whole machine
  (rayon's default pool), excluding interpreter start-up and construction.
- Every circuit is re-verified independently of the library: the gate string
  is multiplied out in mpmath (40 digits) and the diamond distance to the
  target recomputed. `verified` is that check; `success` requires it.
- Output CSVs are resumable: rows already present for (epsilon, trial,
  synthesizer) are kept and skipped, so a long run can be interrupted.
- A JSON sidecar records the host, thread count, cyclosynth commit and config.
"""
import csv
import json
import multiprocessing as mp
import os
import platform
import subprocess
from time import perf_counter

import mpmath

mpmath.mp.dps = 40

# ─── Exact-ish re-verification ───────────────────────────────────────────────
_W = mpmath.exp(1j * mpmath.pi / 4)
_Q = mpmath.exp(1j * mpmath.pi / 8)
_R2 = 1 / mpmath.sqrt(2)
_GATES = {
    "H": mpmath.matrix([[_R2, _R2], [_R2, -_R2]]),
    "S": mpmath.matrix([[1, 0], [0, 1j]]),
    "s": mpmath.matrix([[1, 0], [0, -1j]]),
    "T": mpmath.matrix([[1, 0], [0, _W]]),
    "t": mpmath.matrix([[1, 0], [0, mpmath.conj(_W)]]),
    "Q": mpmath.matrix([[1, 0], [0, _Q]]),
    "q": mpmath.matrix([[1, 0], [0, mpmath.conj(_Q)]]),
    "X": mpmath.matrix([[0, 1], [1, 0]]),
    "Y": mpmath.matrix([[0, -1j], [1j, 0]]),
    "Z": mpmath.matrix([[1, 0], [0, -1]]),
    "W": _W * mpmath.eye(2),
    "I": mpmath.eye(2),
}


def unitary_of(gates):
    """Matrix of a gate string in matrix (operator-product) order, the order
    both cyclosynth and gridsynth emit: "HT" is the matrix H*T."""
    m = mpmath.eye(2)
    for c in gates:
        m = m * _GATES[c]
    return m


def rz(t):
    t = mpmath.mpf(t)
    return mpmath.matrix([[mpmath.exp(-1j * t / 2), 0], [0, mpmath.exp(1j * t / 2)]])


def ry(t):
    t = mpmath.mpf(t)
    c, s = mpmath.cos(t / 2), mpmath.sin(t / 2)
    return mpmath.matrix([[c, -s], [s, c]])


def diamond(u, v):
    """d(U,V) = sqrt(1 - |tr(U^dag V)|^2 / 4) (half-normalized diamond norm)."""
    tr = (u.H * v)[0, 0] + (u.H * v)[1, 1]
    return float(mpmath.sqrt(max(mpmath.mpf(0), 1 - abs(tr) ** 2 / 4)))


# ─── Per-call timeouts ───────────────────────────────────────────────────────
# Hard per-call cap = max(30 s, 5 x the slowest completed call of that (family,
# gate set, eps) in the 2026-10-17 run on an Apple M4 Pro (scripts/data/{u3,rz}.csv,
# library-default sqrt(T) deadline). The table was first derived from the
# 2026-06-26 runs; the three calls that hit those older caps were rerun under
# this one, and every other call finished well below both. It catches
# pathological calls only; a sqrt(T) call that hits it reports its Clifford+T
# circuit (see sqrt_t_gates). Precisions outside that history need
# GATHER_TIMEOUT_S.
TIMEOUTS_S = {
    "u3": {"cyclosynth_t": {"1e-03": 30, "1e-04": 30, "1e-05": 30, "1e-06": 30, "1e-07": 30, "1e-08": 39},
           "cyclosynth_sqrt_t": {"1e-03": 30, "1e-04": 30, "1e-05": 30, "1e-06": 30, "1e-07": 30, "1e-08": 70}},
    "rz": {"cyclosynth_t": {"1e-03": 30, "1e-04": 30, "1e-05": 30, "1e-06": 30, "1e-07": 229, "1e-08": 1547},
           "cyclosynth_sqrt_t": {"1e-03": 30, "1e-04": 34, "1e-05": 30, "1e-06": 30, "1e-07": 237, "1e-08": 2194}},
}


def timeout_for(family, synth, eps):
    """Per-call timeout in seconds (GATHER_TIMEOUT_S overrides the table)."""
    if os.environ.get("GATHER_TIMEOUT_S"):
        return float(os.environ["GATHER_TIMEOUT_S"])
    try:
        return float(TIMEOUTS_S[family][synth][f"{eps:.0e}"])
    except KeyError:
        raise SystemExit(f"no historical timeout for {family}/{synth} at {eps:.0e}; "
                         "set GATHER_TIMEOUT_S") from None


# ─── Isolated, timed cyclosynth calls ────────────────────────────────────────
_CTX = mp.get_context("spawn")


def _worker(q, synth_kwargs, method, args):
    import cyclosynth
    synth = cyclosynth.Synthesizer(**synth_kwargs)
    t0 = perf_counter()
    try:
        r = getattr(synth, method)(*args)
        dur = (perf_counter() - t0) * 1000.0
        q.put(dict(gates=(r.gates or "") if r else "", duration_ms=dur,
                   lde=r.lde if r else -1))
    except Exception as e:                                # noqa: BLE001
        q.put(dict(gates="", duration_ms=(perf_counter() - t0) * 1000.0,
                   lde=-1, error=repr(e)))


def run_cyc(synth_kwargs, method, args, timeout_s):
    """One synthesize call in a fresh process. A call still running after
    timeout_s is killed and reported with timed_out=True."""
    q = _CTX.Queue()
    p = _CTX.Process(target=_worker, args=(q, synth_kwargs, method, args))
    p.start()
    p.join(timeout_s)
    if p.is_alive():
        p.kill()
        p.join()
        return dict(gates="", duration_ms=timeout_s * 1000.0, lde=-1, timed_out=True)
    try:
        return q.get(timeout=30)
    except Exception:                                     # noqa: BLE001
        return dict(gates="", duration_ms=float("nan"), lde=-1, error="no result")


# ─── Resumable CSV + metadata ────────────────────────────────────────────────
FIELDS_COMMON = ["synthesizer", "t_count", "q_count", "cost", "distance",
                 "verified", "duration_ms", "timed_out", "success", "gates"]


def load_done(path):
    """Existing rows keyed by (epsilon, trial, synthesizer)."""
    if not os.path.exists(path):
        return {}
    with open(path, newline="") as f:
        return {(r["epsilon"], r["trial"], r["synthesizer"]): r for r in csv.DictReader(f)}


def open_writer(path, fields):
    new = not os.path.exists(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    f = open(path, "a", newline="")
    w = csv.DictWriter(f, fieldnames=fields)
    if new:
        w.writeheader()
    return f, w


def _cmd(args):
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:                                     # noqa: BLE001
        return ""


def write_metadata(path, config):
    """Sidecar <csv>.meta.json: host, cores, cyclosynth commit, config."""
    import cyclosynth
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cpu = (_cmd(["sysctl", "-n", "machdep.cpu.brand_string"])
           or _cmd(["sh", "-c", "grep -m1 'model name' /proc/cpuinfo | cut -d: -f2"]))
    meta = dict(
        host=platform.node(), platform=platform.platform(), cpu=cpu,
        logical_cpus=os.cpu_count(),
        rayon_num_threads=os.environ.get("RAYON_NUM_THREADS", "default"),
        python=platform.python_version(), cyclosynth_module=cyclosynth.__file__,
        git_commit=_cmd(["git", "-C", repo, "rev-parse", "HEAD"]),
        git_dirty=bool(_cmd(["git", "-C", repo, "status", "--porcelain", "--", "src"])),
        config=config,
    )
    with open(path + ".meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    if meta["git_dirty"]:   # record exactly what differed from the commit
        with open(path + ".src.diff", "w") as f:
            f.write(_cmd(["git", "-C", repo, "diff", "--", "src"]))
    return meta


def counts(gates):
    """(t_count, q_count) gate counts; lowercase letters are adjoints."""
    return gates.count("T") + gates.count("t"), gates.count("Q") + gates.count("q")


def sqrt_t_gates(rq, rt):
    """The sqrt(T) circuit to report. A sqrt(T) call that hit the timeout falls
    back to the Clifford+T circuit (Algorithm 2's floor), and its row keeps
    timed_out=True so paper_numbers.py can count these."""
    return rt["gates"] if rq.get("timed_out") else rq["gates"]
