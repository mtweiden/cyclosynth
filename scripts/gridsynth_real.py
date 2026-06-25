"""Drop-in replacement for the gridsynth wrappers in comparison_gridsynth.py
that calls Selinger--Ross's real `gridsynth` binary (Haskell `newsynth`)
instead of the pure-Python pygridsynth. Much faster and more reliable, so we
can gather many more targets at deep epsilon with no factoring timeouts.

Exposes the same API: grid_rz(theta, eps) -> gate string,
grid_u3(a, b, g, eps) -> (gate_string, T_count, achieved_dist), plus the
helpers rz, ry, to_u, dphase.

The binary is located via $GRIDSYNTH_BIN, then PATH, then the usual cabal/ghcup
install dirs. gridsynth output gates are {T,S,H,X,W} in *matrix order* (apply
right-to-left); to_u multiplies in written order, which is matrix order, so it
matches. We pass --phase (global-phase freedom -> fewer gates) and tune the
gridsynth epsilon up until the achieved diamond distance is just within the
target (gridsynth is never handicapped), mirroring the pygridsynth path.
"""
import os
import shutil
import subprocess
import numpy as np

W = np.exp(1j * np.pi / 4)
G = {'H': np.array([[1, 1], [1, -1]]) / np.sqrt(2), 'S': np.array([[1, 0], [0, 1j]]),
     'T': np.array([[1, 0], [0, W]]), 'X': np.array([[0, 1], [1, 0]]),
     'Y': np.array([[0, -1j], [1j, 0]]), 'Z': np.array([[1, 0], [0, -1]]),
     'W': W * np.eye(2), 'I': np.eye(2)}


def _find_binary():
    cand = [os.environ.get("GRIDSYNTH_BIN"), shutil.which("gridsynth")]
    home = os.path.expanduser("~")
    cand += [f"{home}/.cabal/bin/gridsynth", f"{home}/.local/bin/gridsynth",
             f"{home}/.ghcup/bin/gridsynth"]
    for c in cand:
        if c and os.path.exists(c):
            return c
    raise FileNotFoundError(
        "gridsynth binary not found; set GRIDSYNTH_BIN or `cabal install newsynth`")


GRIDSYNTH = None  # resolved lazily so the module imports before install finishes


def to_u(s):
    M = np.eye(2, dtype=complex)
    for c in s:
        M = M @ G[c]
    return M


def dphase(U, V):
    # Global-phase-optimal Frobenius/diamond distance. The aligning phase is
    # tr(U V^dag)/|.| (NOT tr(U^dag V), its conjugate); the wrong conjugate
    # spuriously inflates the distance whenever V carries a global phase.
    tr = np.trace(U @ V.conj().T)
    ph = tr / abs(tr) if abs(tr) > 1e-15 else 1.0
    f2 = np.linalg.norm(U - ph * V, 'fro') ** 2
    return np.sqrt(max(f2 * (8 - f2), 0.0)) / 4


def rz(t):
    return np.array([[np.exp(-1j * t / 2), 0], [0, np.exp(1j * t / 2)]])


def ry(t):
    c, s = np.cos(t / 2), np.sin(t / 2)
    return np.array([[c, -s], [s, c]])


def _call(theta, eps):
    """Run gridsynth for Rz(theta) at gridsynth-epsilon eps; return gate string."""
    global GRIDSYNTH
    if GRIDSYNTH is None:
        GRIDSYNTH = _find_binary()
    th = float(theta)
    s = f"{th:.17g}"
    arg = f"({s})" if th < 0 else s
    out = subprocess.run([GRIDSYNTH, "--phase", "-e", f"{eps:.3e}", arg],
                         capture_output=True, text=True, timeout=120)
    if out.returncode != 0:
        raise RuntimeError(f"gridsynth failed: {out.stderr.strip()}")
    # The gate string is the (last) line of all-gate-letter output.
    line = ""
    for ln in out.stdout.splitlines():
        t = ln.strip()
        if t and set(t) <= set("HSTXYZWI"):
            line = t
    return line


def grid_rz(theta, eps):
    """Loosest (min-T) gridsynth Rz string with achieved diamond dist <= eps."""
    target = rz(theta)
    s_ok = None
    # gridsynth's -e is an operator-norm bound, so eps itself already yields a
    # circuit within the target diamond distance; we probe one step looser
    # (fewer gates) and tighten only if needed. Loosest accepted is returned.
    for mult in (1.5, 1.25, 1.1, 1.0, 0.85, 0.7, 0.55, 0.4):
        s = _call(theta, eps * mult)
        if dphase(to_u(s), target) <= eps:
            return s          # loosest accepted = fewest T
        s_ok = s
    return s_ok


def grid_u3(a, b, g, eps):
    """gridsynth via the hardware-canonical ZSX decomposition

        U3 = Rz(a) Ry(b) Rz(g) = X . Rz(-a) . SX . Rz(b) . SX . Rz(g),

    SX = sqrt(X) = HSH (Clifford); the three Rz are synthesized independently.
    A shared per-rotation precision multiplier is tuned so the *total* diamond
    distance is just within eps. Returns (gate_string, T_count, achieved_dist)."""
    U3 = rz(a) @ ry(b) @ rz(g)
    best = None
    # Per-rotation budget swept loose->tight; the loosest split whose *total*
    # diamond distance is within eps is accepted (fair to gridsynth -- it gets
    # the easiest spec that still meets the target, never handicapped).
    for mult in (1.4, 1.1, 0.9, 0.7):
        e = eps * mult / 3.0
        sa, sb, sg = grid_rz(-a, e), grid_rz(b, e), grid_rz(g, e)
        s = "X" + sa + "HSH" + sb + "HSH" + sg
        d = dphase(to_u(s), U3)
        if d <= eps:
            return s, s.count('T'), d
        best = (s, s.count('T'), d)
    return best


if __name__ == "__main__":
    # Smoke test once the binary is installed.
    GRIDSYNTH = _find_binary()
    print("binary:", GRIDSYNTH)
    for eps in (1e-3, 1e-6, 1e-8):
        s = grid_rz(1.2345, eps)
        print(f"Rz(1.2345) eps={eps:.0e}: T={s.count('T')}  "
              f"len={len(s)}  dist={dphase(to_u(s), rz(1.2345)):.2e}")
