"""gridsynth (optimal Clifford+T z-rotation synthesis) vs cyclosynth.

Two comparisons, both at matched *diamond distance* (gridsynth's epsilon is
operator-norm, so its input epsilon is tuned upward until the achieved diamond
distance is just within tolerance — i.e. gridsynth is given its easiest fair
spec, never handicapped):

  A) Rz(theta): gridsynth vs cyclosynth Clifford+T vs Clifford+sqrt(T).
     Expectation: cyclosynth Clifford+T ~= gridsynth (validates our search is
     T-competitive on the case an optimal dedicated tool exists), sqrt(T) wins.

  B) U3 = Rz(a) Ry(b) Rz(g): gridsynth must Euler-decompose into 3 z-rotations
     (Ry(b) = S H Rz(b) H S^dag), each synthesized separately; cyclosynth does
     the general unitary directly. Expectation: gridsynth ~3x worse.

Cost: T-count for Clifford+T; T + 3*sqrt(T) for Clifford+sqrt(T).
"""
import sys, csv
import numpy as np
import mpmath
import pygridsynth
import cyclosynth

W = np.exp(1j * np.pi / 4)
G = {'H': np.array([[1, 1], [1, -1]]) / np.sqrt(2), 'S': np.array([[1, 0], [0, 1j]]),
     'T': np.array([[1, 0], [0, W]]), 'X': np.array([[0, 1], [1, 0]]),
     'Y': np.array([[0, -1j], [1j, 0]]), 'Z': np.array([[1, 0], [0, -1]]),
     'W': W * np.eye(2)}


def to_u(s):
    M = np.eye(2, dtype=complex)
    for c in s:
        M = M @ G[c]
    return M


def dphase(U, V):
    tr = np.trace(U @ V.conj().T)  # tr(U V^dag): correct global-phase alignment
    ph = tr / abs(tr) if abs(tr) > 1e-15 else 1.0
    f2 = np.linalg.norm(U - ph * V, 'fro') ** 2
    return np.sqrt(max(f2 * (8 - f2), 0.0)) / 4


def rz(t):
    return np.array([[np.exp(-1j * t / 2), 0], [0, np.exp(1j * t / 2)]])


def ry(t):
    c, s = np.cos(t / 2), np.sin(t / 2)
    return np.array([[c, -s], [s, c]])


def grid_rz(theta, eps):
    """Loosest (min-T) gridsynth Rz string with achieved diamond dist <= eps.
    Bounded Diophantine/factoring timeouts keep deep-eps runs from hanging."""
    s_ok = None
    for mult in (16, 8, 4, 2, 1, 0.5, 0.25):
        s = pygridsynth.gridsynth_gates(mpmath.mpf(theta), mpmath.mpf(eps * mult),
                                        dtimeout=8000, ftimeout=8000)
        if dphase(to_u(s), rz(theta)) <= eps:
            return s              # loosest accepted = fewest T
        s_ok = s
    return s_ok


def grid_u3(a, b, g, eps):
    """gridsynth via the hardware-canonical ZSX decomposition

        U3 = Rz(a) Ry(b) Rz(g) = X . Rz(-a) . SX . Rz(b) . SX . Rz(g),

    SX = sqrt(X) = HSH (Clifford); the three Rz are synthesized independently.
    A shared per-rotation precision multiplier is tuned so the *total* diamond
    distance is just within eps (fair to gridsynth, never handicapped).
    Returns (gate_string, T_count, achieved_dist)."""
    U3 = rz(a) @ ry(b) @ rz(g)
    best = None
    for mult in (4, 2, 1, 0.5, 0.25):
        e = eps * mult / 3.0
        sa, sb, sg = grid_rz(-a, e), grid_rz(b, e), grid_rz(g, e)
        s = "X" + sa + "HSH" + sb + "HSH" + sg         # X Rz(-a) SX Rz(b) SX Rz(g)
        d = dphase(to_u(s), U3)
        if d <= eps:
            return s, s.count('T'), d
        best = (s, s.count('T'), d)
    return best


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    epss = [1e-3, 1e-4, 1e-5, 1e-6]
    rng = np.random.default_rng(0xC0FFEE)
    # fixed target sets
    rz_angles = rng.uniform(0, 2 * np.pi, n)
    u3 = [(rng.uniform(0, 2 * np.pi), np.arccos(1 - 2 * rng.uniform()),
           rng.uniform(0, 2 * np.pi)) for _ in range(n)]

    def cyc_T(U, eps):
        r = cyclosynth.Synthesizer(epsilon=eps).synthesize(U)
        return r.t_count if r else None

    def cyc_Q(U, eps):
        r = cyclosynth.Synthesizer(epsilon=eps, sqrt_t=True,
                                   optimize_cost=True).synthesize(U)
        return r.cost if r else None

    rows = []
    print(f"# n={n} targets per eps, matched diamond distance\n")
    print("=== A) Rz(theta) ===")
    print(f"{'eps':>6} {'n':>4} {'grid_T':>7} {'cyc_T':>6} {'cyc_T/grid':>10} "
          f"{'cycQ_cost':>9} {'cycQ/grid':>9}")
    for eps in epss:
        gT, cT, cQ = [], [], []
        for th in rz_angles:
            t, q = cyc_T(rz(th), eps), cyc_Q(rz(th), eps)
            if t is None or q is None:
                continue                       # skip targets not reachable
            gT.append(grid_rz(th, eps).count('T')); cT.append(t); cQ.append(q)
        gT, cT, cQ = map(np.array, (gT, cT, cQ))
        print(f"{eps:>6.0e} {len(gT):>4} {np.median(gT):>7.0f} {np.median(cT):>6.0f} "
              f"{np.median(cT)/np.median(gT):>10.2f} {np.median(cQ):>9.1f} "
              f"{np.median(cQ)/np.median(gT):>9.2f}")
        rows.append(("Rz", eps, len(gT), np.median(gT), np.median(cT), np.median(cQ)))

    print("\n=== B) U3 = Rz(a) Ry(b) Rz(g) (gridsynth via ZSX) ===")
    print(f"{'eps':>6} {'n':>4} {'gridU3_T':>9} {'cyc_T':>6} {'grid/cyc':>9} "
          f"{'cycQ_cost':>9} {'grid/cycQ':>9}")
    for eps in epss:
        gT, cT, cQ = [], [], []
        for (a, b, g) in u3:
            U3 = rz(a) @ ry(b) @ rz(g)
            t, q = cyc_T(U3, eps), cyc_Q(U3, eps)
            if t is None or q is None:
                continue
            gT.append(grid_u3(a, b, g, eps)[1]); cT.append(t); cQ.append(q)
        gT, cT, cQ = map(np.array, (gT, cT, cQ))
        print(f"{eps:>6.0e} {len(gT):>4} {np.median(gT):>9.0f} {np.median(cT):>6.0f} "
              f"{np.median(gT)/np.median(cT):>9.2f} {np.median(cQ):>9.1f} "
              f"{np.median(gT)/np.median(cQ):>9.2f}")
        rows.append(("U3", eps, len(gT), np.median(gT), np.median(cT), np.median(cQ)))

    with open("scripts/data/comparison_gridsynth.csv", "w", newline="") as f:
        wtr = csv.writer(f)
        wtr.writerow(["kind", "eps", "n", "grid_T", "cyc_T", "cyc_sqrtT_cost"])
        wtr.writerows(rows)
    print("\nsaved scripts/data/comparison_gridsynth.csv")


if __name__ == "__main__":
    main()
