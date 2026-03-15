"""
Clifford+T synthesis via integer enumeration.

Key insight: y = sigma_to_xy @ uv is constrained to a 4D subspace:
    y[2k+1] = (y[2k-1+1] ... see relations below)

With uv = [Re(v1), Im(v1), Re(v2), Im(v2)]:
    y[0] = Re(v1)*s/2,  y[2] = Im(v1)*s/2,  y[4] = Re(v2)*s/2,  y[6] = Im(v2)*s/2
    y[1] = (y[0]+y[2])/sqrt(2)
    y[3] = (y[2]-y[0])/sqrt(2)
    y[5] = (y[4]+y[6])/sqrt(2)
    y[7] = (y[6]-y[4])/sqrt(2)

Substituting p_i = b_i + d_i, q_i = b_i - d_i:
- Unitarity:  c1*p1 + a1*q1 + c2*p2 + a2*q2 = 0      (clean bilinear form)
- Norm:       (p1^2+q1^2+p2^2+q2^2) = 2R              (R = 2^k - a1^2-c1^2-a2^2-c2^2)
- Alignment:  inner contribution = (y[2]*p1 + y[0]*q1 + y[6]*p2 + y[4]*q2) / sqrt(2)
- Parity:     p_i ≡ q_i (mod 2)  (so that b_i,d_i are integers)
"""

import numpy as np
from numpy import ndarray, sqrt, isclose
from math import gcd

# ---------------------------------------------------------------------------
# Coordinate transforms (from bandb3.py)
# ---------------------------------------------------------------------------
r2 = sqrt(2.0)

sigma_to_uv = np.array([
    [1, 1/r2, 0, -1/r2, 0, 0, 0, 0],
    [0, 1/r2, 1,  1/r2, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1/r2, 0, -1/r2],
    [0, 0, 0, 0, 0, 1/r2, 1,  1/r2],
])
sigma_to_xy = 0.5 * np.array([
    [ 1,    0,    0, 0],
    [ 1/r2, 1/r2, 0, 0],
    [ 0,    1,    0, 0],
    [-1/r2, 1/r2, 0, 0],
    [0, 0,  1,    0  ],
    [0, 0,  1/r2, 1/r2],
    [0, 0,  0,    1  ],
    [0, 0, -1/r2, 1/r2],
])

def uv_to_xy(uv: ndarray, k: int) -> ndarray:
    scale = 2 ** (k / 2)
    return sigma_to_xy @ uv * scale

def xy_to_uv(xy: ndarray, k: int) -> ndarray:
    scale = 2 ** (k / 2)
    return sigma_to_uv @ xy / scale

def to_unitary(x: ndarray, k: int) -> ndarray:
    u = xy_to_uv(x, k)
    u1 = u[0] + 1j * u[1]
    u2 = u[2] + 1j * u[3]
    return np.array([[u1, -u2.conj()], [u2, u1.conj()]])

# ---------------------------------------------------------------------------
# Integer null-space basis for a single row vector w ∈ Z^4
# ---------------------------------------------------------------------------
def _extended_gcd(a: int, b: int):
    if b == 0:
        return a, 1, 0
    g, s, t = _extended_gcd(b, a % b)
    return g, t, s - (a // b) * t

def integer_null_basis(w: ndarray) -> ndarray:
    """
    Given integer row vector w of length n, return (n-1) x n integer matrix N
    such that  N @ w == 0  and  N has full row rank n-1.

    Algorithm: reduce w to [g, 0, ..., 0] via unimodular column operations;
    the last n-1 columns of the transformation matrix form the null basis.
    """
    w = np.array(w, dtype=np.int64).copy()
    n = len(w)
    U = np.eye(n, dtype=np.int64)

    for i in range(1, n):
        if w[i] == 0:
            continue
        g, s, t = _extended_gcd(int(w[0]), int(w[i]))
        new_col0 = s * U[:, 0] + t * U[:, i]
        new_coli = -(w[i] // g) * U[:, 0] + (w[0] // g) * U[:, i]
        U[:, 0] = new_col0
        U[:, i] = new_coli
        w[0] = g
        w[i] = 0

    return U[:, 1:].T  # (n-1) x n

# ---------------------------------------------------------------------------
# LLL basis reduction
# ---------------------------------------------------------------------------
def lll_reduce(B: ndarray, delta: float = 0.75):
    """
    LLL-reduce integer basis B (rows = basis vectors, shape m x n).
    Returns reduced basis as integer array.
    """
    B = np.array(B, dtype=np.int64)
    m = len(B)
    Bf = B.astype(float)

    def gram_schmidt(Bf):
        Bs = Bf.copy()
        mu = np.zeros((m, m))
        for i in range(m):
            for j in range(i):
                mu[i, j] = Bf[i] @ Bs[j] / (Bs[j] @ Bs[j])
                Bs[i] = Bs[i] - mu[i, j] * Bs[j]
        return Bs, mu

    k = 1
    while k < m:
        Bs, mu = gram_schmidt(Bf)
        for j in range(k - 1, -1, -1):
            r = int(round(mu[k, j]))
            if r != 0:
                Bf[k] -= r * Bf[j]
                B[k] -= r * B[j]
                Bs, mu = gram_schmidt(Bf)
        if Bs[k] @ Bs[k] >= (delta - mu[k, k-1]**2) * (Bs[k-1] @ Bs[k-1]):
            k += 1
        else:
            Bf[[k, k-1]] = Bf[[k-1, k]]
            B[[k, k-1]] = B[[k-1, k]]
            k = max(k - 1, 1)
    return B

# ---------------------------------------------------------------------------
# Phase 2: CVP in (p, q) space via LLL + Schnorr-Euchner
# ---------------------------------------------------------------------------
def phase2_pq(a1: int, c1: int, a2: int, c2: int,
              R: int, y_full: ndarray) -> list:
    """
    Find all (b1, d1, b2, d2) satisfying:
        unitarity:  c1*p1 + a1*q1 + c2*p2 + a2*q2 = 0   where p_i=b_i+d_i, q_i=b_i-d_i
        norm:       b1^2+d1^2+b2^2+d2^2 = R
        parity:     p_i ≡ q_i (mod 2)   <=> b_i, d_i ∈ Z  (automatic)

    CVP target direction in (p,q) space:  (y[2], y[0], y[6], y[4])
    (derived from the y-symmetry relations).

    Uses the (p,q) lattice:
        L_pq = { (p1,q1,p2,q2) ∈ Z^4 : c1*p1 + a1*q1 + c2*p2 + a2*q2 = 0 }
    with parity enforced by working in the (b,d) sub-lattice.
    """
    if R < 0:
        return []
    if R == 0:
        return [(0, 0, 0, 0)]

    # Unitarity constraint in (p,q) form: w_pq · (p1,q1,p2,q2) = 0
    w_pq = np.array([c1, a1, c2, a2], dtype=np.int64)

    # Null basis: 3x4 integer matrix in (p,q) space
    N_pq = integer_null_basis(w_pq)   # shape (3, 4)

    # The parity sublattice: p_i ≡ q_i (mod 2).
    # Enforce by restricting to (b,d) coordinates:
    #   p = b + d, q = b - d  =>  (p,q) = M * (b,d)
    #   M = [[1,0,1,0],[0,1,0,1],[1,0,-1,0],[0,1,0,-1]]  (reordered)
    # More directly: work in (b,d) coordinates throughout.
    # The (b,d) null basis is N_bd = N_pq @ M where M maps (b,d) -> (p,q).
    #   M[0] = (1,0,1,0)  (p1 = b1+d1, indexed as col of (b1,d1,b2,d2))
    # Actually M in the ordering (p1,q1,p2,q2) <- (b1,d1,b2,d2):
    M_pq_to_bd = np.array([
        [1, 1, 0, 0],   # p1 = b1 + d1
        [1,-1, 0, 0],   # q1 = b1 - d1
        [0, 0, 1, 1],   # p2 = b2 + d2
        [0, 0, 1,-1],   # q2 = b2 - d2
    ], dtype=np.int64)

    N_bd = N_pq @ M_pq_to_bd  # (3, 4) null basis in (b,d) space

    # LLL-reduce
    N_lll = lll_reduce(N_bd)   # (3, 4)

    # CVP target: maximize inner alignment contribution.
    # In (p,q): direction is (y[2], y[0], y[6], y[4]) (from y-symmetry).
    # Pull back to (b,d): dir_bd = M^T @ dir_pq / (some factor)
    # Directly: y_inner = (y[1],y[3],y[5],y[7]), same as before.
    y_inner = np.array([y_full[1], y_full[3], y_full[5], y_full[7]])
    norm_yi = np.linalg.norm(y_inner)

    if norm_yi < 1e-12:
        t_ambient = np.zeros(4)
    else:
        t_ambient = y_inner * sqrt(float(R)) / norm_yi

    # Express t_ambient in lattice coordinates via least-squares projection
    G = N_lll.astype(float) @ N_lll.astype(float).T   # 3x3 Gram matrix
    try:
        t_lat = np.linalg.solve(G, N_lll.astype(float) @ t_ambient)
    except np.linalg.LinAlgError:
        t_lat = np.zeros(3)

    solutions = []
    _schnorr_euchner(N_lll, t_lat, R, a1, c1, a2, c2, solutions)
    return solutions


def _schnorr_euchner(N: ndarray, t_lat: ndarray, R: int,
                     a1: int, c1: int, a2: int, c2: int,
                     solutions: list):
    """
    Enumerate integer points z ∈ Z^3 such that ||N^T z||^2 = R,
    centered around t_lat.  Converts each valid z to (b1,d1,b2,d2).

    Uses a simple layer-by-layer enumeration in the QR-reduced basis,
    equivalent to Schnorr-Euchner in dimension 3.
    """
    Nf = N.astype(float)
    # QR decomposition of N^T: N^T = Q R  =>  ||N^T z||^2 = ||R z||^2
    # But N is 3x4, so N^T is 4x3.  Use economy QR.
    _, R_mat = np.linalg.qr(Nf.T, mode='reduced')  # R_mat is 3x3 upper triangular

    # Flip sign so diagonal of R_mat is positive
    signs = np.sign(np.diag(R_mat))
    signs[signs == 0] = 1
    R_mat = signs[:, None] * R_mat

    # Work in coordinates w = R_mat @ z, target w0 = R_mat @ t_lat
    # ||N^T z||^2 = ||R_mat z||^2 (up to orthogonal factor; norms preserved by Q)
    w0 = R_mat @ t_lat

    # Bound on ||z||: ||N^T z|| = sqrt(R), and smallest singular value of N
    # gives ||z|| <= sqrt(R) / sigma_min.  Use a loose bound.
    radius_sq = float(R)

    # Enumerate z[2] (outermost index in upper-triangular SE)
    r22 = R_mat[2, 2]
    if abs(r22) < 1e-12:
        return
    z2_center = w0[2] / r22
    z2_lo = int(np.floor(z2_center - sqrt(radius_sq) / abs(r22))) - 1
    z2_hi = int(np.ceil( z2_center + sqrt(radius_sq) / abs(r22))) + 1

    for z2 in range(z2_lo, z2_hi + 1):
        rem2 = radius_sq - (R_mat[2, 2] * z2 - w0[2]) ** 2
        if rem2 < -1e-9:
            continue

        # Enumerate z[1]
        r11 = R_mat[1, 1]
        r12 = R_mat[1, 2]
        if abs(r11) < 1e-12:
            continue
        w1_eff = w0[1] - r12 * z2
        z1_center = w1_eff / r11
        z1_lo = int(np.floor(z1_center - sqrt(max(rem2, 0)) / abs(r11))) - 1
        z1_hi = int(np.ceil( z1_center + sqrt(max(rem2, 0)) / abs(r11))) + 1

        for z1 in range(z1_lo, z1_hi + 1):
            rem1 = rem2 - (R_mat[1, 1] * z1 + R_mat[1, 2] * z2 - w0[1]) ** 2
            if rem1 < -1e-9:
                continue

            # Solve for z[0]
            r00 = R_mat[0, 0]
            r01 = R_mat[0, 1]
            r02 = R_mat[0, 2]
            if abs(r00) < 1e-12:
                continue
            w0_eff = w0[0] - r01 * z1 - r02 * z2
            # Need (r00 * z0 - w0_eff)^2 = rem1
            if rem1 < -1e-9:
                continue
            val = sqrt(max(rem1, 0))
            for sign in (+1, -1):
                z0f = (w0_eff + sign * val) / r00
                z0 = int(round(z0f))
                # Verify exactly
                z = np.array([z0, z1, z2], dtype=np.int64)
                bd = N.T @ z   # (b1, d1, b2, d2)
                b1, d1, b2, d2 = int(bd[0]), int(bd[1]), int(bd[2]), int(bd[3])
                # Check norm exactly
                if b1*b1 + d1*d1 + b2*b2 + d2*d2 != R:
                    continue
                # Check unitarity exactly
                if b1*(a1+c1) + d1*(c1-a1) + b2*(a2+c2) + d2*(c2-a2) != 0:
                    continue
                solutions.append((b1, d1, b2, d2))


# ---------------------------------------------------------------------------
# Phase 1: enumerate (a1, c1, a2, c2) with Cauchy-Schwarz pruning
# ---------------------------------------------------------------------------
def phase1_enumerate(y: ndarray, k: int,
                     eps: float = 1e-4) -> list:
    """
    Enumerate outer variables (a1, c1, a2, c2) centered on the projection
    of y_outer = (y[0], y[2], y[4], y[6]).

    Cauchy-Schwarz pruning: after fixing partial outer vars x_partial,
        (x_partial · y_partial + sqrt(R_rem) * ||y_rem||)^2 >= threshold
    must hold, otherwise skip.

    Returns list of full 8-vectors (a1,b1,c1,d1,a2,b2,c2,d2).
    """
    target_norm = 2 ** k
    threshold_sq = (2 ** (k - 1)) * (1 - eps ** 2)  # alignment threshold on |x·y|^2 / 2^k

    # Scale factor so that ||y_scaled|| = sqrt(target_norm)
    y_norm = np.linalg.norm(y)
    scale = sqrt(target_norm) / y_norm
    y_outer = np.array([y[0], y[2], y[4], y[6]])

    # Centers for outer vars
    a1_c = int(round(y[0] * scale))
    c1_c = int(round(y[2] * scale))
    a2_c = int(round(y[4] * scale))
    c2_c = int(round(y[6] * scale))

    max_outer = int(sqrt(target_norm)) + 1
    solutions = []

    # Precompute inner y norms for Cauchy-Schwarz
    y_inner_sq = y[1]**2 + y[3]**2 + y[5]**2 + y[7]**2  # = y_outer_sq (from symmetry)

    for a1 in _centered_range(a1_c, max_outer):
        if a1 * a1 > target_norm:
            continue
        rem_a1 = target_norm - a1 * a1

        # Cauchy-Schwarz pruning after a1:
        # max alignment from remaining vars <= (a1*y[0] + sqrt(rem_a1)*||y_rest||)
        dot_a1 = a1 * y[0]
        y_rest_sq = y[1]**2 + y[2]**2 + y[3]**2 + y[4]**2 + y[5]**2 + y[6]**2 + y[7]**2
        if (abs(dot_a1) + sqrt(rem_a1 * y_rest_sq)) ** 2 < threshold_sq * y_norm**2 / (target_norm / (2**(k-1))):
            pass  # (loose prune; tighten below)

        for c1 in _centered_range(c1_c, max_outer):
            if a1*a1 + c1*c1 > target_norm:
                continue
            rem_c1 = target_norm - a1*a1 - c1*c1

            dot_a1c1 = a1*y[0] + c1*y[2]
            y_rest2_sq = y[1]**2 + y[3]**2 + y[4]**2 + y[5]**2 + y[6]**2 + y[7]**2
            # Cauchy-Schwarz: |dot_a1c1 + inner| <= |dot_a1c1| + sqrt(rem_c1)*||y_rem||
            if (abs(dot_a1c1) + sqrt(rem_c1 * y_rest2_sq)) ** 2 < threshold_sq * (y_norm ** 2):
                continue

            for a2 in _centered_range(a2_c, max_outer):
                if a1*a1 + c1*c1 + a2*a2 > target_norm:
                    continue
                rem_a2 = target_norm - a1*a1 - c1*c1 - a2*a2

                dot_outer3 = dot_a1c1 + a2*y[4]
                y_rest3_sq = y[1]**2 + y[3]**2 + y[5]**2 + y[6]**2 + y[7]**2
                if (abs(dot_outer3) + sqrt(rem_a2 * y_rest3_sq)) ** 2 < threshold_sq * (y_norm ** 2):
                    continue

                for c2 in _centered_range(c2_c, max_outer):
                    outer_norm_sq = a1*a1 + c1*c1 + a2*a2 + c2*c2
                    if outer_norm_sq > target_norm:
                        continue
                    R = target_norm - outer_norm_sq

                    # Cauchy-Schwarz pruning: outer dot + inner bound
                    dot_outer = dot_outer3 + c2*y[6]
                    # Inner contribution bounded by sqrt(R) * ||y_inner||
                    y_inner_norm = sqrt(y_inner_sq)
                    if (abs(dot_outer) + sqrt(R) * y_inner_norm) ** 2 < threshold_sq * (y_norm ** 2):
                        continue

                    # Phase 2: find inner variables via LLL+CVP
                    inner_solutions = phase2_pq(a1, c1, a2, c2, R, y)

                    for (b1, d1, b2, d2) in inner_solutions:
                        x = np.array([a1, b1, c1, d1, a2, b2, c2, d2], dtype=np.int64)
                        solutions.append(x)

    return solutions


def _centered_range(center: int, max_radius: int):
    """Yield integers outward from center: center, center+1, center-1, ..."""
    yield center
    for off in range(1, max_radius + abs(center) + 2):
        yield center + off
        yield center - off


# ---------------------------------------------------------------------------
# Top-level synthesis
# ---------------------------------------------------------------------------
def synthesize(v: ndarray, k: int, eps: float = 1e-4) -> list:
    """
    Given target unitary v = [Re(v1), Im(v1), Re(v2), Im(v2)],
    find all x = (a1,b1,c1,d1,a2,b2,c2,d2) ∈ Z^8 with T-count k satisfying
    the norm, unitarity, and alignment constraints.
    """
    y = uv_to_xy(v, k)
    return phase1_enumerate(y, k, eps)


# ---------------------------------------------------------------------------
# Verification helpers
# ---------------------------------------------------------------------------
def verify(x: ndarray, k: int, y: ndarray, eps: float = 1e-4) -> dict:
    a1, b1, c1, d1, a2, b2, c2, d2 = x
    norm_ok = (a1**2 + b1**2 + c1**2 + d1**2 + a2**2 + b2**2 + c2**2 + d2**2 == 2**k)
    unit_ok = isclose(
        b1*(a1+c1) + d1*(c1-a1) + b2*(a2+c2) + d2*(c2-a2), 0
    )
    dot = float(x @ y)
    align_ok = dot**2 > (2**(k-1)) * (1 - eps**2) * np.dot(y, y) / (2**(k-1))
    return {"norm": norm_ok, "unitarity": unit_ok, "alignment": align_ok, "dot": dot}


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(42)
    k = 6

    # Random unit target
    v = np.random.randn(4)
    v /= np.linalg.norm(v)

    print(f"k = {k}, target v = {v}")
    y = uv_to_xy(v, k)
    print(f"y = {np.round(y, 4)}")

    # Verify y-symmetry relations
    print("\nVerifying y-symmetry:")
    print(f"  y[1] vs (y[0]+y[2])/sqrt(2): {y[1]:.6f} vs {(y[0]+y[2])/r2:.6f}")
    print(f"  y[3] vs (y[2]-y[0])/sqrt(2): {y[3]:.6f} vs {(y[2]-y[0])/r2:.6f}")
    print(f"  y[5] vs (y[4]+y[6])/sqrt(2): {y[5]:.6f} vs {(y[4]+y[6])/r2:.6f}")
    print(f"  y[7] vs (y[6]-y[4])/sqrt(2): {y[7]:.6f} vs {(y[6]-y[4])/r2:.6f}")

    # Test null basis
    a1, c1, a2, c2 = 3, 1, -2, 4
    w_pq = np.array([c1, a1, c2, a2], dtype=np.int64)
    N = integer_null_basis(w_pq)
    print(f"\nNull basis check (should be zeros): {N @ w_pq}")

    # Test LLL
    N_lll = lll_reduce(N)
    print(f"LLL reduced null basis:\n{N_lll}")
    print(f"LLL null check (should be zeros): {N_lll @ w_pq}")

    # Run synthesis
    print(f"\nRunning synthesis (k={k}, eps=1e-2)...")
    eps = 1e-2
    solutions = synthesize(v, k, eps)
    print(f"Found {len(solutions)} solutions")

    for x in solutions[:5]:
        r = verify(x, k, y, eps)
        print(f"  x={x}  norm={r['norm']}  unit={r['unitarity']}  align={r['alignment']}  dot={r['dot']:.2f}")