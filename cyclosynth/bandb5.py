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
        unitarity:  (a1+c1)*b1 + (c1-a1)*d1 + (a2+c2)*b2 + (c2-a2)*d2 = 0
        norm:       b1^2 + d1^2 + b2^2 + d2^2 = R

    Null basis computed directly in (b,d) space — NOT via (p,q) and M, which
    only generates an index-2 sublattice and misses solutions where the two
    (p,q) component pairs have different parities.
    """
    if R < 0:
        return []
    if R == 0:
        return [(0, 0, 0, 0)]

    # Unitarity constraint in (b,d) form: w_bd · (b1,d1,b2,d2) = 0
    w_bd = np.array([a1+c1, c1-a1, a2+c2, c2-a2], dtype=np.int64)

    # Degenerate case: all outer vars zero -> unitarity trivially satisfied
    # -> enumerate full 4D sphere directly
    if not np.any(w_bd):
        results = []
        max_b1 = int(R ** 0.5)
        for b1 in range(-max_b1, max_b1 + 1):
            rem1 = R - b1*b1
            if rem1 < 0: continue
            for d1 in range(-int(rem1**0.5), int(rem1**0.5) + 1):
                rem2 = rem1 - d1*d1
                if rem2 < 0: continue
                for b2 in range(-int(rem2**0.5), int(rem2**0.5) + 1):
                    rem3 = rem2 - b2*b2
                    if rem3 < 0: continue
                    d2s = int(rem3**0.5)
                    if d2s*d2s != rem3: continue
                    for d2 in ([d2s] if d2s == 0 else [d2s, -d2s]):
                        results.append((b1, d1, b2, d2))
        return results

    # Null basis: 3x4 integer matrix in (b,d) space
    N_bd = integer_null_basis(w_bd)   # shape (3, 4)

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
    Enumerate all integer points z ∈ Z^3 such that ||N^T z||^2 = R exactly.
    t_lat is the CVP target in lattice coordinates — used only for centering
    the search (ordering candidates nearest-first), NOT shifted into residuals.
    Converts each valid z to (b1,d1,b2,d2) via bd = N^T @ z.
    """
    Nf = N.astype(float)
    # QR: N^T = Q R_mat, so ||N^T z||^2 = ||R_mat z||^2
    _, R_mat = np.linalg.qr(Nf.T, mode='reduced')  # R_mat is 3x3 upper triangular
    signs = np.sign(np.diag(R_mat)); signs[signs == 0] = 1
    R_mat = signs[:, None] * R_mat

    radius = float(R) ** 0.5

    # Enumerate z2 centered at t_lat[2]
    r22 = R_mat[2, 2]
    if abs(r22) < 1e-12:
        return
    z2_center = t_lat[2]
    z2_lo = int(np.floor(z2_center - radius / abs(r22))) - 1
    z2_hi = int(np.ceil( z2_center + radius / abs(r22))) + 1

    for z2 in range(z2_lo, z2_hi + 1):
        # Exact: rem2 = R - (r22*z2)^2
        rem2 = float(R) - (R_mat[2, 2] * z2) ** 2
        if rem2 < -1e-9:
            continue

        r11 = R_mat[1, 1]; r12 = R_mat[1, 2]
        if abs(r11) < 1e-12:
            continue
        # Center z1: project t_lat onto z1 axis given z2
        z1_center = t_lat[1] - (r12 / r11) * z2
        z1_lo = int(np.floor(z1_center - rem2**0.5 / abs(r11))) - 1
        z1_hi = int(np.ceil( z1_center + rem2**0.5 / abs(r11))) + 1

        for z1 in range(z1_lo, z1_hi + 1):
            # Exact: rem1 = rem2 - (r11*z1 + r12*z2)^2
            rem1 = rem2 - (R_mat[1, 1] * z1 + R_mat[1, 2] * z2) ** 2
            if rem1 < -1e-9:
                continue

            # Solve exactly for z0: (r00*z0 + r01*z1 + r02*z2)^2 = rem1
            r00 = R_mat[0, 0]; r01 = R_mat[0, 1]; r02 = R_mat[0, 2]
            if abs(r00) < 1e-12:
                continue
            inner = r01 * z1 + r02 * z2
            val = (max(rem1, 0)) ** 0.5
            for sign in (+1.0, -1.0):
                z0f = (-inner + sign * val) / r00
                z0 = int(round(z0f))
                z = np.array([z0, z1, z2], dtype=np.int64)
                bd = N.T @ z   # (b1, d1, b2, d2)
                b1, d1, b2, d2 = int(bd[0]), int(bd[1]), int(bd[2]), int(bd[3])
                # Verify norm exactly
                if b1*b1 + d1*d1 + b2*b2 + d2*d2 != R:
                    continue
                # Verify unitarity exactly
                if b1*(a1+c1) + d1*(c1-a1) + b2*(a2+c2) + d2*(c2-a2) != 0:
                    continue
                sol = (b1, d1, b2, d2)
                if sol not in solutions:
                    solutions.append(sol)


# ---------------------------------------------------------------------------
# Phase 1: enumerate (a1, c1, a2, c2) with Cauchy-Schwarz pruning
# ---------------------------------------------------------------------------
def phase1_enumerate(y: ndarray, k: int,
                     eps: float = 1e-4,
                     max_solutions: int = None) -> list:
    """
    Enumerate outer variables (a1, c1, a2, c2) centered on the projection
    of y_outer = (y[0], y[2], y[4], y[6]).

    Cauchy-Schwarz pruning: after fixing partial outer vars x_partial,
        (x_partial · y_partial + sqrt(R_rem) * ||y_rem||)^2 >= threshold
    must hold, otherwise skip.

    Returns list of full 8-vectors (a1,b1,c1,d1,a2,b2,c2,d2).
    """
    target_norm = 2 ** k
    # Correct threshold: x·y = 2^(k-1)·u·v, so |x·y|² > 2^(2k-2)·(1-eps²)
    # Since ||y||² = 2^(k-1), this equals ||y||^4 · (1-eps²) / 2^(k-1)... 
    # simplest: just use 2^(2k-2)·(1-eps²) directly.
    threshold_xy = 2 ** (2*k - 2) * (1 - eps ** 2)

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

        dot_a1 = a1 * y[0]
        y_rest_sq = y[1]**2 + y[2]**2 + y[3]**2 + y[4]**2 + y[5]**2 + y[6]**2 + y[7]**2
        if (abs(dot_a1) + sqrt(rem_a1 * y_rest_sq)) ** 2 < threshold_xy:
            continue

        for c1 in _centered_range(c1_c, max_outer):
            if a1*a1 + c1*c1 > target_norm:
                continue
            rem_c1 = target_norm - a1*a1 - c1*c1

            dot_a1c1 = a1*y[0] + c1*y[2]
            y_rest2_sq = y[1]**2 + y[3]**2 + y[4]**2 + y[5]**2 + y[6]**2 + y[7]**2
            if (abs(dot_a1c1) + sqrt(rem_c1 * y_rest2_sq)) ** 2 < threshold_xy:
                continue

            for a2 in _centered_range(a2_c, max_outer):
                if a1*a1 + c1*c1 + a2*a2 > target_norm:
                    continue
                rem_a2 = target_norm - a1*a1 - c1*c1 - a2*a2

                dot_outer3 = dot_a1c1 + a2*y[4]
                y_rest3_sq = y[1]**2 + y[3]**2 + y[5]**2 + y[6]**2 + y[7]**2
                if (abs(dot_outer3) + sqrt(rem_a2 * y_rest3_sq)) ** 2 < threshold_xy:
                    continue

                for c2 in _centered_range(c2_c, max_outer):
                    outer_norm_sq = a1*a1 + c1*c1 + a2*a2 + c2*c2
                    if outer_norm_sq > target_norm:
                        continue
                    R = target_norm - outer_norm_sq

                    dot_outer = dot_outer3 + c2*y[6]
                    y_inner_norm = sqrt(y_inner_sq)
                    if (abs(dot_outer) + sqrt(R) * y_inner_norm) ** 2 < threshold_xy:
                        continue

                    # Phase 2: find inner variables via LLL+CVP
                    inner_solutions = phase2_pq(a1, c1, a2, c2, R, y)

                    for (b1, d1, b2, d2) in inner_solutions:
                        x = np.array([a1, b1, c1, d1, a2, b2, c2, d2], dtype=np.int64)
                        # Post-filter: alignment check with correct threshold
                        if float(x @ y) ** 2 >= threshold_xy:
                            solutions.append(x)
                            if max_solutions is not None and len(solutions) >= max_solutions:
                                return solutions

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

# T gate right-multiply: maps v -> v' = mat_to_uv(V · T†)
# Used to reduce odd-k (odd T-count) to even-k search.
# T† = [[1,0],[0,e^{-iπ/4}]], acting on uv = [Re(u1),Im(u1),Re(u2),Im(u2)]:
#   u1' = u1,  u2' = u2 * e^{-iπ/4} = u2 * (1-i)/√2
_r2 = float(sqrt(2.0))
_T_dag_on_uv = np.array([
    [1,  0,      0,      0    ],  # Re(u1) unchanged
    [0,  1,      0,      0    ],  # Im(u1) unchanged
    [0,  0,  1/_r2,  1/_r2   ],  # Re(u2') = (Re(u2) + Im(u2))/√2
    [0,  0, -1/_r2,  1/_r2   ],  # Im(u2') = (-Re(u2) + Im(u2))/√2
])

def synthesize(v: ndarray, k: int, eps: float = 1e-4, odd: bool = False,
               max_solutions: int = None) -> list:
    """
    Find all x = (a1,b1,c1,d1,a2,b2,c2,d2) ∈ Z^8 satisfying the norm,
    unitarity, and alignment constraints.

    v   : target uv = [Re(v1), Im(v1), Re(v2), Im(v2)], unit vector
    k   : lde parameter; T-count = 2k (even branch) or 2k+1 (odd branch)
    odd : if True, search for U s.t. U·T ≈ V  (odd T-count branch)
          Preprocesses v → v·T† before searching, so returned x satisfies
          to_unitary(x, k) · T ≈ V.
    """
    if odd:
        v = _T_dag_on_uv @ v
        norm = np.linalg.norm(v)
        if norm < 1e-12:
            return []
        v = v / norm
    y = uv_to_xy(v, k)
    return phase1_enumerate(y, k, eps, max_solutions=max_solutions)


# ---------------------------------------------------------------------------
# Verification helpers
# ---------------------------------------------------------------------------
def verify(x: ndarray, k: int, y: ndarray, eps: float = 1e-4) -> dict:
    a1, b1, c1, d1, a2, b2, c2, d2 = [int(v) for v in x]
    norm_ok = (a1**2 + b1**2 + c1**2 + d1**2 + a2**2 + b2**2 + c2**2 + d2**2 == 2**k)
    unit_ok = (b1*(a1+c1) + d1*(c1-a1) + b2*(a2+c2) + d2*(c2-a2) == 0)
    dot = float(x @ y)
    # x·y = 2^(k-1)·u·v, so |x·y|² > 2^(2k-2)·(1-eps²)
    threshold_xy = 2 ** (2*k - 2) * (1 - eps**2)
    align_ok = dot**2 >= threshold_xy
    udotv = dot / 2**(k-1)
    return {"norm": norm_ok, "unitarity": unit_ok, "alignment": align_ok,
            "dot_xy": dot, "udotv": udotv}


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(42)
    r2_ = float(sqrt(2.0))
    T   = np.array([[1,0],[0,np.exp(1j*np.pi/4)]])
    Td  = T.conj().T
    H   = np.array([[1,1],[1,-1]]) / r2_
    S   = np.array([[1,0],[0,1j]])
    I2  = np.eye(2, dtype=complex)
    X   = np.array([[0,1],[1,0]], dtype=complex)
    Y   = np.array([[0,-1j],[1j,0]], dtype=complex)
    Z   = np.array([[1,0],[0,-1]], dtype=complex)

    def find_gate(G, odd=False, k_max=3):
        """
        Find G in the synthesis lattice.

        Even branch (odd=False): try all 8 global phases of G, find the one
        that fits [[u1,-u2*],[u2,u1*]], pass that uv to synthesize.

        Odd branch (odd=True): try all 8 global phases of G·T†, same logic,
        then the final gate returned is U·T.
        """
        phases = [np.exp(1j*n*np.pi/4) for n in range(8)]
        target = G @ Td if odd else G
        for ph in phases:
            Gp = ph * target
            u1=Gp[0,0]; u2=Gp[1,0]
            if not np.allclose(Gp, [[u1,-np.conj(u2)],[u2,np.conj(u1)]]):
                continue
            v = np.array([u1.real, u1.imag, u2.real, u2.imag])
            if np.linalg.norm(v) < 1e-10: continue
            v /= np.linalg.norm(v)
            for k in range(k_max+1):
                sols = synthesize(v, k=k, eps=1.0)
                for x in sols:
                    U = to_unitary(x, k)
                    Ufinal = U @ T if odd else U
                    if np.allclose(Ufinal, ph * G if odd else Gp):
                        return k, ph
        return None, None

    '''print("Gate | Even-k branch | Odd-k branch (via G·T†)")
    print("-"*55)
    for name, G in [('I',I2),('X',X),('Y',Y),('Z',Z),
                    ('H',H),('S',S),('T',T),('THT',T@H@T)]:
        ke, _ = find_gate(G, odd=False)
        ko, _ = find_gate(G, odd=True)
        even_str = f"k={ke}" if ke is not None else "—"
        odd_str  = f"k={ko} (T-count={2*ko+1})" if ko is not None else "—"
        print(f"  {name:4s} | even: {even_str:6s} | odd: {odd_str}")'''

    np.random.seed(42)
    k = 20
    v = np.random.randn(4)
    v /= np.linalg.norm(v)
    y = uv_to_xy(v, k)

    print(f"k={k}, target v = {np.round(v, 4)}")
    print(f"||y||^2 = {y@y:.1f}  (should be 2^(k-1) = {2**(k-1)})")
    print(f"x·y = 2^(k-1)·u·v threshold for |x·y|^2: 2^(2k-2)·(1-eps^2)")
    print()

    for eps in [1e-2]:
        solutions = synthesize(v, k, eps, False, max_solutions=1)
        print(f"eps={eps}: {len(solutions)} solutions")
        for x in solutions:
            r = verify(x, k, y, eps)
            #print(f"x dot y: {float(np.dot(x, y) * 2**(-k+1))}, u dot v: {float(np.dot(xy_to_uv(x, k), v))}")
            # print(f"Solution: {x}")
            # print(f"u: {xy_to_uv(x, k)}")
            print(f"unitary:\n{to_unitary(x, k)}")
            print(f"  norm={r['norm']} unit={r['unitarity']} align={r['alignment']}  u·v={r['udotv']:.6f}  (need > {np.sqrt(1-eps**2):.6f})")
        if not solutions:
            print(f"  (k={k} is insufficient for eps={eps}; need k ~ {int(3*np.log2(1/eps))+1})")
