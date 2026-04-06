"""
LLL/CVP-Based Ancilla-Free Clifford+T Synthesis
================================================
Generated from synthesis_pseudocode.md.

Constraints on x = (a1, b1, c1, d1, a2, b2, c2, d2) ∈ ℤ⁸:
  [Norm]      ‖x‖²  = 2^(k-1)
  [Unitarity] b1(a1+c1) + d1(c1-a1) + b2(a2+c2) + d2(c2-a2) = 0
  [Alignment] (x · y)² > 2^k * (1 - ε²)

where y = sqrt(2^k) * y_hat  and  ‖y_hat‖ = 1.
"""

from __future__ import annotations
import numpy as np
from numpy import ndarray
from numpy import array, sqrt
from typing import List, Tuple

r2 = sqrt(2)
# Go from xy to uv
sigma_to_uv = array([
    [1, 1/r2, 0, -1/r2, 0, 0, 0, 0],
    [0, 1/r2, 1,  1/r2, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1/r2, 0, -1/r2],
    [0, 0, 0, 0, 0, 1/r2, 1,  1/r2],
])
# Go from uv to xy
sigma_to_xy = 1/2 * array([
    [ 1,    0,    0, 0],
    [ 1/r2, 1/r2, 0, 0],
    [ 0,    1,    0, 0],
    [-1/r2, 1/r2, 0, 0],
    [0, 0,  1,    0,  ],
    [0, 0,  1/r2, 1/r2],
    [0, 0,  0,    1,  ],
    [0, 0, -1/r2, 1/r2],
])


def uv_to_xy(uv: ndarray, k: int = 3) -> ndarray:
    scale = 1 << (k // 2) if k % 2 else 2 ** (k / 2)
    return sigma_to_xy @ uv * scale  # Try figuring out how to make shift work

def xy_to_uv(xy: ndarray, k: int = 3) -> ndarray:
    scale = 1 << (k // 2) if k % 2 else 2 ** (k / 2)
    return sigma_to_uv @ xy / scale

# =============================================================================
# Helper 1: Extended GCD
# =============================================================================

def extended_gcd(a: int, b: int) -> Tuple[int, int, int]:
    """
    Returns (g, s, t) such that g = gcd(a, b) and s*a + t*b = g.
    """
    if b == 0:
        return a, 1, 0
    g, s1, t1 = extended_gcd(b, a % b)
    return g, t1, s1 - (a // b) * t1


# =============================================================================
# Helper 2: HNF Kernel — ℤ-basis for {v ∈ ℤ⁴ : c·v = 0}
# =============================================================================

def hnf_kernel(c: ndarray) -> Tuple[ndarray, ndarray, ndarray]:
    """
    Given c ∈ ℤ⁴ (nonzero), returns three integer vectors f1, f2, f3 ∈ ℤ⁴
    that form a ℤ-basis for the kernel { v ∈ ℤ⁴ : c·v = 0 }.

    Algorithm: column HNF via successive extended-GCD elimination.
    """
    U = np.eye(4, dtype=np.int64)
    r = c.copy().astype(np.int64)

    for i in range(3):
        # Find first nonzero entry at position >= i
        j = next((jj for jj in range(i, 4) if r[jj] != 0), None)
        if j is None:
            break  # remaining entries already zero

        # Move pivot to column i
        if j != i:
            U[:, [i, j]] = U[:, [j, i]]
            r[i], r[j] = r[j], r[i]

        # Eliminate r[jj] for jj > i via unimodular column operations
        for jj in range(i + 1, 4):
            if r[jj] == 0:
                continue
            g, s, t = extended_gcd(int(r[i]), int(r[jj]))
            new_col_i = s * U[:, i] + t * U[:, jj]
            new_col_j = -(r[jj] // g) * U[:, i] + (r[i] // g) * U[:, jj]
            U[:, i]  = new_col_i
            U[:, jj] = new_col_j
            r[i]  = g
            r[jj] = 0

    # Columns 1, 2, 3 of U now span ker(c)
    return U[:, 1].copy(), U[:, 2].copy(), U[:, 3].copy()


# =============================================================================
# Helper 3: Gram-Schmidt for a 4×3 real matrix
# =============================================================================

def gram_schmidt(B: ndarray) -> Tuple[ndarray, ndarray]:
    """
    B is a (4, 3) float matrix whose columns are basis vectors.

    Returns:
      B_star : (4, 3) — Gram-Schmidt orthogonal vectors as columns
      MU     : (3, 3) — lower-triangular matrix with
                        MU[i, j] = (b_i · b_j*) / ‖b_j*‖²  for i > j
    """
    B = B.astype(float)
    n = B.shape[1]          # 3
    B_star = np.zeros_like(B)
    MU     = np.zeros((n, n))

    for i in range(n):
        B_star[:, i] = B[:, i].copy()
        for j in range(i):
            ns_j = np.dot(B_star[:, j], B_star[:, j])
            if ns_j == 0.0:
                MU[i, j] = 0.0
            else:
                MU[i, j] = np.dot(B[:, i], B_star[:, j]) / ns_j
            B_star[:, i] -= MU[i, j] * B_star[:, j]

    return B_star, MU


# =============================================================================
# Helper 4: LLL Basis Reduction (3 vectors in ℤ⁴)
# =============================================================================

def lll(f1: ndarray, f2: ndarray, f3: ndarray,
        delta: float = 0.75) -> Tuple[ndarray, ndarray, ndarray]:
    """
    LLL reduction for three integer vectors in ℤ⁴.
    Returns an LLL-reduced basis (f1', f2', f3') as integer column vectors.
    """
    B = np.column_stack([f1, f2, f3]).astype(np.int64)
    n = 3

    def _gs():
        B_star, MU = gram_schmidt(B.astype(float))
        ns = [float(np.dot(B_star[:, k], B_star[:, k])) for k in range(n)]
        return B_star, MU, ns

    B_star, MU, ns = _gs()
    k = 1  # 0-indexed; LLL works on indices 0..n-1, starts checking at k=1

    while k < n:
        # --- Size-reduce B[:,k] against all B[:,j], j < k ---
        for j in range(k - 1, -1, -1):
            q = int(np.round(MU[k, j]))
            if q != 0:
                B[:, k] -= q * B[:, j]
        # Recompute GS (only B[:,k] changed, but recompute full for simplicity)
        B_star, MU, ns = _gs()

        # --- Lovász condition ---
        if ns[k] >= (delta - MU[k, k - 1] ** 2) * ns[k - 1]:
            k += 1
        else:
            # Swap columns k-1 and k
            B[:, [k - 1, k]] = B[:, [k, k - 1]]
            B_star, MU, ns = _gs()
            k = max(k - 1, 1)

    return B[:, 0].copy(), B[:, 1].copy(), B[:, 2].copy()


# =============================================================================
# Spiral-order integer step
# =============================================================================

def _next_spiral(z: int, c: float) -> int:
    """
    Given current integer z and real center c, return the next integer to try
    in spiral order (enumerated by increasing |z - c|):
        round(c), round(c)±1, round(c)±2, ...
    """
    offset = z - round(c)
    if offset > 0:
        return z - 2 * offset        # +d → −d
    else:
        return z - 2 * offset + 1   # −d → +(d+1),  0 → +1


# =============================================================================
# Helper 5: Schnorr-Euchner Enumeration in L_c (3D lattice)
# =============================================================================

def schnorr_euchner(
    B_prime: ndarray,    # (4, 3) integer matrix — LLL-reduced columns f1', f2', f3'
    t:       ndarray,    # ℝ⁴  CVP target
    R_sq:    float,      # squared radius (enumerate ‖v‖² ≤ R_sq)
    p:       float,      # fixed partial dot product from Phase 1
    y_odd:   ndarray,    # ℝ⁴  odd components of y  (alignment half)
    threshold: float,    # alignment lower bound: (x·y)² > threshold
) -> List[ndarray]:
    """
    Enumerate all lattice vectors v = z0*f1' + z1*f2' + z2*f3' such that:
      ‖v‖²  ≤  R_sq    (norm budget)
      (p + v·y_odd)²  >  threshold   (alignment condition)

    Returns a list of ℤ⁴ vectors.

    Stack frame: (level, zk, dist_sq, center_k, fixed)
      level   : 0-indexed, outermost=2, innermost=0
      zk      : current integer coordinate at this level
      dist_sq : accumulated ‖·‖² contribution from levels > level
      center_k: real-valued center for this level's spiral
      fixed   : list [z_{level+1}, z_{level+2}, ...] (outermost appended last)
                so fixed[i] belongs to column B_prime[:, level+1+i]
    """
    B  = B_prime.astype(float)
    B_star, MU = gram_schmidt(B)
    ns = [float(np.dot(B_star[:, k], B_star[:, k])) for k in range(3)]

    # Pure projected centers (before branching adjustments)
    t_hat = np.array([
        np.dot(t, B_star[:, k]) / ns[k] if ns[k] > 0.0 else 0.0
        for k in range(3)
    ])

    y_odd_norm = float(np.linalg.norm(y_odd))
    solutions: List[ndarray] = []

    # --- Initialise stack at outermost level (index 2) ---
    center2  = t_hat[2]
    z2_init  = int(np.round(center2))
    stack    = [(2, z2_init, 0.0, center2, [])]

    while stack:
        k_lev, zk, dist_sq, centerk, fixed = stack.pop()

        contrib  = ns[k_lev] * (zk - centerk) ** 2
        new_dist = dist_sq + contrib

        if new_dist > R_sq:
            # Out of ball — try next spiral step at this level
            zk_next = _next_spiral(zk, centerk)
            if dist_sq + ns[k_lev] * (zk_next - centerk) ** 2 <= R_sq:
                stack.append((k_lev, zk_next, dist_sq, centerk, fixed))
            continue

        if k_lev == 0:
            # ---- Leaf node ----
            # fixed = [z1, z2]  (z1 at level 1, z2 at level 2)
            z0 = zk
            z1 = fixed[0] if len(fixed) > 0 else 0
            z2 = fixed[1] if len(fixed) > 1 else 0
            v  = (z0 * B_prime[:, 0]
                + z1 * B_prime[:, 1]
                + z2 * B_prime[:, 2])

            dot_total = p + float(np.dot(v, y_odd))
            if dot_total ** 2 > threshold:
                solutions.append(v.copy())

            # Continue spiral at level 0
            zk_next = _next_spiral(zk, centerk)
            if dist_sq + ns[0] * (zk_next - centerk) ** 2 <= R_sq:
                stack.append((0, zk_next, dist_sq, centerk, fixed))

        else:
            # ---- Internal node ----
            # Push next spiral step at this level
            zk_next = _next_spiral(zk, centerk)
            if dist_sq + ns[k_lev] * (zk_next - centerk) ** 2 <= R_sq:
                stack.append((k_lev, zk_next, dist_sq, centerk, fixed))

            # Build new_fixed = [zk (at k_lev)] + fixed
            new_fixed = [zk] + fixed

            # Compute center for level k_lev−1:
            #   c_{k-1} = t_hat[k-1] − Σ_{j > k-1} μ_{j, k-1} · z_j
            # new_fixed[i] is at level k_lev + i, with μ_{k_lev+i, k_lev-1}
            center_km1 = t_hat[k_lev - 1]
            for idx, zj in enumerate(new_fixed):
                actual_level = k_lev + idx  # 0-indexed level of this fixed coordinate
                # MU[i,j] defined for i > j; here actual_level > k_lev-1  ✓
                center_km1 -= MU[actual_level, k_lev - 1] * zj

            # ---- Cauchy-Schwarz pruning on alignment ----
            # Partial contribution from levels >= k_lev
            partial_v = sum(
                new_fixed[i] * B_prime[:, k_lev + i]
                for i in range(len(new_fixed))
            )
            remaining  = R_sq - new_dist            # norm budget left for lower levels
            max_add    = (remaining ** 0.5) * y_odd_norm   # C-S upper bound
            dot_partial = p + float(np.dot(partial_v, y_odd))

            # Prune if no completion (at lower levels) can push
            # (dot_partial + inner)² above threshold for any |inner| ≤ max_add
            if ((dot_partial + max_add) ** 2 <= threshold
                    and (dot_partial - max_add) ** 2 <= threshold):
                continue

            z_km1_init = int(np.round(center_km1))
            stack.append((k_lev - 1, z_km1_init, new_dist, center_km1, new_fixed))

    return solutions


# =============================================================================
# Brute-force fallback: all v ∈ ℤ⁴ with ‖v‖² = R (degenerate c_vec = 0)
# =============================================================================

def _brute_force_norm_shell(
    R: int,
    y_odd: ndarray,
    p: float,
    threshold: float,
) -> List[Tuple[int, int, int, int]]:
    """
    Enumerate all (b1, d1, b2, d2) ∈ ℤ⁴ with b1²+d1²+b2²+d2² = R
    that satisfy the alignment condition (p + v·y_odd)² > threshold.
    Used only in the degenerate case c_vec = (0,0,0,0).
    """
    results = []
    max_b1 = int(R ** 0.5)
    for b1 in range(-max_b1, max_b1 + 1):
        rem1 = R - b1 * b1
        if rem1 < 0:
            continue
        max_d1 = int(rem1 ** 0.5)
        for d1 in range(-max_d1, max_d1 + 1):
            rem2 = rem1 - d1 * d1
            if rem2 < 0:
                continue
            max_b2 = int(rem2 ** 0.5)
            for b2 in range(-max_b2, max_b2 + 1):
                rem3 = rem2 - b2 * b2
                if rem3 < 0:
                    continue
                # d2² = rem3 must be a perfect square
                d2_abs = int(round(rem3 ** 0.5))
                if d2_abs * d2_abs != rem3:
                    continue
                for d2 in ([0] if d2_abs == 0 else [d2_abs, -d2_abs]):
                    v = np.array([b1, d1, b2, d2], dtype=np.float64)
                    dot = p + float(np.dot(v, y_odd))
                    if dot ** 2 > threshold:
                        results.append((b1, d1, b2, d2))
    return results


# =============================================================================
# Phase 2: Inner Solver — given fixed (a1, c1, a2, c2), find (b1, d1, b2, d2)
# =============================================================================

def phase2(
    a1: int, c1: int, a2: int, c2: int,
    R: int,             # remaining norm = 2^(k-1) − a1²−c1²−a2²−c2²
    y: ndarray,         # ℝ⁸ full scaled target
    p: float,           # fixed partial dot product = a1*y[0]+c1*y[2]+a2*y[4]+c2*y[6]
    threshold: float,
) -> List[Tuple[int, int, int, int]]:
    """
    Returns a list of (b1, d1, b2, d2) satisfying all three constraints
    for the given fixed outer variables.
    """
    y_odd = np.array([y[1], y[3], y[5], y[7]], dtype=float)

    # Unitarity constraint vector for the inner variables:
    #   c_vec · (b1, d1, b2, d2) = 0
    c_vec = np.array([a1 + c1, c1 - a1, a2 + c2, c2 - a2], dtype=np.int64)

    # Degenerate case: unitarity trivially satisfied
    if np.all(c_vec == 0):
        return _brute_force_norm_shell(R, y_odd, p, threshold)

    # Step 1: HNF kernel → ℤ-basis for L_c = {v ∈ ℤ⁴ : c_vec · v = 0}
    f1, f2, f3 = hnf_kernel(c_vec)

    # Step 2: LLL-reduce the basis
    f1p, f2p, f3p = lll(f1, f2, f3)
    B_prime = np.column_stack([f1p, f2p, f3p])  # (4, 3) integer matrix

    # Step 3: Compute CVP target in L_c
    # Project y_odd onto span(B') and scale to radius √R
    BtB_inv = np.linalg.inv(B_prime.T.astype(float) @ B_prime.astype(float))
    pi_y    = B_prime.astype(float) @ BtB_inv @ B_prime.T.astype(float) @ y_odd
    pi_y_sq = float(np.dot(pi_y, pi_y))

    if pi_y_sq == 0.0:
        t = np.zeros(4)
    else:
        t = (R / pi_y_sq) * pi_y   # point on radius-√R sphere closest to y_odd

    # Step 4: Schnorr-Euchner enumeration
    inner_vecs = schnorr_euchner(
        B_prime   = B_prime,
        t         = t,
        R_sq      = float(R),
        p         = p,
        y_odd     = y_odd,
        threshold = threshold,
    )

    # Step 5: Filter for exact norm (‖v‖² == R)
    results = []
    for v in inner_vecs:
        v_int = np.round(v).astype(np.int64)
        if int(np.dot(v_int, v_int)) == R:
            results.append((int(v_int[0]), int(v_int[1]),
                            int(v_int[2]), int(v_int[3])))

    return results


# =============================================================================
# Phase 1: Outer Schnorr-Euchner over (a1, c1, a2, c2) in ℤ⁴
# =============================================================================

def phase1(
    y: ndarray,
    k: int,
    threshold: float,
) -> List[Tuple[int, int, int, int, int, float]]:
    """
    Enumerate all outer tuples (a1, c1, a2, c2) in ℤ⁴ with
      a1²+c1²+a2²+c2² ≤ 2^(k-1)
    that survive the Cauchy-Schwarz alignment pruning.

    Returns list of (a1, c1, a2, c2, R, p) where
      R = 2^(k-1) − a1²−c1²−a2²−c2²
      p = a1*y[0] + c1*y[2] + a2*y[4] + c2*y[6]
    """
    norm_target = 2 ** (k - 1)
    # y_even holds y at the (a1, c1, a2, c2) positions
    y_even = np.array([y[0], y[2], y[4], y[6]], dtype=float)
    y_odd  = np.array([y[1], y[3], y[5], y[7]], dtype=float)
    y_odd_norm = float(np.linalg.norm(y_odd))

    # Scale y_even to the partial-norm sphere as CVP target
    y_even_norm_sq = float(np.dot(y_even, y_even))
    if y_even_norm_sq == 0.0:
        t_even = np.zeros(4)
    else:
        scale  = (norm_target / y_even_norm_sq) ** 0.5
        t_even = scale * y_even  # ℝ⁴ target for (a1, c1, a2, c2)

    # ℤ⁴ standard basis: B_star = I, MU = 0, norms_sq = (1,1,1,1)
    # Levels 0..3 (0=a1, 1=c1, 2=a2, 3=c2); outermost = level 3 = c2

    candidates = []

    center3 = float(t_even[3])
    z3_init = int(np.round(center3))
    stack   = [(3, z3_init, 0.0, center3, [])]

    while stack:
        k_lev, zk, dist_sq, centerk, fixed = stack.pop()

        contrib  = (zk - centerk) ** 2   # ℤ⁴ is orthonormal: norms_sq[k]=1
        new_dist = dist_sq + contrib

        if new_dist > norm_target:
            zk_next = _next_spiral(zk, centerk)
            if dist_sq + (zk_next - centerk) ** 2 <= norm_target:
                stack.append((k_lev, zk_next, dist_sq, centerk, fixed))
            continue

        if k_lev == 0:
            # ---- Leaf ----
            # fixed = [z1, z2, z3] accumulated outermost-first
            a1 = zk
            c1 = fixed[0] if len(fixed) > 0 else 0
            a2 = fixed[1] if len(fixed) > 1 else 0
            c2 = fixed[2] if len(fixed) > 2 else 0
            R  = norm_target - int(new_dist)

            p = (a1 * y[0] + c1 * y[2]
               + a2 * y[4] + c2 * y[6])

            # Cauchy-Schwarz pruning: max achievable |(p + inner_dot)|
            max_inner = (R ** 0.5) * y_odd_norm
            cs_pos = (p + max_inner) ** 2
            cs_neg = (p - max_inner) ** 2
            if max(cs_pos, cs_neg) > threshold:
                candidates.append((a1, c1, a2, c2, R, float(p)))

            # Continue spiral at level 0
            zk_next = _next_spiral(zk, centerk)
            if dist_sq + (zk_next - centerk) ** 2 <= norm_target:
                stack.append((0, zk_next, dist_sq, centerk, fixed))

        else:
            # ---- Internal node ----
            zk_next = _next_spiral(zk, centerk)
            if dist_sq + (zk_next - centerk) ** 2 <= norm_target:
                stack.append((k_lev, zk_next, dist_sq, centerk, fixed))

            # Descend: center for level k_lev−1 is just t_even[k_lev-1]
            # (ℤ⁴ is orthogonal, so no μ cross-terms)
            center_km1 = float(t_even[k_lev - 1])
            new_fixed  = [zk] + fixed
            stack.append((k_lev - 1, int(np.round(center_km1)),
                          new_dist, center_km1, new_fixed))

    return candidates


# =============================================================================
# Top-Level: Full Synthesis Search
# =============================================================================

def synthesize(
    y_hat: ndarray,
    k: int,
    eps: float,
) -> List[ndarray]:
    """
    Find all x = (a1, b1, c1, d1, a2, b2, c2, d2) ∈ ℤ⁸ satisfying:
      ‖x‖²  = 2^(k-1)
      b1(a1+c1) + d1(c1-a1) + b2(a2+c2) + d2(c2-a2) = 0
      (x·y)² > 2^k * (1 − ε²)

    Args:
      y_hat : unit vector in ℝ⁸ giving target direction  (‖y_hat‖ = 1)
      k     : T-count
      eps   : approximation precision

    Returns:
      List of ℤ⁸ solutions as 1-D integer arrays.
    """
    assert np.isclose(np.linalg.norm(y_hat), 1.0), "y_hat must be a unit vector"

    norm_target = 2 ** (k - 1)
    y           = (2 ** k) ** 0.5 * y_hat          # ‖y‖² = 2^k
    threshold   = (2 ** k) * (1.0 - eps ** 2)      # alignment lower bound

    all_solutions: List[ndarray] = []

    # Phase 1 — outer enumeration with C-S pruning
    outer_candidates = phase1(y, k, threshold)

    # Phase 2 — inner CVP via LLL + Schnorr-Euchner for each outer tuple
    for (a1, c1, a2, c2, R, p) in outer_candidates:

        if R < 0:
            continue

        inner = phase2(a1, c1, a2, c2, R, y, p, threshold)

        for (b1, d1, b2, d2) in inner:
            x = np.array([a1, b1, c1, d1, a2, b2, c2, d2], dtype=np.int64)

            # Final verification (should always pass by construction)
            norm_check  = int(np.dot(x, x)) == norm_target
            unit_check  = (b1*(a1+c1) + d1*(c1-a1)
                         + b2*(a2+c2) + d2*(c2-a2)) == 0
            align_check = float(np.dot(x, y)) ** 2 > threshold

            if norm_check and unit_check and align_check:
                all_solutions.append(x)

    return all_solutions


# =============================================================================
# Quick smoke test
# =============================================================================

if __name__ == "__main__":
    import sys
    from numpy import isclose

    print("=== synthesis_cvp.py smoke test ===\n")

    # ---- Unit tests for helpers ----

    # extended_gcd
    g, s, t = extended_gcd(12, 8)
    assert g == 4 and s * 12 + t * 8 == 4, "extended_gcd failed"
    print("extended_gcd: OK")

    # hnf_kernel
    c = np.array([2, -1, 3, 0], dtype=np.int64)
    f1, f2, f3 = hnf_kernel(c)
    for fi in [f1, f2, f3]:
        assert int(np.dot(c, fi)) == 0, f"hnf_kernel: {fi} not in kernel"
    # Check they span a rank-3 lattice
    M = np.column_stack([f1, f2, f3])
    assert np.linalg.matrix_rank(M) == 3, "hnf_kernel: not rank 3"
    print("hnf_kernel: OK")

    # gram_schmidt
    B = np.column_stack([
        np.array([1.0, 0, 0, 0]),
        np.array([1.0, 1, 0, 0]),
        np.array([1.0, 1, 1, 0]),
    ])
    B_star, MU = gram_schmidt(B)
    for i in range(3):
        for j in range(i):
            assert isclose(np.dot(B_star[:, i], B_star[:, j]), 0), \
                "gram_schmidt: not orthogonal"
    print("gram_schmidt: OK")

    # lll
    # A classically bad basis for Z^3 (embedded in Z^4)
    f1b = np.array([1, 0, 0, 0], dtype=np.int64)
    f2b = np.array([1000, 1, 0, 0], dtype=np.int64)
    f3b = np.array([1000, 1000, 1, 0], dtype=np.int64)
    r1, r2, r3 = lll(f1b, f2b, f3b)
    # After LLL the first vector should be short
    assert np.dot(r1, r1) <= np.dot(f2b, f2b), "lll: didn't reduce"
    print("lll: OK")

    # ---- Full synthesize test ----
    from random import seed
    seed(42)

    def _random_unit():
        v = np.random.randn(4)
        v /= np.linalg.norm(v)
        return v

    # Build a random 4-component unit vector, embed into 8D as y_hat
    np.random.seed(42)
    uv = _random_unit()
    print(f"Random 4D unit vector (for y_hat): {uv}")
    # Map through sigma_to_xy to get an 8-component vector
    r2v = np.sqrt(2)
    sigma_to_xy = 0.5 * np.array([
        [ 1,    0,    0, 0],
        [ 1/r2v, 1/r2v, 0, 0],
        [ 0,    1,    0, 0],
        [-1/r2v, 1/r2v, 0, 0],
        [0, 0,  1,    0  ],
        [0, 0,  1/r2v, 1/r2v],
        [0, 0,  0,    1  ],
        [0, 0, -1/r2v, 1/r2v],
    ])
    y8 = sigma_to_xy @ uv
    print(f"Mapped to 8D via sigma_to_xy: {y8}")
    y_hat = y8 / np.linalg.norm(y8)

    k   = 6
    eps = 0.3

    print(f"\nRunning synthesize with k={k}, eps={eps} ...")
    solutions = synthesize(y_hat, k, eps)
    print(f"Found {len(solutions)} solution(s).")

    norm_target = 2 ** (k - 1)
    for x in solutions[:5]:
        y_full = (2 ** k) ** 0.5 * y_hat
        print(f"\nChecking solution: {x}")
        print(f"to uv: {xy_to_uv(x, k)}")
        print(f"x dot y: {float(np.dot(x, y_full) * 2**(-k))}, u dot v: {float(np.dot(xy_to_uv(x, k), uv))}")
        a1, b1, c1, d1, a2, b2, c2, d2 = x
        assert int(np.dot(x, x)) == norm_target,    f"Norm violated: {x}"
        assert (b1*(a1+c1) + d1*(c1-a1)
              + b2*(a2+c2) + d2*(c2-a2)) == 0,      f"Unitarity violated: {x}"
        assert float(np.dot(x, y_full))**2 > (2**k)*(1-eps**2), \
            f"Alignment violated: {x}"
    print("All constraints verified on returned solutions.")
    print("\nAll tests passed.")