import numpy as np
from numpy import array
from numpy import isclose
from numpy import ndarray
from numpy import sqrt
from numpy import round

from numba import njit, prange

from random import random

#===============================================================================
# Some constants
#===============================================================================
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

#===============================================================================
# Some helpful functions
#===============================================================================
def uv_to_xy(uv: ndarray, k: int = 3) -> ndarray:
    scale = 2 ** (k / 2)
    return sigma_to_xy @ uv * scale

def xy_to_uv(xy: ndarray, k: int = 3) -> ndarray:
    scale = 2 ** (k / 2)
    return sigma_to_uv @ xy / scale

def to_unit_vector(y: ndarray) -> ndarray:
    return y / sqrt((y**2).sum())

def to_unitary(x: ndarray, k) -> ndarray:
    u = xy_to_uv(x, k)
    u1 = u[0] + 1j * u[1]
    u2 = u[2] + 1j * u[3]
    utry = array([[u1, -u2.conj()], [u2, u1.conj()]])
    return utry

# def check_alignment(y: ndarray, x: ndarray, k: int = 3, eps: float = 1e-4) -> bool:
#     inner = abs((y.dot(x)).sum())
#     target = 2 ** (k - 1) * (1 - eps ** 2)
#     return inner > target

def check_norm(x: ndarray, k: int = 3) -> bool:
    mag_squared = (x.transpose() @ x).sum()
    target = 2 ** (k)
    return isclose(mag_squared, target)

def check_unitarity(x: ndarray) -> bool:
    a1, b1, c1, d1, a2, b2, c2, d2 = x
    lhs_1 = a1 * d1
    lhs_2 = a2 * d2
    rhs_1 = a1 * b1 + b1 * c1 + c1 * d1
    rhs_2 = a2 * b2 + b2 * c2 + c2 * d2
    lhs = lhs_1 + lhs_2
    rhs = rhs_1 + rhs_2
    return isclose(lhs, rhs)

def random_u3() -> ndarray:
    a, b, c, d = random(), random(), random(), random()
    norm = sqrt(a**2 + b**2 + c**2 + d**2)
    a, b, c, d = a / norm, b / norm, c / norm, d / norm
    assert isclose(a**2 + b**2 + c**2 + d**2, 1)
    return array([a, b, c, d])

#===============================================================================
# An attempt at branch and bound
#===============================================================================
@njit
def _record_if_aligned(
    a1: int, b1: int, c1: int, d1: int,
    a2: int, b2: int, c2: int, d2: int,
    align_vec: ndarray,
    align_thresh_sq: float,
    solutions: ndarray,
    sol_count: int,
    max_solutions: int,
) -> int:
    """Record solution only if it passes alignment check."""
    if align_thresh_sq > 0.0:
        dot = (a1 * align_vec[0] + b1 * align_vec[1]
             + c1 * align_vec[2] + d1 * align_vec[3]
             + a2 * align_vec[4] + b2 * align_vec[5]
             + c2 * align_vec[6] + d2 * align_vec[7])
        if dot * dot < align_thresh_sq:
            return sol_count
    if sol_count < max_solutions:
        solutions[sol_count, 0] = a1
        solutions[sol_count, 1] = b1
        solutions[sol_count, 2] = c1
        solutions[sol_count, 3] = d1
        solutions[sol_count, 4] = a2
        solutions[sol_count, 5] = b2
        solutions[sol_count, 6] = c2
        solutions[sol_count, 7] = d2
        sol_count += 1
    return sol_count


@njit
def search_phase2(
    a1: int, c1: int, a2: int, c2: int,
    remaining_norm: int,
    align_vec: ndarray,
    align_thresh_sq: float,
    solutions: ndarray,
    sol_count: int,
    max_solutions: int,
) -> int:
    """
    Given fixed (a1, c1, a2, c2), enumerate all (b1, d1, b2, d2) satisfying:
        1. b1^2 + d1^2 + b2^2 + d2^2 = remaining_norm
        2. b1(a1+c1) + d1(c1-a1) + b2(a2+c2) + d2(c2-a2) = 0

    Only records solutions passing the alignment check:
        (x . align_vec)^2 >= align_thresh_sq
    """

    # Precompute the coefficients of the linear unitarity constraint:
    # b1 * coeff_b1 + d1 * coeff_d1 + b2 * coeff_b2 + d2 * coeff_d2 = 0
    coeff_b1 = a1 + c1
    coeff_d1 = c1 - a1
    coeff_b2 = a2 + c2
    coeff_d2 = c2 - a2

    abs_b1 = abs(coeff_b1)
    abs_d1 = abs(coeff_d1)
    abs_b2 = abs(coeff_b2)
    abs_d2 = abs(coeff_d2)
    max_coeff = max(abs_b1, abs_d1, abs_b2, abs_d2)

    if max_coeff == 0:
        max_b1 = int(remaining_norm ** 0.5)
        for b1 in range(-max_b1, max_b1 + 1):
            rem1 = remaining_norm - b1 * b1
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
                    d2_abs = int(round(rem3 ** 0.5))
                    if d2_abs * d2_abs != rem3:
                        continue
                    for d2_sign in range(2):
                        if d2_sign == 0:
                            d2 = d2_abs
                        else:
                            d2 = -d2_abs
                        if d2_sign == 1 and d2_abs == 0:
                            continue
                        sol_count = _record_if_aligned(
                            a1, b1, c1, d1, a2, b2, c2, d2,
                            align_vec, align_thresh_sq,
                            solutions, sol_count, max_solutions,
                        )
        return sol_count

    if max_coeff == abs_b1:
        solve_var = 0
    elif max_coeff == abs_d1:
        solve_var = 1
    elif max_coeff == abs_b2:
        solve_var = 2
    else:
        solve_var = 3

    if solve_var == 0:
        solve_coeff = coeff_b1
        max_d1 = int(remaining_norm ** 0.5)
        for d1 in range(-max_d1, max_d1 + 1):
            rem1 = remaining_norm - d1 * d1
            if rem1 < 0:
                continue
            max_b2 = int(rem1 ** 0.5)
            for b2 in range(-max_b2, max_b2 + 1):
                rem2 = rem1 - b2 * b2
                if rem2 < 0:
                    continue
                max_d2 = int(rem2 ** 0.5)
                for d2 in range(-max_d2, max_d2 + 1):
                    rem3 = rem2 - d2 * d2
                    if rem3 < 0:
                        continue
                    numer = -(d1 * coeff_d1 + b2 * coeff_b2 + d2 * coeff_d2)
                    if numer % solve_coeff != 0:
                        continue
                    b1 = numer // solve_coeff
                    if b1 * b1 != rem3:
                        continue
                    sol_count = _record_if_aligned(
                        a1, b1, c1, d1, a2, b2, c2, d2,
                        align_vec, align_thresh_sq,
                        solutions, sol_count, max_solutions,
                    )

    elif solve_var == 1:
        solve_coeff = coeff_d1
        max_b1 = int(remaining_norm ** 0.5)
        for b1 in range(-max_b1, max_b1 + 1):
            rem1 = remaining_norm - b1 * b1
            if rem1 < 0:
                continue
            max_b2 = int(rem1 ** 0.5)
            for b2 in range(-max_b2, max_b2 + 1):
                rem2 = rem1 - b2 * b2
                if rem2 < 0:
                    continue
                max_d2 = int(rem2 ** 0.5)
                for d2 in range(-max_d2, max_d2 + 1):
                    rem3 = rem2 - d2 * d2
                    if rem3 < 0:
                        continue
                    numer = -(b1 * coeff_b1 + b2 * coeff_b2 + d2 * coeff_d2)
                    if numer % solve_coeff != 0:
                        continue
                    d1 = numer // solve_coeff
                    if d1 * d1 != rem3:
                        continue
                    sol_count = _record_if_aligned(
                        a1, b1, c1, d1, a2, b2, c2, d2,
                        align_vec, align_thresh_sq,
                        solutions, sol_count, max_solutions,
                    )

    elif solve_var == 2:
        solve_coeff = coeff_b2
        max_b1 = int(remaining_norm ** 0.5)
        for b1 in range(-max_b1, max_b1 + 1):
            rem1 = remaining_norm - b1 * b1
            if rem1 < 0:
                continue
            max_d1 = int(rem1 ** 0.5)
            for d1 in range(-max_d1, max_d1 + 1):
                rem2 = rem1 - d1 * d1
                if rem2 < 0:
                    continue
                max_d2 = int(rem2 ** 0.5)
                for d2 in range(-max_d2, max_d2 + 1):
                    rem3 = rem2 - d2 * d2
                    if rem3 < 0:
                        continue
                    numer = -(b1 * coeff_b1 + d1 * coeff_d1 + d2 * coeff_d2)
                    if numer % solve_coeff != 0:
                        continue
                    b2 = numer // solve_coeff
                    if b2 * b2 != rem3:
                        continue
                    sol_count = _record_if_aligned(
                        a1, b1, c1, d1, a2, b2, c2, d2,
                        align_vec, align_thresh_sq,
                        solutions, sol_count, max_solutions,
                    )

    else:
        solve_coeff = coeff_d2
        max_b1 = int(remaining_norm ** 0.5)
        for b1 in range(-max_b1, max_b1 + 1):
            rem1 = remaining_norm - b1 * b1
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
                    numer = -(b1 * coeff_b1 + d1 * coeff_d1 + b2 * coeff_b2)
                    if numer % solve_coeff != 0:
                        continue
                    d2 = numer // solve_coeff
                    if d2 * d2 != rem3:
                        continue
                    sol_count = _record_if_aligned(
                        a1, b1, c1, d1, a2, b2, c2, d2,
                        align_vec, align_thresh_sq,
                        solutions, sol_count, max_solutions,
                    )

    return sol_count

@njit
def search_phase1(
    target_norm: int,
    y_list: ndarray,
    align_vec: ndarray,
    align_thresh_sq: float,
    solutions: ndarray,
    max_solutions: int,
) -> int:
    """
    Enumerate all (a1, c1, a2, c2) centered around the target direction y,
    then for each combination call phase 2 to find valid (b1, d1, b2, d2).

    Uses Cauchy-Schwarz pruning: after fixing coordinates, bounds the max
    possible alignment from remaining coordinates and prunes impossible branches.
    """
    sol_count = 0
    do_prune = align_thresh_sq > 0.0
    thresh = align_thresh_sq ** 0.5

    # Precompute remaining alignment norms for Cauchy-Schwarz pruning.
    # Phase 1 fixes indices 0(a1), 2(c1), 4(a2), 6(c2).
    # After fixing a1: remaining indices are 1,2,3,4,5,6,7
    # After fixing a1,c1: remaining indices are 1,3,4,5,6,7
    # After fixing a1,c1,a2: remaining indices are 1,3,5,6,7
    # After fixing a1,c1,a2,c2: remaining indices are 1,3,5,7 (phase 2)
    av_sq_total = 0.0
    for i in range(8):
        av_sq_total += align_vec[i] * align_vec[i]
    av_sq_after_a1 = av_sq_total - align_vec[0] * align_vec[0]
    av_sq_after_c1 = av_sq_after_a1 - align_vec[2] * align_vec[2]
    av_sq_after_a2 = av_sq_after_c1 - align_vec[4] * align_vec[4]
    av_sq_after_c2 = av_sq_after_a2 - align_vec[6] * align_vec[6]

    # Compute scale factor to map y onto the sphere ||x||^2 = target_norm
    y_norm_sq = 0.0
    for i in range(8):
        y_norm_sq += y_list[i] * y_list[i]
    scale = (target_norm / y_norm_sq) ** 0.5

    a1_center = int(round(y_list[0] * scale))
    c1_center = int(round(y_list[2] * scale))
    a2_center = int(round(y_list[4] * scale))
    c2_center = int(round(y_list[6] * scale))

    max_a1 = int(target_norm ** 0.5)

    for a1_offset in range(max_a1 + abs(a1_center) + 1):
        for a1_sign in range(2 if a1_offset > 0 else 1):
            if a1_sign == 0:
                a1 = a1_center + a1_offset
            else:
                a1 = a1_center - a1_offset
            if a1 * a1 > target_norm:
                continue

            rem1 = target_norm - a1 * a1

            # Cauchy-Schwarz prune after a1
            if do_prune:
                pdot1 = a1 * align_vec[0]
                max_align = abs(pdot1) + (rem1 * av_sq_after_a1) ** 0.5
                if max_align < thresh:
                    continue

            max_c1 = int(rem1 ** 0.5)

            for c1_offset in range(max_c1 + abs(c1_center) + 1):
                for c1_sign in range(2 if c1_offset > 0 else 1):
                    if c1_sign == 0:
                        c1 = c1_center + c1_offset
                    else:
                        c1 = c1_center - c1_offset
                    if c1 * c1 > rem1:
                        continue

                    rem2 = rem1 - c1 * c1

                    # Cauchy-Schwarz prune after a1, c1
                    if do_prune:
                        pdot2 = pdot1 + c1 * align_vec[2]
                        max_align = abs(pdot2) + (rem2 * av_sq_after_c1) ** 0.5
                        if max_align < thresh:
                            continue

                    max_a2 = int(rem2 ** 0.5)

                    for a2_offset in range(max_a2 + abs(a2_center) + 1):
                        for a2_sign in range(2 if a2_offset > 0 else 1):
                            if a2_sign == 0:
                                a2 = a2_center + a2_offset
                            else:
                                a2 = a2_center - a2_offset
                            if a2 * a2 > rem2:
                                continue

                            rem3 = rem2 - a2 * a2

                            # Cauchy-Schwarz prune after a1, c1, a2
                            if do_prune:
                                pdot3 = pdot2 + a2 * align_vec[4]
                                max_align = abs(pdot3) + (rem3 * av_sq_after_a2) ** 0.5
                                if max_align < thresh:
                                    continue

                            max_c2 = int(rem3 ** 0.5)

                            for c2_offset in range(max_c2 + abs(c2_center) + 1):
                                for c2_sign in range(2 if c2_offset > 0 else 1):
                                    if c2_sign == 0:
                                        c2 = c2_center + c2_offset
                                    else:
                                        c2 = c2_center - c2_offset
                                    if c2 * c2 > rem3:
                                        continue

                                    remaining_norm = rem3 - c2 * c2

                                    # Cauchy-Schwarz prune after a1, c1, a2, c2
                                    if do_prune:
                                        pdot4 = pdot3 + c2 * align_vec[6]
                                        max_align = abs(pdot4) + (remaining_norm * av_sq_after_c2) ** 0.5
                                        if max_align < thresh:
                                            continue

                                    sol_count = search_phase2(
                                        a1, c1, a2, c2,
                                        remaining_norm,
                                        align_vec,
                                        align_thresh_sq,
                                        solutions,
                                        sol_count,
                                        max_solutions,
                                    )

                                    if sol_count >= max_solutions:
                                        return sol_count
    return sol_count


def branch_and_bound(
    y_hat: ndarray,
    k: int,
    max_solutions: int = 100000,
    align_vec: ndarray | None = None,
    epsilon: float = 0.0,
) -> ndarray:
    """
    Find integer vectors x = (a1, b1, c1, d1, a2, b2, c2, d2) satisfying:
        - ||x||^2 = 2^k
        - b1(a1+c1) + d1(c1-a1) + b2(a2+c2) + d2(c2-a2) = 0
        - (x . align_vec)^2 >= 2^k * (1 - epsilon^2)  [if align_vec given]

    Args:
        y_hat: Unit direction vector for centering search (8 components).
        k: Determines the target norm 2^k (T-count).
        max_solutions: Maximum number of solutions to collect.
        align_vec: Alignment vector Σ_uv^T @ v for filtering (8 components).
            If None, no alignment filtering is applied.
        epsilon: Precision for alignment filtering.

    Returns:
        ndarray: Array of shape (n, 8) containing the solutions found.
    """
    assert isclose(np.linalg.norm(y_hat, 2), 1), "y_hat must be a unit vector"
    target_norm = 2 ** k
    y_list = np.array([float(yi) for yi in y_hat])

    if align_vec is not None and epsilon > 0:
        # (x . align_vec)^2 >= scale^2 * (1 - epsilon^2)
        # where scale = 2^(k/2) and align_vec has norm 1
        align_thresh_sq = float(target_norm * (1.0 - epsilon ** 2))
        av = np.array([float(a) for a in align_vec])
    else:
        align_thresh_sq = 0.0
        av = np.zeros(8)

    solutions = np.zeros((max_solutions, 8), dtype=np.int64)
    sol_count = search_phase1(
        target_norm, y_list, av, align_thresh_sq,
        solutions, max_solutions,
    )
    return solutions[:sol_count]


@njit
def check_alignment(
    x: ndarray,
    y: ndarray,
    target_norm: int,
    epsilon: float = 1e-4,
) -> bool:
    """
    Check if x satisfies the alignment condition with y_hat:
        (x . y_hat) > sqrt(target_norm * (1 - epsilon^2))

    Args:
        x (ndarray): Candidate integer vector (8 components).

        y (ndarray): Transformed vector to align with (8 components).

        target_norm (int): Target squared norm (2^k).

        epsilon (float): Tolerance parameter for alignment.

    Returns:
        (bool): True if alignment condition is satisfied, False otherwise.
    """
    target_alignment_2 = (target_norm >> 1) * (1 - epsilon ** 2)
    alignment_2 = (x.dot(y)) ** 2
    return alignment_2 > target_alignment_2


#===============================================================================
# Local search: fast enumeration near a target point
#===============================================================================
@njit
def _solve_b1d1(a1, c1, a2, b2, c2, d2, R, rhs,
                align_vec, align_thresh_sq,
                solutions, sol_count, max_solutions):
    """
    Solve for (b1, d1) given all other coordinates.
    Constraints: b1² + d1² = R, b1*A + d1*B = rhs
    where A = a1+c1, B = c1-a1.
    Returns updated sol_count.
    """
    A = a1 + c1
    B = c1 - a1

    if A == 0 and B == 0:
        if rhs != 0:
            return sol_count
        # Enumerate b1, d1 with b1² + d1² = R
        max_b1 = int(R ** 0.5)
        for b1 in range(-max_b1, max_b1 + 1):
            d1_sq = R - b1 * b1
            if d1_sq < 0:
                continue
            d1_abs = int(round(d1_sq ** 0.5))
            if d1_abs * d1_abs != d1_sq:
                continue
            for d1_sign in range(2):
                d1 = d1_abs if d1_sign == 0 else -d1_abs
                if d1_sign == 1 and d1_abs == 0:
                    continue
                sol_count = _record_if_aligned(
                    a1, b1, c1, d1, a2, b2, c2, d2,
                    align_vec, align_thresh_sq,
                    solutions, sol_count, max_solutions,
                )
        return sol_count

    if A == 0:
        # d1 = rhs / B
        if rhs % B != 0:
            return sol_count
        d1 = rhs // B
        b1_sq = R - d1 * d1
        if b1_sq < 0:
            return sol_count
        b1_abs = int(round(b1_sq ** 0.5))
        if b1_abs * b1_abs != b1_sq:
            return sol_count
        for b1_sign in range(2):
            b1 = b1_abs if b1_sign == 0 else -b1_abs
            if b1_sign == 1 and b1_abs == 0:
                continue
            sol_count = _record_if_aligned(
                a1, b1, c1, d1, a2, b2, c2, d2,
                align_vec, align_thresh_sq,
                solutions, sol_count, max_solutions,
            )
        return sol_count

    if B == 0:
        # b1 = rhs / A
        if rhs % A != 0:
            return sol_count
        b1 = rhs // A
        d1_sq = R - b1 * b1
        if d1_sq < 0:
            return sol_count
        d1_abs = int(round(d1_sq ** 0.5))
        if d1_abs * d1_abs != d1_sq:
            return sol_count
        for d1_sign in range(2):
            d1 = d1_abs if d1_sign == 0 else -d1_abs
            if d1_sign == 1 and d1_abs == 0:
                continue
            sol_count = _record_if_aligned(
                a1, b1, c1, d1, a2, b2, c2, d2,
                align_vec, align_thresh_sq,
                solutions, sol_count, max_solutions,
            )
        return sol_count

    # General case: solve quadratic in d1
    # d1²·(A²+B²) - 2·rhs·B·d1 + (rhs² - A²·R) = 0
    S2 = A * A + B * B  # = 2*(a1²+c1²)
    disc = 4 * A * A * (S2 * R - rhs * rhs)
    if disc < 0:
        return sol_count

    sqrt_disc = int(round(disc ** 0.5))
    if sqrt_disc * sqrt_disc != disc:
        return sol_count

    for sign in range(2):
        numer = 2 * rhs * B + (sqrt_disc if sign == 0 else -sqrt_disc)
        if sign == 1 and sqrt_disc == 0:
            continue
        denom = 2 * S2
        if denom == 0:
            continue
        if numer % denom != 0:
            continue
        d1 = numer // denom
        # b1 = (rhs - d1*B) / A
        numer_b1 = rhs - d1 * B
        if numer_b1 % A != 0:
            continue
        b1 = numer_b1 // A
        if b1 * b1 + d1 * d1 != R:
            continue
        sol_count = _record_if_aligned(
            a1, b1, c1, d1, a2, b2, c2, d2,
            align_vec, align_thresh_sq,
            solutions, sol_count, max_solutions,
        )

    return sol_count


@njit
def search_local(
    target_norm: int,
    y_ideal: ndarray,
    window: int,
    align_vec: ndarray,
    align_thresh_sq: float,
    solutions: ndarray,
    max_solutions: int,
) -> int:
    """
    Enumerate integer solutions in a window around the ideal target point.

    For each (a1, c1, a2, c2, b2, d2) near ideal, algebraically solves
    for (b1, d1) using norm + unitarity constraints. O(1) per candidate
    instead of O(N³) brute force in phase 2.
    """
    sol_count = 0
    do_prune = align_thresh_sq > 0.0
    thresh = align_thresh_sq ** 0.5

    # Centers for all 8 coordinates
    a1_c = int(round(y_ideal[0]))
    b2_c = int(round(y_ideal[5]))
    c1_c = int(round(y_ideal[2]))
    d2_c = int(round(y_ideal[7]))
    a2_c = int(round(y_ideal[4]))
    c2_c = int(round(y_ideal[6]))

    # Precompute alignment norms for pruning
    av_sq_after_a1 = 0.0
    for i in [1, 2, 3, 4, 5, 6, 7]:
        av_sq_after_a1 += align_vec[i] * align_vec[i]
    av_sq_after_c1 = av_sq_after_a1 - align_vec[2] * align_vec[2]
    av_sq_after_a2 = av_sq_after_c1 - align_vec[4] * align_vec[4]
    av_sq_after_c2 = av_sq_after_a2 - align_vec[6] * align_vec[6]

    W = window

    for a1 in range(a1_c - W, a1_c + W + 1):
        if a1 * a1 > target_norm:
            continue
        rem1 = target_norm - a1 * a1

        if do_prune:
            pdot1 = a1 * align_vec[0]
            max_align = abs(pdot1) + (rem1 * av_sq_after_a1) ** 0.5
            if max_align < thresh:
                continue

        for c1 in range(c1_c - W, c1_c + W + 1):
            if c1 * c1 > rem1:
                continue
            rem2 = rem1 - c1 * c1

            if do_prune:
                pdot2 = pdot1 + c1 * align_vec[2]
                max_align = abs(pdot2) + (rem2 * av_sq_after_c1) ** 0.5
                if max_align < thresh:
                    continue

            for a2 in range(a2_c - W, a2_c + W + 1):
                if a2 * a2 > rem2:
                    continue
                rem3 = rem2 - a2 * a2

                if do_prune:
                    pdot3 = pdot2 + a2 * align_vec[4]
                    max_align = abs(pdot3) + (rem3 * av_sq_after_a2) ** 0.5
                    if max_align < thresh:
                        continue

                for c2 in range(c2_c - W, c2_c + W + 1):
                    if c2 * c2 > rem3:
                        continue
                    rem4 = rem3 - c2 * c2

                    if do_prune:
                        pdot4 = pdot3 + c2 * align_vec[6]
                        max_align = abs(pdot4) + (rem4 * av_sq_after_c2) ** 0.5
                        if max_align < thresh:
                            continue

                    # Now enumerate (b2, d2) near ideal, solve for (b1, d1)
                    for b2 in range(b2_c - W, b2_c + W + 1):
                        if b2 * b2 > rem4:
                            continue
                        rem5 = rem4 - b2 * b2

                        for d2 in range(d2_c - W, d2_c + W + 1):
                            if d2 * d2 > rem5:
                                continue
                            R = rem5 - d2 * d2  # = b1² + d1²

                            if R < 0:
                                continue

                            # Unitarity: b1*(a1+c1) + d1*(c1-a1)
                            #          = -(b2*(a2+c2) + d2*(c2-a2))
                            rhs = -(b2 * (a2 + c2) + d2 * (c2 - a2))

                            sol_count = _solve_b1d1(
                                a1, c1, a2, b2, c2, d2, R, rhs,
                                align_vec, align_thresh_sq,
                                solutions, sol_count, max_solutions,
                            )

                            if sol_count >= max_solutions:
                                return sol_count
    return sol_count


def local_search(
    v: ndarray,
    k: int,
    epsilon: float,
    max_solutions: int = 100000,
) -> ndarray:
    """
    Find aligned integer solutions near the target direction.

    Uses a windowed search around the ideal point, much faster than
    full-sphere enumeration for synthesis at high T-count.

    Args:
        v: Target uv parameterization (4 components, unit vector).
        k: T-count (target norm = 2^k).
        epsilon: Approximation precision.
        max_solutions: Maximum solutions to collect.

    Returns:
        ndarray: Array of shape (n, 8) containing solutions found.
    """
    target_norm = 2 ** k
    scale = 2 ** (k / 2)

    # Ideal point in x-space (non-integer)
    # Σ_xy has ||Σ_xy v|| = 1/√2 for unit v, so rescale to target sphere
    y_raw = (sigma_to_xy @ v) * scale
    y_norm_sq = np.sum(y_raw ** 2)
    rescale = np.sqrt(target_norm / y_norm_sq) if y_norm_sq > 0 else 1.0
    y_ideal = y_raw * rescale
    y_ideal_arr = np.array([float(yi) for yi in y_ideal])

    # Alignment vector and threshold
    align_vec = sigma_to_uv.T @ v
    av = np.array([float(a) for a in align_vec])
    align_thresh_sq = float(target_norm * (1.0 - epsilon ** 2))

    # Search window: scale with sqrt of target_norm and epsilon
    # At the optimal T-count, solutions are within O(epsilon * scale) of ideal
    window = max(3, int(np.ceil(epsilon * scale)) + 2)
    # Cap window to avoid excessive search at low k
    window = min(window, int(np.ceil(scale)) + 1)

    solutions = np.zeros((max_solutions, 8), dtype=np.int64)
    sol_count = search_local(
        target_norm, y_ideal_arr, window, av, align_thresh_sq,
        solutions, max_solutions,
    )
    return solutions[:sol_count]


@njit
def fast_search(
    target_norm: int,
    align_vec: ndarray,
    align_thresh_sq: float,
    solutions: ndarray,
    max_solutions: int,
) -> int:
    """
    Full-sphere enumeration with Cauchy-Schwarz pruning and algebraic solver.

    Enumerates (a1, c1, a2, c2, b2, d2) over the full sphere (no window),
    using aggressive Cauchy-Schwarz pruning to skip infeasible branches,
    then algebraically solves for (b1, d1) in O(1).
    """
    sol_count = 0
    do_prune = align_thresh_sq > 0.0
    thresh = align_thresh_sq ** 0.5

    # Precompute cumulative alignment vector norms for Cauchy-Schwarz pruning
    # Enumeration order: a1(0), c1(2), a2(4), c2(6), b2(5), d2(7)
    # then solve b1(1), d1(3)
    av_sq_all = 0.0
    for i in range(8):
        av_sq_all += align_vec[i] * align_vec[i]
    av_sq_after_a1 = av_sq_all - align_vec[0] * align_vec[0]
    av_sq_after_c1 = av_sq_after_a1 - align_vec[2] * align_vec[2]
    av_sq_after_a2 = av_sq_after_c1 - align_vec[4] * align_vec[4]
    av_sq_after_c2 = av_sq_after_a2 - align_vec[6] * align_vec[6]
    av_sq_after_b2 = av_sq_after_c2 - align_vec[5] * align_vec[5]
    av_sq_after_d2 = av_sq_after_b2 - align_vec[7] * align_vec[7]

    max_a1 = int(target_norm ** 0.5)
    for a1 in range(-max_a1, max_a1 + 1):
        rem1 = target_norm - a1 * a1
        if rem1 < 0:
            continue

        if do_prune:
            pdot1 = a1 * align_vec[0]
            if abs(pdot1) + (rem1 * av_sq_after_a1) ** 0.5 < thresh:
                continue

        max_c1 = int(rem1 ** 0.5)
        for c1 in range(-max_c1, max_c1 + 1):
            rem2 = rem1 - c1 * c1
            if rem2 < 0:
                continue

            if do_prune:
                pdot2 = pdot1 + c1 * align_vec[2]
                if abs(pdot2) + (rem2 * av_sq_after_c1) ** 0.5 < thresh:
                    continue

            max_a2 = int(rem2 ** 0.5)
            for a2 in range(-max_a2, max_a2 + 1):
                rem3 = rem2 - a2 * a2
                if rem3 < 0:
                    continue

                if do_prune:
                    pdot3 = pdot2 + a2 * align_vec[4]
                    if abs(pdot3) + (rem3 * av_sq_after_a2) ** 0.5 < thresh:
                        continue

                max_c2 = int(rem3 ** 0.5)
                for c2 in range(-max_c2, max_c2 + 1):
                    rem4 = rem3 - c2 * c2
                    if rem4 < 0:
                        continue

                    if do_prune:
                        pdot4 = pdot3 + c2 * align_vec[6]
                        if abs(pdot4) + (rem4 * av_sq_after_c2) ** 0.5 < thresh:
                            continue

                    max_b2 = int(rem4 ** 0.5)
                    for b2 in range(-max_b2, max_b2 + 1):
                        rem5 = rem4 - b2 * b2
                        if rem5 < 0:
                            continue

                        if do_prune:
                            pdot5 = pdot4 + b2 * align_vec[5]
                            if abs(pdot5) + (rem5 * av_sq_after_b2) ** 0.5 < thresh:
                                continue

                        max_d2 = int(rem5 ** 0.5)
                        for d2 in range(-max_d2, max_d2 + 1):
                            R = rem5 - d2 * d2
                            if R < 0:
                                continue

                            if do_prune:
                                pdot6 = pdot5 + d2 * align_vec[7]
                                if abs(pdot6) + (R * av_sq_after_d2) ** 0.5 < thresh:
                                    continue

                            rhs = -(b2 * (a2 + c2) + d2 * (c2 - a2))

                            sol_count = _solve_b1d1(
                                a1, c1, a2, b2, c2, d2, R, rhs,
                                align_vec, align_thresh_sq,
                                solutions, sol_count, max_solutions,
                            )

                            if sol_count >= max_solutions:
                                return sol_count
    return sol_count


@njit
def _solve_b2d2(a1, b1, c1, d1, a2, c2, R, rhs,
                align_vec, align_thresh_sq,
                solutions, sol_count, max_solutions):
    """
    Solve for (b2, d2) given all other coordinates.
    Constraints: b2² + d2² = R, b2*A + d2*B = rhs
    where A = a2+c2, B = c2-a2.
    """
    A = a2 + c2
    B = c2 - a2

    if A == 0 and B == 0:
        if rhs != 0:
            return sol_count
        max_b2 = int(R ** 0.5)
        for b2 in range(-max_b2, max_b2 + 1):
            d2_sq = R - b2 * b2
            if d2_sq < 0:
                continue
            d2_abs = int(round(d2_sq ** 0.5))
            if d2_abs * d2_abs != d2_sq:
                continue
            for d2_sign in range(2):
                d2 = d2_abs if d2_sign == 0 else -d2_abs
                if d2_sign == 1 and d2_abs == 0:
                    continue
                sol_count = _record_if_aligned(
                    a1, b1, c1, d1, a2, b2, c2, d2,
                    align_vec, align_thresh_sq,
                    solutions, sol_count, max_solutions,
                )
        return sol_count

    if A == 0:
        if rhs % B != 0:
            return sol_count
        d2 = rhs // B
        b2_sq = R - d2 * d2
        if b2_sq < 0:
            return sol_count
        b2_abs = int(round(b2_sq ** 0.5))
        if b2_abs * b2_abs != b2_sq:
            return sol_count
        for b2_sign in range(2):
            b2 = b2_abs if b2_sign == 0 else -b2_abs
            if b2_sign == 1 and b2_abs == 0:
                continue
            sol_count = _record_if_aligned(
                a1, b1, c1, d1, a2, b2, c2, d2,
                align_vec, align_thresh_sq,
                solutions, sol_count, max_solutions,
            )
        return sol_count

    if B == 0:
        if rhs % A != 0:
            return sol_count
        b2 = rhs // A
        d2_sq = R - b2 * b2
        if d2_sq < 0:
            return sol_count
        d2_abs = int(round(d2_sq ** 0.5))
        if d2_abs * d2_abs != d2_sq:
            return sol_count
        for d2_sign in range(2):
            d2 = d2_abs if d2_sign == 0 else -d2_abs
            if d2_sign == 1 and d2_abs == 0:
                continue
            sol_count = _record_if_aligned(
                a1, b1, c1, d1, a2, b2, c2, d2,
                align_vec, align_thresh_sq,
                solutions, sol_count, max_solutions,
            )
        return sol_count

    # General case: solve quadratic in d2
    S2 = A * A + B * B
    disc = 4 * A * A * (S2 * R - rhs * rhs)
    if disc < 0:
        return sol_count

    sqrt_disc = int(round(disc ** 0.5))
    if sqrt_disc * sqrt_disc != disc:
        return sol_count

    for sign in range(2):
        numer = 2 * rhs * B + (sqrt_disc if sign == 0 else -sqrt_disc)
        if sign == 1 and sqrt_disc == 0:
            continue
        denom = 2 * S2
        if denom == 0:
            continue
        if numer % denom != 0:
            continue
        d2 = numer // denom
        numer_b2 = rhs - d2 * B
        if numer_b2 % A != 0:
            continue
        b2 = numer_b2 // A
        if b2 * b2 + d2 * d2 != R:
            continue
        sol_count = _record_if_aligned(
            a1, b1, c1, d1, a2, b2, c2, d2,
            align_vec, align_thresh_sq,
            solutions, sol_count, max_solutions,
        )

    return sol_count


@njit
def fast_search_u1(
    target_norm: int,
    align_vec: ndarray,
    align_thresh_sq: float,
    solutions: ndarray,
    max_solutions: int,
) -> int:
    """
    Full-sphere enumeration optimized for targets where alignment is
    concentrated in u1 (indices 0-3).

    Enumerates (a1, b1, c1, d1, a2, c2) with Cauchy-Schwarz pruning,
    then solves for (b2, d2) algebraically.
    """
    sol_count = 0
    do_prune = align_thresh_sq > 0.0
    thresh = align_thresh_sq ** 0.5

    # Sort enumeration order by decreasing |align_vec[i]| for best pruning.
    # For u1 indices {0,1,2,3}: sort by |av[i]| descending.
    # Solve: b2(5), d2(7). Enumerate: sorted u1 indices + a2(4), c2(6).
    av_sq_all = 0.0
    for i in range(8):
        av_sq_all += align_vec[i] * align_vec[i]

    # Determine best order for u1 coords (indices 0,1,2,3)
    u1_idx = np.array([0, 1, 2, 3], dtype=np.int64)
    u1_av_abs = np.array([abs(align_vec[i]) for i in u1_idx])
    for i in range(4):
        for j in range(i + 1, 4):
            if u1_av_abs[j] > u1_av_abs[i]:
                u1_av_abs[i], u1_av_abs[j] = u1_av_abs[j], u1_av_abs[i]
                u1_idx[i], u1_idx[j] = u1_idx[j], u1_idx[i]
    i0, i1, i2, i3 = u1_idx[0], u1_idx[1], u1_idx[2], u1_idx[3]

    # Build permutation table: perm[k] tells which v-variable maps to coord k
    # i.e. coords[i0]=v0, coords[i1]=v1, coords[i2]=v2, coords[i3]=v3
    # We need a1=coords[0], b1=coords[1], c1=coords[2], d1=coords[3]
    perm = np.zeros(4, dtype=np.int64)
    perm[i0] = 0
    perm[i1] = 1
    perm[i2] = 2
    perm[i3] = 3
    # perm[k] = which enumeration variable (0-3) maps to coordinate k
    p_a1 = perm[0]  # which of v0,v1,v2,v3 is a1
    p_b1 = perm[1]  # which is b1
    p_c1 = perm[2]  # which is c1
    p_d1 = perm[3]  # which is d1

    av_sq_after_0 = av_sq_all - align_vec[i0] * align_vec[i0]
    av_sq_after_1 = av_sq_after_0 - align_vec[i1] * align_vec[i1]
    av_sq_after_2 = av_sq_after_1 - align_vec[i2] * align_vec[i2]
    av_sq_after_3 = av_sq_after_2 - align_vec[i3] * align_vec[i3]
    av_sq_after_a2 = av_sq_after_3 - align_vec[4] * align_vec[4]
    av_sq_after_c2 = av_sq_after_a2 - align_vec[6] * align_vec[6]

    max_v0 = int(target_norm ** 0.5)
    for v0 in range(-max_v0, max_v0 + 1):
        rem1 = target_norm - v0 * v0
        if rem1 < 0:
            continue
        if do_prune:
            pdot1 = v0 * align_vec[i0]
            if abs(pdot1) + (rem1 * av_sq_after_0) ** 0.5 < thresh:
                continue

        max_v1 = int(rem1 ** 0.5)
        for v1 in range(-max_v1, max_v1 + 1):
            rem2 = rem1 - v1 * v1
            if rem2 < 0:
                continue
            if do_prune:
                pdot2 = pdot1 + v1 * align_vec[i1]
                if abs(pdot2) + (rem2 * av_sq_after_1) ** 0.5 < thresh:
                    continue

            max_v2 = int(rem2 ** 0.5)
            for v2 in range(-max_v2, max_v2 + 1):
                rem3 = rem2 - v2 * v2
                if rem3 < 0:
                    continue
                if do_prune:
                    pdot3 = pdot2 + v2 * align_vec[i2]
                    if abs(pdot3) + (rem3 * av_sq_after_2) ** 0.5 < thresh:
                        continue

                max_v3 = int(rem3 ** 0.5)
                for v3 in range(-max_v3, max_v3 + 1):
                    rem4 = rem3 - v3 * v3
                    if rem4 < 0:
                        continue
                    if do_prune:
                        pdot4 = pdot3 + v3 * align_vec[i3]
                        if abs(pdot4) + (rem4 * av_sq_after_3) ** 0.5 < thresh:
                            continue

                    # Reconstruct a1,b1,c1,d1 from sorted order
                    vs = (v0, v1, v2, v3)
                    a1 = vs[p_a1]
                    b1 = vs[p_b1]
                    c1 = vs[p_c1]
                    d1 = vs[p_d1]

                    max_a2 = int(rem4 ** 0.5)
                    for a2 in range(-max_a2, max_a2 + 1):
                        rem5 = rem4 - a2 * a2
                        if rem5 < 0:
                            continue
                        if do_prune:
                            pdot5 = pdot4 + a2 * align_vec[4]
                            if abs(pdot5) + (rem5 * av_sq_after_a2) ** 0.5 < thresh:
                                continue

                        max_c2 = int(rem5 ** 0.5)
                        for c2 in range(-max_c2, max_c2 + 1):
                            R = rem5 - c2 * c2
                            if R < 0:
                                continue
                            if do_prune:
                                pdot6 = pdot5 + c2 * align_vec[6]
                                if abs(pdot6) + (R * av_sq_after_c2) ** 0.5 < thresh:
                                    continue

                            # Unitarity: b2*(a2+c2) + d2*(c2-a2)
                            #          = -(b1*(a1+c1) + d1*(c1-a1))
                            rhs = -(b1 * (a1 + c1) + d1 * (c1 - a1))

                            sol_count = _solve_b2d2(
                                a1, b1, c1, d1, a2, c2, R, rhs,
                                align_vec, align_thresh_sq,
                                solutions, sol_count, max_solutions,
                            )

                            if sol_count >= max_solutions:
                                return sol_count
    return sol_count


def aligned_search(
    v: ndarray,
    k: int,
    epsilon: float,
    max_solutions: int = 100000,
) -> ndarray:
    """
    Full-sphere aligned search with Cauchy-Schwarz pruning + algebraic solver.

    Args:
        v: Target uv parameterization (4 components, unit vector).
        k: T-count (target norm = 2^k).
        epsilon: Approximation precision.
        max_solutions: Maximum solutions to collect.

    Returns:
        ndarray: Array of shape (n, 8) containing solutions found.
    """
    target_norm = 2 ** k

    align_vec = sigma_to_uv.T @ v
    av = np.array([float(a) for a in align_vec])
    align_thresh_sq = float(target_norm * (1.0 - epsilon ** 2))

    # Choose strategy: if alignment is concentrated in u1 (indices 0-3),
    # enumerate those first for better pruning, solve u2 coords algebraically.
    u1_av_sq = sum(av[i] ** 2 for i in range(4))
    u2_av_sq = sum(av[i] ** 2 for i in range(4, 8))

    solutions = np.zeros((max_solutions, 8), dtype=np.int64)
    if u1_av_sq >= u2_av_sq:
        sol_count = fast_search_u1(
            target_norm, av, align_thresh_sq,
            solutions, max_solutions,
        )
    else:
        sol_count = fast_search(
            target_norm, av, align_thresh_sq,
            solutions, max_solutions,
        )
    return solutions[:sol_count]


#===============================================================================
# Gridsynth-style rounding for Rz synthesis
#===============================================================================
@njit
def _gridsynth_core(target_norm, target_re, target_im, eff_rem_max,
                    rem_int_max, u_range, solutions, max_solutions):
    """Numba-jitted core of gridsynth lattice search."""
    r2 = np.sqrt(2)
    sol_count = 0

    for tsign_idx in range(2):
        tsign = 1 if tsign_idx == 0 else -1
        s_re = tsign * target_re
        s_im = tsign * target_im
        for u in range(-u_range, u_range + 1):
            a_ideal = s_re - u / r2
            a_base = int(round(a_ideal))
            for da in range(-1, 2):
                a = a_base + da
                v_start = (u % 2) - u_range
                if v_start < -u_range:
                    v_start += 2
                for v in range(v_start, u_range + 1, 2):
                    c_ideal = s_im - v / r2
                    c_base = int(round(c_ideal))
                    for dc in range(-1, 2):
                        c = c_base + dc
                        b = (u + v) // 2
                        d = (v - u) // 2

                        norm_int = a*a + b*b + c*c + d*d
                        if norm_int > target_norm:
                            continue

                        cross = b*(a+c) + d*(c-a)
                        rem_int = target_norm - norm_int

                        eff_rem = rem_int - r2 * cross
                        if eff_rem > eff_rem_max * 1.5:
                            continue
                        # Cap rem_int to bound inner loop cost
                        if rem_int > rem_int_max:
                            continue

                        neg_cross = -cross
                        max_a2 = int(rem_int ** 0.5)
                        for a2 in range(-max_a2, max_a2 + 1):
                            rem2 = rem_int - a2*a2
                            if rem2 < 0:
                                continue
                            max_c2 = int(rem2 ** 0.5)
                            for c2 in range(-max_c2, max_c2 + 1):
                                R = rem2 - c2*c2
                                if R < 0:
                                    continue
                                A2 = a2 + c2
                                B2 = c2 - a2
                                rhs = neg_cross

                                if A2 == 0 and B2 == 0:
                                    if rhs != 0:
                                        continue
                                    max_b2 = int(R ** 0.5)
                                    for b2 in range(-max_b2, max_b2+1):
                                        d2_sq = R - b2*b2
                                        if d2_sq < 0:
                                            continue
                                        d2a = int(round(d2_sq ** 0.5))
                                        if d2a*d2a == d2_sq:
                                            for d2s in range(2):
                                                d2 = d2a if d2s == 0 else -d2a
                                                if d2s == 1 and d2a == 0:
                                                    continue
                                                if sol_count >= max_solutions:
                                                    return sol_count
                                                solutions[sol_count, 0] = a
                                                solutions[sol_count, 1] = b
                                                solutions[sol_count, 2] = c
                                                solutions[sol_count, 3] = d
                                                solutions[sol_count, 4] = a2
                                                solutions[sol_count, 5] = b2
                                                solutions[sol_count, 6] = c2
                                                solutions[sol_count, 7] = d2
                                                sol_count += 1
                                    continue

                                if A2 != 0 and B2 == 0:
                                    if rhs % A2 != 0:
                                        continue
                                    b2 = rhs // A2
                                    d2_sq = R - b2*b2
                                    if d2_sq < 0:
                                        continue
                                    d2a = int(round(d2_sq ** 0.5))
                                    if d2a*d2a != d2_sq:
                                        continue
                                    for d2s in range(2):
                                        d2 = d2a if d2s == 0 else -d2a
                                        if d2s == 1 and d2a == 0:
                                            continue
                                        if sol_count >= max_solutions:
                                            return sol_count
                                        solutions[sol_count, 0] = a
                                        solutions[sol_count, 1] = b
                                        solutions[sol_count, 2] = c
                                        solutions[sol_count, 3] = d
                                        solutions[sol_count, 4] = a2
                                        solutions[sol_count, 5] = b2
                                        solutions[sol_count, 6] = c2
                                        solutions[sol_count, 7] = d2
                                        sol_count += 1
                                    continue

                                if A2 == 0 and B2 != 0:
                                    if rhs % B2 != 0:
                                        continue
                                    d2 = rhs // B2
                                    b2_sq = R - d2*d2
                                    if b2_sq < 0:
                                        continue
                                    b2a = int(round(b2_sq ** 0.5))
                                    if b2a*b2a != b2_sq:
                                        continue
                                    for b2s in range(2):
                                        b2 = b2a if b2s == 0 else -b2a
                                        if b2s == 1 and b2a == 0:
                                            continue
                                        if sol_count >= max_solutions:
                                            return sol_count
                                        solutions[sol_count, 0] = a
                                        solutions[sol_count, 1] = b
                                        solutions[sol_count, 2] = c
                                        solutions[sol_count, 3] = d
                                        solutions[sol_count, 4] = a2
                                        solutions[sol_count, 5] = b2
                                        solutions[sol_count, 6] = c2
                                        solutions[sol_count, 7] = d2
                                        sol_count += 1
                                    continue

                                S2 = A2*A2 + B2*B2
                                disc = 4*A2*A2*(S2*R - rhs*rhs)
                                if disc < 0:
                                    continue
                                sqrt_d = int(round(disc ** 0.5))
                                if sqrt_d*sqrt_d != disc:
                                    continue
                                for sgn_idx in range(2):
                                    sgn = 1 if sgn_idx == 0 else -1
                                    if sgn_idx == 1 and sqrt_d == 0:
                                        continue
                                    nm = 2*rhs*B2 + sgn*sqrt_d
                                    dm = 2*S2
                                    if dm == 0 or nm % dm != 0:
                                        continue
                                    d2 = nm // dm
                                    nb = rhs - d2*B2
                                    if nb % A2 != 0:
                                        continue
                                    b2 = nb // A2
                                    if b2*b2 + d2*d2 != R:
                                        continue
                                    if sol_count >= max_solutions:
                                        return sol_count
                                    solutions[sol_count, 0] = a
                                    solutions[sol_count, 1] = b
                                    solutions[sol_count, 2] = c
                                    solutions[sol_count, 3] = d
                                    solutions[sol_count, 4] = a2
                                    solutions[sol_count, 5] = b2
                                    solutions[sol_count, 6] = c2
                                    solutions[sol_count, 7] = d2
                                    sol_count += 1

                        if sol_count >= max_solutions:
                            return sol_count

    return sol_count


def gridsynth_candidates(theta: float, t: int, epsilon: float,
                         n_candidates: int = 200):
    """
    Generate candidate DOmega approximations to e^{-i*theta/2} at exponent t.

    Uses lattice rounding in Z[omega] with parameterization u=b-d, v=b+d,
    then searches nearby integer points for valid unitaries.

    Returns list of valid x = (a1,b1,c1,d1,a2,b2,c2,d2) solutions.
    """
    target_norm = 2 ** t
    scale = 2 ** (t / 2)

    eff_rem_max = target_norm * epsilon * epsilon

    target_re = scale * np.cos(theta / 2)
    target_im = -scale * np.sin(theta / 2)

    # For small t, lattice is sparse — need wider range to cover sphere.
    # For large t, smaller range keeps inner loops bounded.
    # u_range controls both candidate diversity and inner loop cost
    # (larger |u,v| → larger cross → larger rem_int → bigger inner loop).
    if t <= 14:
        max_comp = int(np.sqrt(target_norm))
        u_range = min(2 * max_comp, 100)
    elif t <= 22:
        u_range = 50
    else:
        u_range = 20

    # Cap rem_int to bound inner (a2,c2) loop cost.
    # rem_int < 2^t*(ε² + c*u_range/2^(t/2)) where c accounts for cross terms.
    # This is generous enough not to lose solutions but prevents explosion.
    rem_int_max = int(target_norm * (epsilon**2 + 8.0 * u_range / scale)) + 1000

    solutions = np.zeros((n_candidates, 8), dtype=np.int64)
    sol_count = _gridsynth_core(target_norm, target_re, target_im,
                                eff_rem_max, rem_int_max, u_range,
                                solutions, n_candidates)
    return [solutions[i] for i in range(sol_count)]


@njit
def _general_gridsynth_core(target_norm, u1_re, u1_im, u2_re, u2_im,
                            u1_range, u2_window, solutions, max_solutions):
    """
    Numba-jitted core for general unitary gridsynth.

    Rounds both u1 and u2 to Z[ω] lattice, then solves (b2,d2) algebraically
    from norm + unitarity constraints.
    """
    r2 = np.sqrt(2)
    sol_count = 0

    for tsign_idx in range(2):
        tsign = 1 if tsign_idx == 0 else -1
        # Sign ambiguity: both x and -x give same diamond distance
        s1_re = tsign * u1_re
        s1_im = tsign * u1_im
        s2_re = tsign * u2_re
        s2_im = tsign * u2_im

        for u1u in range(-u1_range, u1_range + 1):
            a1_ideal = s1_re - u1u / r2
            a1_base = int(round(a1_ideal))
            for da1 in range(-1, 2):
                a1 = a1_base + da1
                v1_start = (u1u % 2) - u1_range
                if v1_start < -u1_range:
                    v1_start += 2
                for u1v in range(v1_start, u1_range + 1, 2):
                    c1_ideal = s1_im - u1v / r2
                    c1_base = int(round(c1_ideal))
                    for dc1 in range(-1, 2):
                        c1 = c1_base + dc1
                        b1 = (u1u + u1v) // 2
                        d1 = (u1v - u1u) // 2

                        norm1 = a1*a1 + b1*b1 + c1*c1 + d1*d1
                        if norm1 > target_norm:
                            continue
                        cross1 = b1*(a1+c1) + d1*(c1-a1)
                        rem_norm = target_norm - norm1
                        neg_cross1 = -cross1

                        # Now search u2 near the rounded target
                        for u2u in range(-u2_window, u2_window + 1):
                            a2_ideal = s2_re - u2u / r2
                            a2_base = int(round(a2_ideal))
                            for da2 in range(-1, 2):
                                a2 = a2_base + da2
                                v2_start = (u2u % 2) - u2_window
                                if v2_start < -u2_window:
                                    v2_start += 2
                                for u2v in range(v2_start, u2_window + 1, 2):
                                    c2_ideal = s2_im - u2v / r2
                                    c2_base = int(round(c2_ideal))
                                    for dc2 in range(-1, 2):
                                        c2 = c2_base + dc2

                                        # Remaining norm for b2² + d2²
                                        R = rem_norm - a2*a2 - c2*c2
                                        if R < 0:
                                            continue
                                        # Cross constraint: b2*(a2+c2)+d2*(c2-a2)=neg_cross1
                                        A2 = a2 + c2
                                        B2 = c2 - a2
                                        rhs = neg_cross1

                                        if A2 == 0 and B2 == 0:
                                            if rhs != 0:
                                                continue
                                            max_b2 = int(R ** 0.5)
                                            for b2 in range(-max_b2, max_b2+1):
                                                d2_sq = R - b2*b2
                                                if d2_sq < 0:
                                                    continue
                                                d2a = int(round(d2_sq ** 0.5))
                                                if d2a*d2a == d2_sq:
                                                    for d2s in range(2):
                                                        d2 = d2a if d2s == 0 else -d2a
                                                        if d2s == 1 and d2a == 0:
                                                            continue
                                                        if sol_count >= max_solutions:
                                                            return sol_count
                                                        solutions[sol_count, 0] = a1
                                                        solutions[sol_count, 1] = b1
                                                        solutions[sol_count, 2] = c1
                                                        solutions[sol_count, 3] = d1
                                                        solutions[sol_count, 4] = a2
                                                        solutions[sol_count, 5] = b2
                                                        solutions[sol_count, 6] = c2
                                                        solutions[sol_count, 7] = d2
                                                        sol_count += 1
                                            continue

                                        if A2 != 0 and B2 == 0:
                                            if rhs % A2 != 0:
                                                continue
                                            b2 = rhs // A2
                                            d2_sq = R - b2*b2
                                            if d2_sq < 0:
                                                continue
                                            d2a = int(round(d2_sq ** 0.5))
                                            if d2a*d2a != d2_sq:
                                                continue
                                            for d2s in range(2):
                                                d2 = d2a if d2s == 0 else -d2a
                                                if d2s == 1 and d2a == 0:
                                                    continue
                                                if sol_count >= max_solutions:
                                                    return sol_count
                                                solutions[sol_count, 0] = a1
                                                solutions[sol_count, 1] = b1
                                                solutions[sol_count, 2] = c1
                                                solutions[sol_count, 3] = d1
                                                solutions[sol_count, 4] = a2
                                                solutions[sol_count, 5] = b2
                                                solutions[sol_count, 6] = c2
                                                solutions[sol_count, 7] = d2
                                                sol_count += 1
                                            continue

                                        if A2 == 0 and B2 != 0:
                                            if rhs % B2 != 0:
                                                continue
                                            d2 = rhs // B2
                                            b2_sq = R - d2*d2
                                            if b2_sq < 0:
                                                continue
                                            b2a = int(round(b2_sq ** 0.5))
                                            if b2a*b2a != b2_sq:
                                                continue
                                            for b2s in range(2):
                                                b2 = b2a if b2s == 0 else -b2a
                                                if b2s == 1 and b2a == 0:
                                                    continue
                                                if sol_count >= max_solutions:
                                                    return sol_count
                                                solutions[sol_count, 0] = a1
                                                solutions[sol_count, 1] = b1
                                                solutions[sol_count, 2] = c1
                                                solutions[sol_count, 3] = d1
                                                solutions[sol_count, 4] = a2
                                                solutions[sol_count, 5] = b2
                                                solutions[sol_count, 6] = c2
                                                solutions[sol_count, 7] = d2
                                                sol_count += 1
                                            continue

                                        S2 = A2*A2 + B2*B2
                                        disc = 4*A2*A2*(S2*R - rhs*rhs)
                                        if disc < 0:
                                            continue
                                        sqrt_d = int(round(disc ** 0.5))
                                        if sqrt_d*sqrt_d != disc:
                                            continue
                                        for sgn_idx in range(2):
                                            sgn = 1 if sgn_idx == 0 else -1
                                            if sgn_idx == 1 and sqrt_d == 0:
                                                continue
                                            nm = 2*rhs*B2 + sgn*sqrt_d
                                            dm = 2*S2
                                            if dm == 0 or nm % dm != 0:
                                                continue
                                            d2 = nm // dm
                                            nb = rhs - d2*B2
                                            if nb % A2 != 0:
                                                continue
                                            b2 = nb // A2
                                            if b2*b2 + d2*d2 != R:
                                                continue
                                            if sol_count >= max_solutions:
                                                return sol_count
                                            solutions[sol_count, 0] = a1
                                            solutions[sol_count, 1] = b1
                                            solutions[sol_count, 2] = c1
                                            solutions[sol_count, 3] = d1
                                            solutions[sol_count, 4] = a2
                                            solutions[sol_count, 5] = b2
                                            solutions[sol_count, 6] = c2
                                            solutions[sol_count, 7] = d2
                                            sol_count += 1

                        if sol_count >= max_solutions:
                            return sol_count

    return sol_count


def general_gridsynth_candidates(v: ndarray, t: int, epsilon: float,
                                 n_candidates: int = 200):
    """
    Gridsynth for general (non-Rz) unitaries at exponent t.

    v = (Re(u1), Im(u1), Re(u2), Im(u2)) is the target uv vector.
    Rounds both u1 and u2 to Z[ω] lattice, solves constraints algebraically.
    """
    target_norm = 2 ** t
    scale = 2 ** (t / 2)

    u1_re = scale * v[0]
    u1_im = scale * v[1]
    u2_re = scale * v[2]
    u2_im = scale * v[3]

    # u1 range: for small t, cover full sphere; for large t, smaller
    if t <= 14:
        max_comp = int(np.sqrt(target_norm))
        u1_range = min(2 * max_comp, 100)
    elif t <= 22:
        u1_range = 50
    else:
        u1_range = 20

    # u2 search window: neighborhood around rounded target.
    # Wider window finds better candidates. For small t, lattice is
    # sparse so need wider range.
    if t <= 14:
        max_comp = int(np.sqrt(target_norm))
        u2_window = min(2 * max_comp, 50)
    elif t <= 22:
        u2_window = 20
    else:
        u2_window = 10

    solutions = np.zeros((n_candidates, 8), dtype=np.int64)
    sol_count = _general_gridsynth_core(
        target_norm, u1_re, u1_im, u2_re, u2_im,
        u1_range, u2_window, solutions, n_candidates)
    return [solutions[i] for i in range(sol_count)]


#===============================================================================
# Meet-in-the-middle synthesis
#===============================================================================
def enumerate_all(k: int) -> ndarray:
    """
    Enumerate ALL valid unitaries at T-count k (no alignment filtering).

    Returns all x with ||x||² = 2^k and unitarity constraint satisfied.
    """
    target_norm = 2 ** k
    av = np.zeros(8)  # no alignment filtering
    solutions = np.zeros((1000000, 8), dtype=np.int64)
    sol_count = fast_search(target_norm, av, 0.0, solutions, 1000000)
    return solutions[:sol_count]


@njit
def domega_mul(a: int, b: int, c: int, d: int,
               e: int, f: int, g: int, h: int):
    """
    Multiply two DOmega elements: (a+bω+cω²+dω³)(e+fω+gω²+hω³).
    Returns (p, q, r, s) where result = p+qω+rω²+sω³.
    """
    p = a*e - b*h - c*g - d*f
    q = a*f + b*e - c*h - d*g
    r = a*g + b*f + c*e - d*h
    s = a*h + b*g + c*f + d*e
    return p, q, r, s


@njit
def domega_conj(a: int, b: int, c: int, d: int):
    """Conjugate of DOmega element: conj(a+bω+cω²+dω³) = a-dω-cω²-bω³."""
    return a, -d, -c, -b


def combine_unitaries(x_L: ndarray, x_R: ndarray) -> ndarray:
    """
    Combine two x-space solutions: U = U_L · U_R in SU(2).

    U_L = [[u1_L, -conj(u2_L)], [u2_L, conj(u1_L)]]
    U_R = [[u1_R, -conj(u2_R)], [u2_R, conj(u1_R)]]

    Result u1 = u1_L*u1_R - conj(u2_L)*u2_R
           u2 = u2_L*u1_R + conj(u1_L)*u2_R
    """
    a1L, b1L, c1L, d1L = int(x_L[0]), int(x_L[1]), int(x_L[2]), int(x_L[3])
    a2L, b2L, c2L, d2L = int(x_L[4]), int(x_L[5]), int(x_L[6]), int(x_L[7])
    a1R, b1R, c1R, d1R = int(x_R[0]), int(x_R[1]), int(x_R[2]), int(x_R[3])
    a2R, b2R, c2R, d2R = int(x_R[4]), int(x_R[5]), int(x_R[6]), int(x_R[7])

    # u1 = u1_L * u1_R - conj(u2_L) * u2_R
    p1 = domega_mul(a1L, b1L, c1L, d1L, a1R, b1R, c1R, d1R)
    ca2L, cb2L, cc2L, cd2L = domega_conj(a2L, b2L, c2L, d2L)
    p2 = domega_mul(ca2L, cb2L, cc2L, cd2L, a2R, b2R, c2R, d2R)
    u1 = (p1[0]-p2[0], p1[1]-p2[1], p1[2]-p2[2], p1[3]-p2[3])

    # u2 = u2_L * u1_R + conj(u1_L) * u2_R
    p3 = domega_mul(a2L, b2L, c2L, d2L, a1R, b1R, c1R, d1R)
    ca1L, cb1L, cc1L, cd1L = domega_conj(a1L, b1L, c1L, d1L)
    p4 = domega_mul(ca1L, cb1L, cc1L, cd1L, a2R, b2R, c2R, d2R)
    u2 = (p3[0]+p4[0], p3[1]+p4[1], p3[2]+p4[2], p3[3]+p4[3])

    return np.array([u1[0], u1[1], u1[2], u1[3],
                     u2[0], u2[1], u2[2], u2[3]], dtype=np.int64)


def meet_in_middle(
    V: ndarray,
    t: int,
    epsilon: float,
    t_split: int = 4,
    max_solutions: int = 100,
    verbose: bool = False,
) -> list[dict]:
    """
    Meet-in-the-middle search for T-count t.

    Splits t = t_L + t_R, enumerates all valid unitaries at t_L,
    then for each, searches for aligned solutions at t_R for the
    residual target.

    Args:
        V: Target unitary.
        t: Total T-count.
        epsilon: Approximation precision.
        t_split: T-count for the left half (enumerate all).
        max_solutions: Max solutions to return.
        verbose: Print progress.

    Returns:
        List of dicts with 'solution' (combined x), 't_count', 'distance'.
    """
    t_L = min(t_split, t)
    t_R = t - t_L

    # Step 1: enumerate all valid unitaries at t_L
    left_solutions = enumerate_all(t_L)
    if verbose:
        print(f"  MITM: {len(left_solutions)} left solutions at t_L={t_L}")

    results = []

    # Step 2: for each left solution, search for right solution
    for x_L in left_solutions:
        U_L = to_unitary(x_L, t_L)
        # Residual target: U_R such that U_L @ U_R ≈ V, i.e., U_R ≈ U_L† @ V
        V_R = U_L.conj().T @ V
        v_R = unitary_to_uv(V_R)

        # Search at t_R for solutions aligned with V_R
        right_solutions = aligned_search(v_R, t_R, epsilon, max_solutions=10)

        for x_R in right_solutions:
            # Combine: U = U_L · U_R
            x_combined = combine_unitaries(x_L, x_R)
            t_combined = t_L + t_R
            U_combined = to_unitary(x_combined, t_combined)
            dist = diamond_distance(U_combined, V)

            if dist < epsilon:
                results.append({
                    'solution': x_combined,
                    't_count': t_combined,
                    'distance': dist,
                })
                if len(results) >= max_solutions:
                    return results

    return results


#===============================================================================
# Synthesis functions (Algorithm 3.14 from arXiv:2510.05816)
#===============================================================================
def diamond_distance(U: ndarray, V: ndarray) -> float:
    """
    Diamond distance between two single-qubit unitaries (Proposition 2.1).

    d_diamond(U, V) = sqrt(1 - |tr(UV†)|² / 4)
    """
    tr = np.trace(U @ V.conj().T)
    return np.sqrt(max(0.0, 1.0 - abs(tr) ** 2 / 4.0))


def unitary_to_uv(V: ndarray) -> ndarray:
    """
    Extract uv parameterization from a 2x2 SU(2) unitary.

    Returns (Re(u1), Im(u1), Re(u2), Im(u2)) where
    V = [[u1, -conj(u2)], [u2, conj(u1)]].
    """
    u1 = V[0, 0]
    u2 = V[1, 0]
    return array([u1.real, u1.imag, u2.real, u2.imag])


def rz_unitary(theta: float) -> ndarray:
    """Construct Rz(theta) = diag(e^{-i*theta/2}, e^{i*theta/2})."""
    return array([
        [np.exp(-1j * theta / 2), 0],
        [0, np.exp(1j * theta / 2)],
    ])


def domega_to_dcn(values: list[int], k: int):
    """
    Convert DOmega coefficients (a + bω + cω² + dω³)/(√2)^k
    to a DyadicComplexNumber (numerator / 2^denom_exp).

    For even k: straightforward, denom_exp = k // 2.
    For odd k: multiply numerator by √2 = (ω - ω³), denom_exp = (k+1) // 2.
    """
    from cyclosynth.algebra import DyadicComplexNumber
    a, b, c, d = values
    if k % 2 == 0:
        return DyadicComplexNumber([a, b, c, d], k // 2)
    else:
        new = [b - d, a + c, b + d, c - a]
        return DyadicComplexNumber(new, (k + 1) // 2)


def solution_to_u2matrix(x: ndarray, k: int):
    """
    Convert an integer solution to an exact U2Matrix.

    Given x = (a1, b1, c1, d1, a2, b2, c2, d2), builds the unitary
    U = [[u1, -conj(u2)], [u2, conj(u1)]] / (√2)^k
    where u_j = a_j + b_j*ω + c_j*ω² + d_j*ω³.
    """
    from cyclosynth.matrix import U2Matrix
    a1, b1, c1, d1 = int(x[0]), int(x[1]), int(x[2]), int(x[3])
    a2, b2, c2, d2 = int(x[4]), int(x[5]), int(x[6]), int(x[7])

    u1 = domega_to_dcn([a1, b1, c1, d1], k)
    u2 = domega_to_dcn([a2, b2, c2, d2], k)
    # conj(u_j) has DOmega coefficients [a, -d, -c, -b]
    conj_u1 = domega_to_dcn([a1, -d1, -c1, -b1], k)
    neg_conj_u2 = domega_to_dcn([-a2, d2, c2, b2], k)

    return U2Matrix([u1, neg_conj_u2, u2, conj_u1])


def solution_to_gates(x: ndarray, k: int) -> str:
    """
    Convert an integer solution to a Clifford+T gate sequence.

    Uses BlochDecomposer to decompose the exact unitary into
    discrete rotations, then translates to Clifford+T gates.
    """
    from cyclosynth.bloch import BlochDecomposer
    u2 = solution_to_u2matrix(x, k)
    decomposer = BlochDecomposer(u2)
    return decomposer.decompose()


def _generate_left_prefixes(t_prime: int) -> list[ndarray]:
    """
    Generate all left prefixes L_{t'} from Lemma 3.10.

    L_n = {∏_{i=1}^n HS^{b_i}T | b_i∈{0,1}} ∪ {T·∏_{i=1}^{n-1} HS^{b_i}T}

    Returns list of 2x2 unitary matrices.
    """
    H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    S = np.array([[1, 0], [0, 1j]])
    T_gate = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]])
    HS0T = H @ T_gate          # HS^0 T = HT
    HS1T = H @ S @ T_gate      # HS^1 T = HST

    if t_prime == 0:
        return [np.eye(2, dtype=complex)]

    # First family: ∏_{i=1}^{t'} HS^{b_i}T for all b∈{0,1}^{t'}
    prefixes = []
    for bits in range(2 ** t_prime):
        U = np.eye(2, dtype=complex)
        for i in range(t_prime):
            U = U @ (HS1T if ((bits >> i) & 1) else HS0T)
        prefixes.append(U)

    # Second family: T · ∏_{i=1}^{t'-1} HS^{b_i}T for all b∈{0,1}^{t'-1}
    for bits in range(2 ** max(0, t_prime - 1)):
        U = T_gate.copy()
        for i in range(t_prime - 1):
            U = U @ (HS1T if ((bits >> i) & 1) else HS0T)
        prefixes.append(U)

    return prefixes


def _enumerate_at_t(V: ndarray, v: ndarray, t: int, epsilon: float,
                    max_solutions: int, direct_limit: int) -> list:
    """
    Algorithm 3.11: Enumerate Clifford+T unitaries at lde t near V.

    Uses Algorithm 3.6 (direct enumeration via aligned_search) for
    t <= direct_limit. For t > direct_limit, uses divide-and-conquer
    with Matsumoto-Amano left prefixes (Lemma 3.10).

    The parameter t is the lde (denominator exponent), NOT the T-count.
    The actual T-count is approximately 2*t - 2 (even T-count, det=1)
    for the SU(2) representation used by aligned_search.

    For d&c with prefix T-count t_prime:
    - Residual T-count = (2t - 2) - t_prime
    - Residual lde = t - floor(t_prime / 2) (from paper's Eqs 3.2-3.4)
    - If t_prime is odd, residual has odd T-count and needs zeta-bar
      target adjustment (paper's Algorithm 3.6 step (a) for odd case)
    """
    zeta = np.exp(1j * np.pi / 4)

    if t <= direct_limit:
        # Algorithm 3.6: direct integer-point enumeration
        return list(aligned_search(v, t, epsilon, max_solutions))

    # Algorithm 3.11: divide-and-conquer with MA left prefixes.
    # Choose t_prime (number of prefix T-gates) so residual lde fits
    # within direct_limit.
    # Residual lde = t - floor(t_prime/2), so we need:
    #   t - floor(t_prime/2) <= direct_limit
    #   floor(t_prime/2) >= t - direct_limit
    #   t_prime >= 2*(t - direct_limit)
    t_prime = 2 * (t - direct_limit)

    prefixes = _generate_left_prefixes(t_prime)
    results = []

    for U_L in prefixes:
        V_R = U_L.conj().T @ V
        k_residual = t - t_prime // 2
        odd_residual = (t_prime % 2 == 1)

        if odd_residual:
            # Odd residual T-count: adjust target by zeta-bar
            # (paper's Algorithm 3.6 step (a) for odd t)
            u1_R = V_R[0, 0]
            u2_R = V_R[1, 0]
            u1_adj = np.conj(zeta) * u1_R
            u2_adj = np.conj(zeta) * u2_R
            V_adj = np.array([[u1_adj, -np.conj(u2_adj)],
                              [u2_adj, np.conj(u1_adj)]])
            v_R = unitary_to_uv(V_adj)
        else:
            v_R = unitary_to_uv(V_R)

        right_sols = aligned_search(v_R, k_residual, epsilon, max_solutions)

        for sol in right_sols:
            if odd_residual:
                # Odd-t reconstruction (paper's Eq 3.3 / step (d)):
                # U_R = [[u1, -u2†*zeta], [u2, u1†*zeta]]
                uv = xy_to_uv(sol, k_residual)
                u1_s = uv[0] + 1j * uv[1]
                u2_s = uv[2] + 1j * uv[3]
                U_R = np.array([[u1_s, -np.conj(u2_s) * zeta],
                                [u2_s, np.conj(u1_s) * zeta]])
            else:
                U_R = to_unitary(sol, k_residual)

            U = U_L @ U_R
            dist = diamond_distance(U, V)
            if dist < epsilon:
                results.append({
                    '_dc': True,
                    'sol_R': sol,
                    't_R': k_residual,
                    'U_L': U_L,
                    'U': U,
                    'distance': dist,
                    'odd_residual': odd_residual,
                })
                if len(results) >= max_solutions:
                    return results

    return results


def enumerate_with_dc(V: ndarray, k: int, epsilon: float,
                      max_solutions: int = 100,
                      direct_limit: int = 12) -> list:
    """
    Algorithm 3.11: Enumerate with divide-and-conquer using MA left prefixes.

    Uses the corrected lde-based splitting: prefix T-count t_prime gives
    residual lde = k - floor(t_prime/2). For odd t_prime, uses zeta-bar
    target adjustment per paper's Algorithm 3.6 step (a).

    Args:
        V: Target 2x2 unitary.
        k: lde (denominator exponent) to search at.
        epsilon: Approximation precision.
        max_solutions: Max solutions to return.
        direct_limit: Max lde for direct aligned_search.

    Returns list of dicts with keys: sol_R, t_R, U_L, U, distance, odd_residual.
    """
    v = unitary_to_uv(V)
    return _enumerate_at_t(V, v, k, epsilon, max_solutions, direct_limit)


def synthesize(
    V: ndarray,
    epsilon: float,
    max_t: int = 50,
    max_solutions: int = 100000,
    verbose: bool = True,
) -> dict | None:
    """
    Deterministic Clifford+T synthesis (Algorithm 3.14).

    Finds minimum-lde Clifford+T circuit U such that
    d_diamond(U, V) < epsilon.

    Uses Algorithm 3.6 (direct enumeration) for small lde and
    Algorithm 3.11 (divide-and-conquer with MA left prefixes) for
    larger lde, following arXiv:2510.05816.

    The search parameter is the lde (least denominator exponent),
    not the T-count directly. T-count ≈ 2*lde for the SU(2)
    representation used here.

    Args:
        V: Target 2x2 unitary matrix (numpy array).
        epsilon: Approximation precision in diamond distance.
        max_t: Maximum lde to try.
        max_solutions: Max integer solutions to enumerate per level.
        verbose: Print progress.

    Returns:
        Dict with keys: solution, unitary, lde, distance, gates.
        None if no solution found within max_t.
    """
    import time as _time
    v = unitary_to_uv(V)

    # Direct enumeration (Algorithm 3.6) is feasible up to this lde.
    # Beyond this, divide-and-conquer (Algorithm 3.11) splits the problem.
    # Note: the parameter t below is the lde (denominator exponent), not
    # the T-count. T-count ≈ 2*lde for the SU(2) representation.
    direct_limit = 12

    for t in range(max_t):
        _t0 = _time.time()

        candidates = _enumerate_at_t(V, v, t, epsilon, max_solutions,
                                     direct_limit)

        _elapsed = _time.time() - _t0

        if t <= direct_limit:
            # Direct results: list of 8D integer solutions
            method = "direct"
            best_dist = float('inf')
            for sol in candidates:
                U = to_unitary(sol, t)
                dist = diamond_distance(U, V)
                if dist < best_dist:
                    best_dist = dist
                if dist < epsilon:
                    result = {
                        'solution': sol,
                        'unitary': U,
                        'lde': t,
                        'distance': dist,
                    }
                    try:
                        result['gates'] = solution_to_gates(sol, t)
                    except Exception as e:
                        result['gates'] = None
                        result['gate_error'] = str(e)
                    if verbose:
                        print(f"  t={t}: {method} {_elapsed:.2f}s"
                              f" => FOUND d={dist:.6e}")
                    return result
        else:
            # D&C results: list of dicts with combined unitary info
            t_prime = 2 * (t - direct_limit)
            n_pfx = len(_generate_left_prefixes(t_prime))
            method = f"d&c(t'={t_prime},{n_pfx}pfx)"
            best_dist = float('inf')
            for item in candidates:
                dist = item['distance']
                if dist < best_dist:
                    best_dist = dist
                if dist < epsilon:
                    result = {
                        'unitary': item['U'],
                        'lde': t,
                        'distance': dist,
                    }
                    # Gate extraction for the residual part
                    try:
                        result['gates'] = solution_to_gates(
                            item['sol_R'], item['t_R'])
                    except Exception as e:
                        result['gates'] = None
                        result['gate_error'] = str(e)
                    if verbose:
                        print(f"  t={t}: {method} {_elapsed:.2f}s"
                              f" => FOUND d={dist:.6e}")
                    return result

        if verbose:
            if len(candidates) > 0:
                print(f"  t={t}: {method} {_elapsed:.2f}s"
                      f", {len(candidates)} cands"
                      f", best={best_dist:.6e}")
            else:
                print(f"  t={t}: {method} {_elapsed:.2f}s")

    return None


if __name__ == "__main__":
    import argparse

    # Common options shared by all subcommands
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("-e", "--epsilon", type=float, default=1e-2,
                        help="Diamond distance precision (default: 0.01)")
    common.add_argument("--max-lde", type=int, default=50,
                        help="Maximum lde to search (default: 50)")
    common.add_argument("--direct-limit", type=int, default=12,
                        help="Max lde for direct search before d&c (default: 12)")
    common.add_argument("-q", "--quiet", action="store_true",
                        help="Suppress per-level progress output")

    parser = argparse.ArgumentParser(
        description="Clifford+T synthesis (arXiv:2510.05816)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  %(prog)s rx 0.3
  %(prog)s ry 0.7 -e 0.001
  %(prog)s rz pi/7
  %(prog)s u3 0.1 0.2 0.3
  %(prog)s random -e 0.05 --max-lde 20
""")
    sub = parser.add_subparsers(dest="gate", required=True)

    # Rx, Ry, Rz each take one angle
    for name in ("rx", "ry", "rz"):
        p = sub.add_parser(name, parents=[common],
                           help=f"{name.upper()}(angle)")
        p.add_argument("angle", type=str,
                       help="Rotation angle (supports pi, e.g. pi/7)")

    # U3(theta, phi, lam) — general parameterization
    p_u3 = sub.add_parser("u3", parents=[common],
                           help="U3(theta, phi, lam)")
    p_u3.add_argument("theta", type=str)
    p_u3.add_argument("phi", type=str)
    p_u3.add_argument("lam", type=str)

    # Random unitary
    sub.add_parser("random", parents=[common],
                   help="Random Haar-uniform unitary")

    args = parser.parse_args()

    def parse_angle(s: str) -> float:
        """Parse angle string, supporting expressions like pi/7, 2*pi, etc."""
        s = s.replace("pi", str(np.pi))
        return float(eval(s))

    if args.gate == "rx":
        angle = parse_angle(args.angle)
        c, s = np.cos(angle / 2), np.sin(angle / 2)
        V = np.array([[c, -1j * s], [-1j * s, c]])
        desc = f"Rx({args.angle})"
    elif args.gate == "ry":
        angle = parse_angle(args.angle)
        c, s = np.cos(angle / 2), np.sin(angle / 2)
        V = np.array([[c, -s], [s, c]])
        desc = f"Ry({args.angle})"
    elif args.gate == "rz":
        angle = parse_angle(args.angle)
        V = rz_unitary(angle)
        desc = f"Rz({args.angle})"
    elif args.gate == "u3":
        theta = parse_angle(args.theta)
        phi = parse_angle(args.phi)
        lam = parse_angle(args.lam)
        V = np.array([
            [np.cos(theta / 2),
             -np.exp(1j * lam) * np.sin(theta / 2)],
            [np.exp(1j * phi) * np.sin(theta / 2),
             np.exp(1j * (phi + lam)) * np.cos(theta / 2)],
        ])
        desc = f"U3({args.theta}, {args.phi}, {args.lam})"
    elif args.gate == "random":
        V = random_u3()
        desc = "Random"
    else:
        parser.error(f"Unknown gate: {args.gate}")

    print(f"Synthesizing {desc} to precision epsilon={args.epsilon:.1e}")
    print(f"Expected lde ≈ {1.5 * np.log2(1 / args.epsilon) + 1:.0f}")
    print()

    result = synthesize(V, args.epsilon, max_t=args.max_lde,
                        verbose=not args.quiet)

    if result is None:
        print(f"\nNo solution found within max lde {args.max_lde}")
    else:
        print(f"\nSynthesis successful!")
        print(f"  lde:      {result['lde']}")
        print(f"  Distance: {result['distance']:.6e}")
        if result.get('gates'):
            print(f"  Gates:    {result['gates']}")
        elif result.get('gate_error'):
            print(f"  Gate extraction failed: {result['gate_error']}")

        U = result['unitary']
        print(f"\n  Verification:")
        print(f"    ||U||_2 = {np.linalg.norm(U, 2):.10f}")
        print(f"    d_diamond(U, V) = {diamond_distance(U, V):.6e}")
