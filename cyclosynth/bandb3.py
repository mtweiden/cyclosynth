import numpy as np
from numpy import array
from numpy import isclose
from numpy import ndarray
from numpy import sqrt
from numpy import round

from numba import njit

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
    scale = 1 << (k // 2) if k % 2 else 2 ** (k / 2)
    return sigma_to_xy @ uv * scale  # Try figuring out how to make shift work

def xy_to_uv(xy: ndarray, k: int = 3) -> ndarray:
    scale = 1 << (k // 2) if k % 2 else 2 ** (k / 2)
    return sigma_to_uv @ xy / scale

def to_unit_vector(y: ndarray) -> ndarray:
    return y / sqrt((y**2).sum())

def to_unitary(x: ndarray, k) -> ndarray:
    u = xy_to_uv(x, k)
    u1 = u[0] + 1j * u[1]
    u2 = u[2] + 1j * u[3]
    utry = array([[u1, -u2.conj()], [u2, u1.conj()]])
    return utry

def check_alignment(y: ndarray, x: ndarray, k: int = 3, eps: float = 1e-4) -> bool:
    inner = abs((y.dot(x)).sum())
    target = 2 ** (k - 1) * (1 - eps ** 2)
    return inner > target

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
def search_phase2(
    a1: int, c1: int, a2: int, c2: int,
    remaining_norm: int,
    solutions: ndarray,
    sol_count: int,
    max_solutions: int,
) -> int:
    """
    Given fixed (a1, c1, a2, c2), enumerate all (b1, d1, b2, d2) satisfying:
        1. b1^2 + d1^2 + b2^2 + d2^2 = remaining_norm
        2. b1(a1+c1) + d1(c1-a1) + b2(a2+c2) + d2(c2-a2) = 0

    We enumerate b1, d1, b2 freely and solve for d2 using the linear constraint.

    Args:
        a1, c1, a2, c2 (int): Fixed coordinates from phase 1.

        remaining_norm (int): Norm budget left for b1, d1, b2, d2.
            Equal to 2^k - a1^2 - c1^2 - a2^2 - c2^2.

        solutions (ndarray): Output array to write solutions into.

        sol_count (int): Current number of solutions found.

        max_solutions (int): Maximum number of solutions to store.

    Returns:
        int: Updated solution count.
    """

    # Precompute the coefficients of the linear unitarity constraint:
    # b1 * coeff_b1 + d1 * coeff_d1 + b2 * coeff_b2 + d2 * coeff_d2 = 0
    coeff_b1 = a1 + c1
    coeff_d1 = c1 - a1
    coeff_b2 = a2 + c2
    coeff_d2 = c2 - a2

    # We need to solve for one variable in terms of the others.
    # Pick the variable with the largest coefficient to minimize rounding issues.
    # If all coefficients are zero, the constraint is trivially satisfied.
    abs_b1 = abs(coeff_b1)
    abs_d1 = abs(coeff_d1)
    abs_b2 = abs(coeff_b2)
    abs_d2 = abs(coeff_d2)
    max_coeff = max(abs_b1, abs_d1, abs_b2, abs_d2)

    if max_coeff == 0:
        # Unitarity is trivially satisfied for any (b1, d1, b2, d2).
        # Enumerate all representations of remaining_norm as sum of 4 squares.
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
                    # d2^2 must equal rem3 exactly
                    d2_abs = int(round(rem3 ** 0.5))
                    if d2_abs * d2_abs != rem3:
                        continue
                    # for d2 in (d2_abs, -d2_abs) if d2_abs > 0 else (0,):
                    for d2_sign in range(2):
                        if d2_sign == 0:
                            d2 = d2_abs
                        else:
                            d2 = -d2_abs
                        if d2_sign == 1 and d2_abs ==0:
                            continue
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

    # Determine which variable to solve for (largest coefficient)
    # solve_var: 0=b1, 1=d1, 2=b2, 3=d2
    if max_coeff == abs_b1:
        solve_var = 0
    elif max_coeff == abs_d1:
        solve_var = 1
    elif max_coeff == abs_b2:
        solve_var = 2
    else:
        solve_var = 3

    # Enumerate three free variables, solve for the fourth.
    # The linear constraint is:
    #   free1*c1 + free2*c2 + free3*c3 + solved*c_solved = 0
    #   solved = -(free1*c1 + free2*c2 + free3*c3) / c_solved

    if solve_var == 0:
        # Solve for b1: b1 = -(d1*coeff_d1 + b2*coeff_b2 + d2*coeff_d2) / coeff_b1
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
                    # Solve for b1
                    numer = -(d1 * coeff_d1 + b2 * coeff_b2 + d2 * coeff_d2)
                    if numer % solve_coeff != 0:
                        continue
                    b1 = numer // solve_coeff
                    if b1 * b1 != rem3:
                        continue
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

    elif solve_var == 1:
        # Solve for d1: d1 = -(b1*coeff_b1 + b2*coeff_b2 + d2*coeff_d2) / coeff_d1
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

    elif solve_var == 2:
        # Solve for b2: b2 = -(b1*coeff_b1 + d1*coeff_d1 + d2*coeff_d2) / coeff_b2
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

    else:
        # Solve for d2: d2 = -(b1*coeff_b1 + d1*coeff_d1 + b2*coeff_b2) / coeff_d2
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
def search_phase1(
    target_norm: int,
    y_list: ndarray,
    solutions: ndarray,
    max_solutions: int,
) -> int:
    """
    Enumerate all (a1, c1, a2, c2) centered around the target direction y,
    then for each combination call phase 2 to find valid (b1, d1, b2, d2).

    Args:
        target_norm (int): Target squared norm (2^k).

        y_list (ndarray): Target direction vector (8 components).

        solutions (ndarray): Output array to write solutions into.

        max_solutions (int): Maximum number of solutions to store.

    Returns:
        int: Number of solutions found.
    """
    sol_count = 0

    # Compute scale factor to map y onto the sphere ||x||^2 = target_norm
    y_norm_sq = 0.0
    for i in range(8):
        y_norm_sq += y_list[i] * y_list[i]
    scale = (target_norm / y_norm_sq) ** 0.5

    # Ideal values for phase 1 coordinates
    # x = (a1, b1, c1, d1, a2, b2, c2, d2)
    # y_list indices: a1=0, b1=1, c1=2, d1=3, a2=4, b2=5, c2=6, d2=7
    a1_center = int(round(y_list[0] * scale))
    c1_center = int(round(y_list[2] * scale))
    a2_center = int(round(y_list[4] * scale))
    c2_center = int(round(y_list[6] * scale))

    max_a1 = int(target_norm ** 0.5)

    # Enumerate a1 outward from center
    for a1_offset in range(max_a1 + abs(a1_center) + 1):
        for a1_sign in range(2 if a1_offset > 0 else 1):
            if a1_sign == 0:
                a1 = a1_center + a1_offset
            else:
                a1 = a1_center - a1_offset
            if a1 * a1 > target_norm:
                continue

            rem1 = target_norm - a1 * a1
            max_c1 = int(rem1 ** 0.5)

            # Enumerate c1 outward from center
            for c1_offset in range(max_c1 + abs(c1_center) + 1):
                for c1_sign in range(2 if c1_offset > 0 else 1):
                    if c1_sign == 0:
                        c1 = c1_center + c1_offset
                    else:
                        c1 = c1_center - c1_offset
                    if c1 * c1 > rem1:
                        continue

                    rem2 = rem1 - c1 * c1
                    max_a2 = int(rem2 ** 0.5)

                    # Enumerate a2 outward from center
                    for a2_offset in range(max_a2 + abs(a2_center) + 1):
                        for a2_sign in range(2 if a2_offset > 0 else 1):
                            if a2_sign == 0:
                                a2 = a2_center + a2_offset
                            else:
                                a2 = a2_center - a2_offset
                            if a2 * a2 > rem2:
                                continue

                            rem3 = rem2 - a2 * a2
                            max_c2 = int(rem3 ** 0.5)

                            # Enumerate c2 outward from center
                            for c2_offset in range(max_c2 + abs(c2_center) + 1):
                                for c2_sign in range(2 if c2_offset > 0 else 1):
                                    if c2_sign == 0:
                                        c2 = c2_center + c2_offset
                                    else:
                                        c2 = c2_center - c2_offset
                                    if c2 * c2 > rem3:
                                        continue

                                    remaining_norm = rem3 - c2 * c2

                                    sol_count = search_phase2(
                                        a1, c1, a2, c2,
                                        remaining_norm,
                                        solutions,
                                        sol_count,
                                        max_solutions,
                                    )

                                    if sol_count >= max_solutions:
                                        return sol_count

    return sol_count


def branch_and_bound(y_hat: ndarray, k: int, max_solutions: int = 100000) -> ndarray:
    """
    Find integer vectors x = (a1, b1, c1, d1, a2, b2, c2, d2) satisfying:
        - ||x||^2 = 2^k
        - b1(a1+c1) + d1(c1-a1) + b2(a2+c2) + d2(c2-a2) = 0

    Search is centered around the target direction y for early discovery
    of good approximations.

    Args:
        y (ndarray): Target direction vector (8 components).

        k (int): Determines the target norm 2^k (related to T-count).

        max_solutions (int): Maximum number of solutions to collect.

    Returns:
        ndarray: Array of shape (n, 8) containing the solutions found.
    """
    assert isclose(np.linalg.norm(y_hat, 2), 1), "y_hat must be a unit vector"
    target_norm = 2 ** k
    y_list = np.array([float(yi) for yi in y])
    solutions = np.zeros((max_solutions, 8), dtype=np.int64)
    sol_count = search_phase1(target_norm, y_list, solutions, max_solutions)
    return solutions[:sol_count]


if __name__ == "__main__":
    k = 6
    max_solutions = 10
    v = random_u3()
    print(f"target unitary: {v}")
    y = (uv_to_xy(v, k))
    print(y)
    _y = [int(round(yi)) for yi in y]
    print(_y)
    y_hat = to_unit_vector(y)
    sols = branch_and_bound(y_hat, k, max_solutions)
    for sol in sols:
        utry = to_unitary(sol, k)
        # print(f"solution: {sol}   utry: {utry}")
        uv = xy_to_uv(sol, k)
        print(f"solution: {sol}")
        print(f"translation: {uv}")
        assert isclose(np.linalg.norm(utry, 2), 1)
