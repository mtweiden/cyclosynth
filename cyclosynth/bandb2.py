from fpylll import LLL
from fpylll import IntegerMatrix
from mpmath import mp
# from mpmath import sqrt
import numpy as np
from numpy import array
from numpy import exp
from numpy import eye
from numpy import isclose
from numpy import ndarray
from numpy import real
from numpy import imag
from numpy import sqrt
from numpy import pi
from numpy import round
from numpy.linalg import solve

from numba import njit

from random import random

mp.dps = 50

#===============================================================================
# Some constants
#===============================================================================
k = 20 # Just for testing around
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
# Our example target vector
def random_rz() -> ndarray:
    # [e^(-i pi / N)        0]
    # [0        e^(+i pi / N)]
    angle = exp(-1j * pi / 4)
    return array([real(angle), imag(angle), 0, 0])
v = random_rz()

#===============================================================================
# Some helpful functions
#===============================================================================
def uv_to_xy(uv: ndarray, k: int = 3) -> ndarray:
    scale = 1 << (k // 2) if k % 2 else 2 ** (k / 2)
    return sigma_to_xy @ uv * scale  # Try figuring out how to make shift work

def xy_to_uv(xy: ndarray, k: int = 3) -> ndarray:
    scale = 1 << (k // 2) if k % 2 else 2 ** (k / 2)
    return sigma_to_uv @ xy / scale

def compute_y_hat(y: ndarray) -> ndarray:
    return y / sqrt((y**2).sum())

y = uv_to_xy(v, k)
back_to_v = xy_to_uv(y, k)
assert isclose(v, back_to_v).all()
y_hat = compute_y_hat(y)
assert isclose(sqrt((y_hat**2).sum()), 1)

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

def to_integer_matrix(M: ndarray, k: int = 3) -> IntegerMatrix:
    B_int = IntegerMatrix(8, 8)
    scale = 2 ** ((k - 1) / 2)
    for i in range(8):
        for j in range(8):
            val = int(scale * M[i, j])
            B_int[i, j] = val
    return B_int

def compute_new_basis(y: ndarray, k: int = 3) -> list[ndarray]:
    y_hat = compute_y_hat(y)
    P = eye(8) - y_hat[:, None] @ y_hat[None, :]
    lll_scale = 0.001
    M = eye(8) + (1/lll_scale - 1) * P
    M = to_integer_matrix(M, 100)
    U_track = IntegerMatrix.identity(8)
    LLL.reduction(M, U_track)
    bases = []
    for i in range(8):
        b = list(U_track[i])[:8]
        # It's philosphically nice to have this inner product be positive
        if array(b) @ y_hat < 0:
            b = [-x for x in list(b)]
        else:
            b = list(b)
        bases.append(array(b, dtype=int))
    return bases

#===============================================================================
# Part 1a: find a better basis for the lattice using LLL
# Part 1b: just find a better starting point to begin enumeration
#===============================================================================
# We want to find a lattice basis that corresponds to being as aligned with our
# target vector as possible. The way we do this is by computing a projector
# matrix P = I - y_hat  y_hat^T. Here, y_hat is just the unit vector in the
# direction of our target vector y. Next we compute M = I + (1/lll_scale - 1) P.
# This will penalize basis vectors not aligned with y. Then we run LLL on M and
# it will give us a new basis that is more aligned with y.

# Scaling approach
bases = compute_new_basis(y, 10)
B = array(bases).transpose()
w = solve(B, y)

# For random U3, does this transformed basis look any nicer?
def random_u3() -> ndarray:
    a, b, c, d = random(), random(), random(), random()
    norm = sqrt(a**2 + b**2 + c**2 + d**2)
    a, b, c, d = a / norm, b / norm, c / norm, d / norm
    assert isclose(a**2 + b**2 + c**2 + d**2, 1)
    return array([a, b, c, d])

v = random_u3()
y = uv_to_xy(v, k)

bases = compute_new_basis(y, 10)
B = array(bases).transpose()
w = solve(B, y)

from numpy import zeros
proj = zeros(8, dtype=int)
for i in range(8):
    val = w[i] * B[:, i]
    val = val.astype(int)
    proj += round(val)


# What if I just round y?
y_rounded = round(y)
# this is definitely close enough...
# I wasted like 2 days learning about this :(

#===============================================================================
# An attempt at branch and bound
#===============================================================================

@njit
def search(
    depth: int,
    x_partial: ndarray,
    norm_2_so_far: int,
    dot_so_far: float,
    y_list: ndarray,
    remaining_y_norm_2: ndarray,
    target_norm_2: int,
    target_alignment: float,
    solution: ndarray,
) -> bool:
    """
    Args:
        depth (int): How many coordinates we've fixed (0 to 8).

        x_partial (ndarray): Fixed-size array of assigned integer coordinates.

        norm_2_so_far (int): Sum of squares of assigned coordinates.

        dot_so_far (float): Partial dot product with y.

        y_list (ndarray): Target direction vector.

        remaining_y_norm_2 (ndarray): Precomputed partial sums of y_i^2 from the right.

        target_norm_2 (int): Target squared norm (2^(k-1)).

        target_alignment (float): Target alignment threshold (2^(k-1) * (1 - epsilon^2)).

        solution (ndarray): Output array to write the solution into.

    Returns:
        bool: True if a solution was found.
    """

    # Base case: all 8 coordinates assigned
    if depth == 8:
        if norm_2_so_far != target_norm_2:
            return False
        if abs(dot_so_far) < target_alignment:
            return False
        # Unitarity check
        a1, b1, c1, d1 = x_partial[0], x_partial[1], x_partial[2], x_partial[3]
        a2, b2, c2, d2 = x_partial[4], x_partial[5], x_partial[6], x_partial[7]
        lhs = (a1 * d1) + (a2 * d2)
        rhs = (a1 * b1) + (a2 * b2) + (b1 * c1) + (b2 * c2) + (c1 * d1) + (c2 * d2)
        if lhs == rhs:
            solution[:] = x_partial[:]
            return True
        return False

    # NORM PRUNING: How much norm_2 budget remains?
    remaining_norm_2_budget = target_norm_2 - norm_2_so_far
    if remaining_norm_2_budget < 0:
        return False

    # Each remaining coordinate is at most sqrt(remaining_norm_2_budget) and we
    # need (8 - depth) coordinates to use up remaining_norm_2_budget exactly.
    max_coord = int(remaining_norm_2_budget ** 0.5)

    # ALIGNMENT PRUNING: Can we be aligned enough with y?
    # Compute the maximum possible dot product from the remaining coordinates
    # using Cauchy-Schwarz. The remaining coordinates contribute at most
    # sqrt(remaining_norm_2_budget) * sqrt(sum of remaining y_i^2)
    max_remaining_dot = (remaining_norm_2_budget * remaining_y_norm_2[depth]) ** 0.5

    # If even the best case can't satisfy alignment, prune
    if abs(dot_so_far) + max_remaining_dot < target_alignment:
        return False

    # Choose search center based on y direction. Ideally this coordinate
    # should be proportional to y[depth] scaled to use the right share of
    # remaining norm_2.
    if remaining_y_norm_2[depth] > 0:
        ideal = y_list[depth] * (remaining_norm_2_budget / remaining_y_norm_2[depth]) ** 0.5
    else:
        ideal = 0.0

    center = int(round(ideal))

    # Search outward from center so we explore the most promising branches first
    for offset in range(max_coord + abs(center) + 1):
        for sign_idx in range(2 if offset > 0 else 1):
            if sign_idx == 0 and offset == 0:
                val = center
            elif sign_idx == 0:
                val = center + offset
            else:
                val = center - offset

            val_2 = val * val
            if val_2 > remaining_norm_2_budget:
                continue

            x_partial[depth] = val
            found = search(
                depth + 1,
                x_partial,
                norm_2_so_far + val_2,
                dot_so_far + val * y_list[depth],
                y_list,
                remaining_y_norm_2,
                target_norm_2,
                target_alignment,
                solution,
            )
            if found:
                return True

    return False


def branch_and_bound(y: ndarray, k: int, epsilon: float) -> ndarray:
    target_norm_2 = 2 ** (k)
    target_alignment = 2 ** (k - 1) * (1 - epsilon ** 2)

    y_list = np.array([float(yi) for yi in y])
    
    # -----
    # Debug: verify y is a valid solution
    print(f"y = {y}")
    print(f"||y||^2 = {sum(yi**2 for yi in y)}, target = {target_norm_2}")
    print(f"target_alignment = {target_alignment}")
    a1, b1, c1, d1 = y[0], y[1], y[2], y[3]
    a2, b2, c2, d2 = y[4], y[5], y[6], y[7]
    lhs = (a1 * d1) + (a2 * d2)
    rhs = (a1 * b1) + (a2 * b2) + (b1 * c1) + (b2 * c2) + (c1 * d1) + (c2 * d2)
    print(f"unitarity: lhs={lhs}, rhs={rhs}, pass={lhs==rhs}")
    # -----

    # Precompute partial sums of y_i^2 from the right
    # remaining_y_norm_2[d] = y[d]^2 + y[d+1]^2 + ... + y[7]^2
    remaining_y_norm_2 = np.zeros(9)
    for i in range(7, -1, -1):
        remaining_y_norm_2[i] = remaining_y_norm_2[i + 1] + y_list[i] ** 2

    x_partial = np.zeros(8, dtype=np.int64)
    solution = np.zeros(8, dtype=np.int64)

    found = search(
        0, x_partial, 0, 0.0, y_list, remaining_y_norm_2,
        target_norm_2, target_alignment, solution,
    )

    if found:
        return solution
    return np.array([], dtype=np.int64)

k = 6
eps = 0.9
v = random_rz()
y = uv_to_xy(v, k)
import pdb; pdb.set_trace()
x = branch_and_bound(y, k, eps)
if len(x) == 0:
    print("No solutions found.")
else:
    max_norm = array(x).dot(x).sum() ** 0.5
    passes_unitary = check_unitarity(array(x))
    passes_norm = check_norm(array(x), k)
    passes_alignment = check_alignment(y, array(x), k, eps)
    if not passes_unitary:
        print(f"Bad solution: {x}, unitary: {passes_unitary}")
    elif not passes_norm:
        print(f"Bad solution: {x}, norm: {sqrt((array(x) @ array(x)).sum()):.4f}")
    elif not passes_alignment:
        inner = abs((y.transpose() @ array(x)).sum())
        print(f"Bad solution: {x}, alignment: {inner:.4f}")
    else:
        print(f"Solution: {x}, norm: {sqrt((array(x) @ array(x)).sum()):.4f}")
