from fpylll import LLL
from fpylll import IntegerMatrix
from mpmath import mp
# from mpmath import sqrt
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
angle = exp(1j * pi / 16)
v = array([real(angle), imag(angle), 0, 0])

#===============================================================================
# Some helpful functions
#===============================================================================
def uv_to_xy(uv: ndarray, k: int = 3) -> ndarray:
    assert k % 2 == 0
    scale = 2 ** (k / 2)
    return sigma_to_xy @ uv * scale  # Try figuring out how to make shift work

def xy_to_uv(xy: ndarray, k: int = 3) -> ndarray:
    assert k % 2 == 0
    scale = 2 ** (k / 2)
    return sigma_to_uv @ xy / scale

def compute_y_hat(y: ndarray) -> ndarray:
    return y / sqrt((y**2).sum())

y = uv_to_xy(v, k)
back_to_v = xy_to_uv(y, k)
assert isclose(v, back_to_v).all()
y_hat = compute_y_hat(y)
assert isclose(sqrt((y_hat**2).sum()), 1)

def check_alignment(y: ndarray, x: ndarray, k: int = 3, eps: float = 1e-4) -> bool:
    inner = abs((y.transpose() @ x).sum())
    target = (2 ** k ) * (1 - eps ** 2)
    return inner > target

def check_norm(x: ndarray, k: int = 3) -> bool:
    mag_squared = (x.transpose() @ x).sum()
    target = 2 ** (k - 1)
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
print(proj)
print(y_rounded)
# this is definitely close enough...
# I wasted like 2 days learning about this :(

#===============================================================================
# An attempt at branch and bound
#===============================================================================
def branch_and_bound(y: ndarray, k: int, epsilon: float) -> ndarray:
    target_norm_2 = 2 ** (k - 1)
    target_alignment = 2 ** (k - 1) * (1 - epsilon ** 2)
    solutions = []
    y_list = tuple(float(yi) for yi in y)

    # Precompute partial sums of y_i^2 from the right
    # remaining_y_norm_2[d] = y[d]^2 + y[d+1]^2 + ... + y[7]^2
    remaining_y_norm_2 = [0.0] * 9
    for i in range(7, -1, -1):
        remaining_y_norm_2[i] = remaining_y_norm_2[i + 1] + y[i] ** 2

    def unitarity_check(x: tuple[int, ...]) -> bool:
        a1, b1, c1, d1, a2, b2, c2, d2 = x
        lhs = (a1 * d1) + (a2 * d2)
        rhs = (a1 * b1) + (a2 * b2) + (b1 * c1) + (b2 * c2) + (c1 * d1) + (c2 * d2)
        return lhs == rhs

    def search(depth: int, x_partial: list[int], norm_2_so_far: int, dot_so_far: float) -> None:
        """
        Args:
            depth (int): How many coordinates we've fixed (0 to 8).

            x_partial (list[int]): List of assigned integer coordinates so far.

            norm_2_so_far (int): Squared sum of squares of assigned coordinates.

            dot_so_far (float): partial dot product with y.
        """

        # Base case: all 8 coordinates assigned
        if depth == 8:
            if norm_2_so_far != target_norm_2:
                return
            if dot_so_far < 0 or abs(dot_so_far) < target_alignment:
                return
            x = tuple(x_partial)
            if unitarity_check(x):
                solutions.append(x)
            return

        # NORM PRUNING: How much norm_2 budget remains?
        remaining_norm_2_budget = target_norm_2 - norm_2_so_far
        if remaining_norm_2_budget < 0:
            return

        # Each remaining coordinate is at most sqrt(remaining_norm_2_budget) and
        # we need (8 - depth) coordinates to use up remaining_norm_2_budget exactly.
        max_coord = int(remaining_norm_2_budget ** 0.5)

        # ALIGNMENT PRUNING: Can we be aligned enough with y?
        # Compute the maximum possible dot product from the remaining coordinates
        # using Cauchy-Schwarz. The remaining coordinates contribute at most
        # sqrt(remaining_norm_2_budget) * sqrt(sum of remaining y_i^2)
        max_remaining_dot = (remaining_norm_2_budget * remaining_y_norm_2[depth]) ** 0.5

        # If even the best case can't satisfy alignment, prune
        best_possible = abs(dot_so_far) + max_remaining_dot
        if best_possible < target_alignment:
            return

        # Choose search center based on y direction. Ideally this coordinate
        # should be proportional to y[depth] scaled to use the right share of
        # remaining norm_2.
        if remaining_y_norm_2[depth] > 0:
            ideal = y_list[depth] * (remaining_norm_2_budget / remaining_y_norm_2[depth]) ** 0.5
        else:
            ideal = 0

        # Search over all valid coordinate values, sorted by distance from ideal
        # so that we explore the most promising branches first
        lo = -max_coord
        hi = max_coord
        candidates = sorted(range(lo, hi + 1), key=lambda v: abs(v - ideal))

        for val in candidates:
            val_2 = val * val
            if val_2 > remaining_norm_2_budget:
                continue

            # NUMBER THEORETIC PRUNING
            # 1. With 2 remaining variables, we can't have the remaining norm be 3 (mod 4)
            #    otherwise we must have a sqrt(2) somewhere.
            remaining_dims = 8 - depth
            remaining = remaining_norm_2_budget - val_2
            if remaining_dims == 2 and remaining % 4 == 3:
                continue
            # 2. With 1 remaining variable, remaining norm must be a perfect square
            if remaining_dims == 1:
                sqrt_r = int(round(remaining ** 0.5))
                if not isclose(sqrt_r * sqrt_r, remaining):
                    continue

            x_partial.append(val)
            search(
                depth + 1,
                x_partial,
                norm_2_so_far + val_2,
                dot_so_far + val * y_list[depth],
            )
            x_partial.pop()

    search(0, [], 0, 0.0)
    return solutions

k = 12
v = random_u3()
y = uv_to_xy(v, k)
solutions = branch_and_bound(y, k, 0.2)
max_norm = max([sqrt((array(x) @ array(x)).sum()) for x in solutions])
for sol in solutions:
    passes_unitary = check_unitarity(array(sol))
    passes_norm = check_norm(array(sol), k)
    passes_alignment = check_alignment(y, array(sol), k)
    if not passes_unitary:
        print(f"Solution: {sol}, unitary: {passes_unitary}")
    if not passes_norm:
        print(f"Solution: {sol}, norm: {sqrt((array(sol) @ array(sol)).sum()):.4f}")
    if not passes_alignment:
        inner = abs((y.transpose() @ array(sol)).sum())
        print(f"Pased bad solution: {sol}, alignment: {inner:.4f}")
y_norm = sqrt((y @ y).sum())
print(f"Found {len(solutions)} solutions. Max norm: {max_norm:.4f}, y norm: {y_norm:.4f}")
