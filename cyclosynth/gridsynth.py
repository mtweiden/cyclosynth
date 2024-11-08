"""An implementation of the Gridsynth approximation algorithm."""
from typing import Generator
from typing import Sequence

from mpmath import mp
from mpmath import matrix
from mpmath import cos
from mpmath import diag
from mpmath import sin
from mpmath import sqrt
from mpmath import inverse

from cyclosynth.algebra import AlgebraicInteger
from cyclosynth.algebra import RingRoot2
from cyclosynth.ellipse import Ellipse
from cyclosynth.matrix import U2Matrix


mp.dps = 100

def gridsynth(angle: float, epsilon: float) -> U2Matrix:
    ...

def enumerate_solutions() -> Sequence[RingRoot2]:
    pass

def enumerate_points_for_k(
    x_lo: float,
    x_hi: float,
    y_lo: float,
    y_hi: float,
    k: int,
) -> Generator[RingRoot2, None, None] | None:
    """
    Enumerate the finite sequence of solutions to alpha/sqrt(2)^k to the scaled
    grid problem for epsilon-region A given least demoninator exponent k.

    Args:
        x_lo (float): The lower bound of the x-coordinate of the epsilon region.
        x_hi (float): The upper bound of the x-coordinate of the epsilon region.
        y_lo (float): The lower bound of the y-coordinate of the epsilon region.
        y_hi (float): The upper bound of the y-coordinate of the epsilon region.
        k (int): The least denominator exponent.

    Returns:
        solutions (Generator[AlgebraicInteger | None, None, None]): The 
            solutions to the scaled grid problem. If there is no solution for
            the given k, return None.
    
    Notes:
      - There are two forms that solutions can take. Either they are in the form
        x+iy where x,y in Z[√2], or they are in the form (x-1/√2)+i(y-1/√2).
    """
    x_lo_ = int(sqrt(2) ** k * x_lo)
    x_hi_ = int(sqrt(2) ** k * x_hi)
    y_lo_ = int(sqrt(2) ** k * y_lo)
    y_hi_ = int(sqrt(2) ** k * y_hi)

    x_lo_shift_ = int(sqrt(2) ** k * (x_lo - 1 / sqrt(2)))
    x_hi_shift_ = int(sqrt(2) ** k * (x_hi - 1 / sqrt(2)))
    y_lo_shift_ = int(sqrt(2) ** k * (y_lo - 1 / sqrt(2)))
    y_hi_shift_ = int(sqrt(2) ** k * (y_hi - 1 / sqrt(2)))

    unshifted_solutions = x_hi_ > x_lo_ and y_hi_ > y_lo_
    shifted_solutions = x_hi_shift_ > x_lo_shift_ and y_hi_shift_ > y_lo_shift_

    if unshifted_solutions:
        for x in range(x_lo_, x_hi_ + 1):
            for y in range(y_lo_, y_hi_ + 1):
                yield RingRoot2([x, y])
    
    if shifted_solutions:
        for x in range(x_lo_shift_, x_hi_shift_ + 1):
            for y in range(y_lo_shift_, y_hi_shift_ + 1):
                yield RingRoot2([x, y])
    
    if not unshifted_solutions and not shifted_solutions:
        yield None

def attempt_factorization() -> Sequence[int]:
    """
    Let n = 2^k - alpha.dagger * alpha.

    Attempt to find a prime factorization of n. Repeat until a factorization
    is found.

    Args:
        n (int): The number to factor.

    Returns:
        prime_factors (Sequence[int]): The prime factors of n.
    """
    pass

def solve_for_beta() -> AlgebraicInteger | None:
    """
    Solve for beta in the equation beta.dagger * beta = n.

    Returns:
        beta (AlgebraicInteger | None): The solution to the equation 
            beta.dagger * beta = n, or None if no solution exists.
    """
    pass

def find_ellipse(angle: float, epsilon: float) -> Ellipse:
    """
    Find the ellipse matrix for the given angle and precision.

    Returns:
        (Ellipse): An ellipse centered at and rotated along the epsilon
            region of the given angle.
    """
    d = 1 - (epsilon ** 2 / 2)  # distance from origin of e-region center
    zx = cos(-angle / 2)  # width of e-region
    zy = sin(-angle / 2)  # height of e-region
    center = (d * zx, d * zy)

    # Eigenvalues for scaling the determining matrix
    ev1, ev2 = 4 / (epsilon ** 4), 1 / (epsilon ** 2)
    
    # Construct the determining matrix
    bmat = matrix([[zx, -zy], [zy, zx]])
    mmat = diag([ev1, ev2])
    mat = bmat @ mmat @ inverse(bmat)

    return Ellipse(mat, center)