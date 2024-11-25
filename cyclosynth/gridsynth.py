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
from mpmath import floor
from mpmath import ceil

from cyclosynth.algebra import AlgebraicInteger
from cyclosynth.algebra import RingRoot2
from cyclosynth.convex import ConvexSet
from cyclosynth.operator import Operator
from cyclosynth.matrix import U2Matrix
from cyclosynth.ratio import IntegerRatio
from cyclosynth.utils import floor_log


mp.dps = 100

# Constants

def gridsynth(angle: float, epsilon: float) -> U2Matrix:
    ...
    # Find ellipse
    # Make ellipse upright
    # Compute the uprighted bounds of the ellipse and unit disc
    # Find grid points
    # Check if grid problem solutions are in the epsilon region


def scaled_gridpoints_1d(
    x_lo: float,
    x_hi: float,
    y_lo: float,
    y_hi: float,
    k: int,
) -> Generator[RingRoot2, None, None]:
    if k % 2 == 0:
        scale_inv = RingRoot2((2 ** (k // 2), 0))
        sign = 1
    else:
        scale_inv = RingRoot2((0, 2 ** (k // 2 + 1)))
        sign = -1
    scale = IntegerRatio(1, scale_inv)
    scale_inv = scale_inv.to_float()

    x0, x1 = x_lo * scale_inv, x_hi * scale_inv
    y0, y1 = sign * y_lo * scale_inv, sign * y_hi * scale_inv

    for candidate in gridpoints_1d(x0, x1, y0, y1):
        yield scale * candidate


def gridpoints_1d(
    x_lo: float,
    x_hi: float,
    y_lo: float,
    y_hi: float,
) -> Generator[RingRoot2, None, None]:
    """
    Find grid problem solutions for a 1D grid problem.

    Based off gridsynth implementation.

    Args:
        x_lo (float): The lower bound of the x-coordinate.
        x_hi (float): The upper bound of the x-coordinate.
        y_lo (float): The lower bound of the y-coordinate.
        y_hi (float): The upper bound of the y-coordinate.
    
    Yields:
        candidate (RingRoot2): A solution to a 1D grid problem.
    """
    # Compute scale factor alpha
    # We expect alpha ~ (x_0 + y_0) / 2
    # and alpha.conj() ~ (x_0 - y_0) / 2
    a = int(floor((x_lo + y_lo) / 2))
    b = int(floor(sqrt(2) * (x_lo - y_lo))) // 4
    alpha = RingRoot2([a, b])

    # Rescale grid problem using offsets
    x0 = x_lo - alpha.to_float()
    x1 = x_hi - alpha.to_float()
    y0 = y_lo - alpha.conj().to_float()
    y1 = y_hi - alpha.conj().to_float()

    # Check if number is a grid problem solution for [x0, x1], [y0, y1]
    def test_solution(number: RingRoot2) -> bool:
        in_A = x_lo <= number.to_float() <= x_hi
        in_B = y_lo <= number.conj().to_float() <= y_hi
        return in_A and in_B
    
    # Use gridpoints_internal to find candidate solutions in a more numerically
    # stable way
    for candidate in gridpoints_internal(x0, x1, y0, y1):
        candidate = candidate + alpha
        if test_solution(candidate):
            yield candidate


def gridpoints_internal(
    x_lo: float,
    x_hi: float,
    y_lo: float,
    y_hi: float,
    scale_output: RingRoot2 = RingRoot2([1, 0]),
    conjugate_output: bool = False,
) -> Generator[RingRoot2, None, None]:
    """
    Internal function used to find grid problem solutions.

    Based off gridsynth implementation.

    Args:
        x_lo (float): The lower bound of the x-coordinate.
        x_hi (float): The upper bound of the x-coordinate.
        y_lo (float): The lower bound of the y-coordinate.
        y_hi (float): The upper bound of the y-coordinate.
        scale_output (RingRoot2): The scale factor to apply to the output.
        conjugate_output (bool): Whether to take the conjugate of the output.
    
    Yields:
        beta (RingRoot2): A potential offset solution to a 1D grid problem.
    """
    # Compute interval widths
    dx = x_hi - x_lo
    dy = y_hi - y_lo

    lambda_ = RingRoot2([1, 1])
    lambda_inv = RingRoot2([-1, 1])

    # Determine a scaling factor n so that we can approximate the width of the
    # interval dy as (1 + sqrt(2))^n
    n, _ = floor_log(abs(dy))
    if dy < 0:
        n = -n

    if n >= 0:
        lambda_n = lambda_ ** n  # lambda ^ n
        lambda_inv_n = lambda_inv ** n  # (lambda^-1) ^ n
        lambda_bul_n = (-lambda_inv) ** n   # (-lambda^-1) ^ n
    else:
        lambda_n = lambda_inv ** -n
        lambda_inv_n = lambda_ ** -n
        lambda_bul_n = lambda_ ** -n

    if dy <= 0 and dx > 0:
        yield from gridpoints_internal(
            y_lo, y_hi, x_lo, x_hi, conjugate_output=True,
        )
    elif dy >= lambda_.to_float() and n % 2 == 0:
        yield from gridpoints_internal(
            lambda_n.to_float() * x_lo,
            lambda_n.to_float() * x_hi,
            lambda_bul_n.to_float() * y_lo,
            lambda_bul_n.to_float() * y_hi,
            scale_output=lambda_inv_n,
        )
    elif dy >= lambda_.to_float() and n % 2 == 1:
        yield from gridpoints_internal(
            lambda_n.to_float() * x_lo,
            lambda_n.to_float() * x_hi,
            lambda_bul_n.to_float() * y_hi,
            lambda_bul_n.to_float() * y_lo,
            scale_output=lambda_inv_n,
        )
    elif dy > 0 and dy < 1 and n % 2 == 0:
        yield from gridpoints_internal(
            lambda_n.to_float() * x_lo,
            lambda_n.to_float() * x_hi,
            lambda_bul_n.to_float() * y_lo,
            lambda_bul_n.to_float() * y_hi,
            scale_output=lambda_n,
        )
    elif dy > 0 and dy < 1 and n % 2 == 1:
        yield from gridpoints_internal(
            lambda_n.to_float() * x_lo,
            lambda_n.to_float() * x_hi,
            lambda_bul_n.to_float() * y_hi,
            lambda_bul_n.to_float() * y_lo,
            scale_output=lambda_n,
        )
    else:
        amin = int(ceil((x_lo + y_lo) / 2))
        amax = int(floor((x_hi + y_hi) / 2))
        for a in range(int(amin), int(amax) + 1):
            bmin = int(ceil((a - y_hi) / sqrt(2)))
            bmax = int(floor((a - y_lo) / sqrt(2)))
            for b in range(int(bmin), int(bmax) + 1):
                beta = RingRoot2([a, b]) * scale_output
                if conjugate_output:
                    beta = beta.conj()
                yield beta


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


def in_epsilon_region(
    angle: float,
    epsilon: float,
    x_coordinate: IntegerRatio,
    y_coordinate: IntegerRatio,
) -> bool:
    """
    Check if the point is in the epsilon region.
    """
    x_goal = cos(-angle / 2)
    y_goal = sin(-angle / 2)
    x = x_coordinate.to_float()
    y = y_coordinate.to_float()
    dist = sqrt((x - x_goal) ** 2 + (y - y_goal) ** 2)
    return dist <= epsilon