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
from cyclosynth.ellipse import Ellipse
from cyclosynth.matrix import U2Matrix
from cyclosynth.ratio import AlgebraicIntegerOverRoot2


mp.dps = 100

def gridsynth(angle: float, epsilon: float) -> U2Matrix:
    ...
    # Find ellipse
    # Make ellipse upright
    # Compute the uprighted bounds of the ellipse and unit disc
    # Find grid points
    # Check if grid problem solutions are in the epsilon region


def solve_grid_problem(
    Ax_lo: float,
    Ax_hi: float,
    Ay_lo: float,
    Ay_hi: float,
    Bx_lo: float,
    Bx_hi: float,
    By_lo: float,
    By_hi: float,
    k: int,
) -> Generator[AlgebraicIntegerOverRoot2, None, None]:
    """
    Solve the 2D grid problem for convex sets (Ax x Ay), (Bx x By).

    Args:
        Ax_lo (float): The lower bound of the x-coordinate of the A region.
        Ax_hi (float): The upper bound of the x-coordinate of the A region.
        Ay_lo (float): The lower bound of the y-coordinate of the A region.
        Ay_hi (float): The upper bound of the y-coordinate of the A region.
        Bx_lo (float): The lower bound of the x-coordinate of the B region.
        Bx_hi (float): The upper bound of the x-coordinate of the B region.
        By_lo (float): The lower bound of the y-coordinate of the B region.
        By_hi (float): The upper bound of the y-coordinate of the B region.
    
    Yields:
        solutions (AlgebraicIntegerOverRoot2): Candidate solutions to the grid
            problem.
    
    Notes:
        - Solutions are alpha+i*beta in Z[omega] where alpha, beta in Z[√2].
    """
    def widen_interval(lo: float, hi: float) -> tuple[float, float]:
        jiggle = (hi - lo) * 1e-4
        return lo - jiggle, hi + jiggle
    
    lambda_ = RingRoot2([1, 1])

    args = (*widen_interval(Ax_lo, Ax_hi), *widen_interval(Ay_lo, Ay_hi), k)
    for beta in solve_scaled_grid_problem_1d(*args):
        beta_bul = beta.conj()
        range_A = Ax_lo, Ax_hi + lambda_.to_float()
        range_B = Bx_lo, Bx_hi + lambda_.to_float()
        xs = solve_scaled_grid_problem_1d(*range_A, *range_B, k + 1)
        if xs is None:
            continue


def solve_scaled_grid_problem_1d(
    x_lo: float,
    x_hi: float,
    y_lo: float,
    y_hi: float,
    k: int,
) -> Generator[AlgebraicIntegerOverRoot2, None, None]:
    scale = RingRoot2([0, 1]) ** k
    scale_inv = AlgebraicIntegerOverRoot2(RingRoot2([1, 0]), k)
    x0 = x_lo * scale.to_float()
    x1 = x_hi * scale.to_float()

    if k % 2 == 0:
        y0 = y_lo * scale.to_float()
        y1 = y_hi * scale.to_float()
    else:
        y0 = -y_lo * scale.to_float()
        y1 = -y_hi * scale.to_float()
    
    for candidate in solve_grid_problem_1d(x0, x1, y0, y1):
        yield scale_inv * AlgebraicIntegerOverRoot2(candidate)


def solve_grid_problem_1d(
    x_lo: float,
    x_hi: float,
    y_lo: float,
    y_hi: float,
) -> Generator[RingRoot2, None, None]:
    """
    Based off gridsynth implementation.
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
    Based off gridsynth implementation.
    """
    # Compute interval widths
    dx = x_hi - x_lo
    dy = y_hi - y_lo

    lambda_ = RingRoot2([1, 1])
    lambda_inv = RingRoot2([-1, 1])
    def floor_log(x: float, b: float = lambda_.to_float()) -> tuple[int, float]:
        """
        Compute integer n such that x = r * b^n where 1 <= r < b.
        """
        if x <= 0:
            raise ValueError("x must be positive")
        elif 1 <= x < b:
            return (0, x)
        elif 1 <= x * b and x < 1:
            return (-1, b * x)
        else:
            n, r = floor_log(x, b * b)
            if r < b:
                return (2 * n, r)
            else:
                return (2 * n + 1, r / b)

    # Determine a scaling factor n so that we can approximate the width of the
    # interval dy as (1 + sqrt(2))^n
    n, _ = floor_log(abs(dy))
    if dy < 0:
        n = -n

    # TODO: Define negative power exponentiation
    if n >= 0:
        lambda_n = lambda_ ** n  # lambda ^ n
        lambda_inv_n = lambda_inv ** n  # (lambda^-1) ^ n
        lambda_bul_n = (-lambda_inv) ** n   # (-lambda^-1) ^ n
    else:
        lambda_n = lambda_inv ** -n
        lambda_inv_n = lambda_ ** -n
        lambda_bul_n = lambda_ ** -n

    if dy <= 0 and dx > 0:
        yield from gridpoints_internal(y_lo, y_hi, x_lo, x_hi, conjugate_output=True)
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


def in_epsilon_region(
    angle: float,
    epsilon: float,
    x_coordinate: AlgebraicIntegerOverRoot2,
    y_coordinate: AlgebraicIntegerOverRoot2,
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