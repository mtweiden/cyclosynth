from __future__ import annotations

from math import gcd

from mpmath import log
from mpmath import sqrt
from mpmath import mpf

from cyclosynth.algebra import AlgebraicInteger
from cyclosynth.algebra import DyadicComplexNumber
from cyclosynth.algebra import RingRoot2
from cyclosynth.algebra import RingRootRoot2Plus2
from cyclosynth.ratio import IntegerRatio
from cyclosynth.ratio import AlgebraicIntegerOverRoot2
from cyclosynth.ratio import AlgebraicIntegerOverRootRoot2Plus2


def lcm(a: int, b: int) -> int:
    """Least common multiple of `a` and `b`."""
    return abs(a * b) // gcd(abs(a), abs(b))


def log2(n: float | mpf | int) -> float:
    return log(n) / log(2)


def power_of_2(n: int) -> bool:
    return log2(n) == int(log2(n))


def is_divisible_by_rootroot2plus2(number: AlgebraicInteger) -> bool:
    """
    Divisibility test described in https://arxiv.org/abs/1501.04944.

    We want to test if beta divides alpha where both are algebraic integers.
    Assume gamma is the product of the non-trivial conjugates of an algebraic
    integer beta. If alpha is divisible by beta, then

        alpha * gamma = new_alpha * (gamma * beta)

    where (gamma * beta) is a (regular) integer, in this case 2.
    """
    # gamma = (2 - sqrt(2)) * sqrt(2 + sqrt(2))
    prod_of_conjugates = RingRootRoot2Plus2([0, 0, 2, -1])
    result = number * prod_of_conjugates
    return all(v % 2 == 0 for v in result.values)


def is_divisible_by_root2(number: AlgebraicInteger) -> bool:
    """
    Divisibility test described in https://arxiv.org/abs/1501.04944.

    We want to test if beta divides alpha where both are algebraic integers.
    Assume gamma is the product of the non-trivial conjugates of an algebraic
    integer beta. If alpha is divisible by beta, then

        alpha * gamma = new_alpha * (gamma * beta)

    where (gamma * beta) is a (regular) integer, in this case -2.
    """
    # gamma = -sqrt(2)
    prod_of_conjugates = RingRoot2([0, -1])
    result = number * prod_of_conjugates
    return all(v % 2 == 0 for v in result.values)


def discrete_sin(n: int, rr2p2: bool = False) -> IntegerRatio:
    """
    The discrete sine function defined for pi/4 and pi/8.

    Args:
        n (int): The denominator of the angle fraction, either 4 or 8.

        rr2p2 (bool): If True, then the type of the returned value will be
            AlgebraicIntegerOverRootRoot2Plus2. Otherwise, the type will be
            AlgebraicIntegerOverRoot2. If `n` is 8, the type will always be
            AlgebraicIntegerOverRootRoot2Plus2.
            (Default: False)
    """
    if n != 4 and n != 8:
        raise ValueError(f'`n` must be 4 or 8, got {n}.')
    if n == 4:
        if not rr2p2:
            numerator = RingRoot2([0, 1])
            value = AlgebraicIntegerOverRoot2(numerator, 2)
        else:
            numerator = RingRootRoot2Plus2([1, 1, 0, 0])
            value = AlgebraicIntegerOverRoot2(numerator, 2)
    else:
        numerator = RingRootRoot2Plus2([1, 1, 0, 0])
        value = AlgebraicIntegerOverRootRoot2Plus2(numerator, 3)
    return value


def discrete_cos(n: int, rr2p2: bool = False) -> IntegerRatio:
    """
    The discrete cosine function defined for pi/4 and pi/8.

    Args:
        n (int): The denominator of the angle fraction, either 4 or 8.

        rr2p2 (bool): If True, then the type of the returned value will be
            AlgebraicIntegerOverRootRoot2Plus2. Otherwise, the type will be
            AlgebraicIntegerOverRoot2. If `n` is 8, the type will always be
            AlgebraicIntegerOverRootRoot2Plus2.
            (Default: False)
    """
    if n != 4 and n != 8:
        raise ValueError(f'`n` must be 4 or 8, got {n}.')
    if n == 4:
        if not rr2p2:
            numerator = RingRoot2([0, 1])
            value = AlgebraicIntegerOverRoot2(numerator, 2)
        else:
            numerator = RingRootRoot2Plus2([1, 1, 0, 0])
            value = AlgebraicIntegerOverRoot2(numerator, 2)
    else:
        numerator = RingRootRoot2Plus2([3, 2, 0, 0])
        value = AlgebraicIntegerOverRootRoot2Plus2(numerator, 3)
    return value


def dyadic_sin(k: int, n: int) -> DyadicComplexNumber:
    """
    The dyadic sine function defined for pi/4 and pi/8.

    Args:
        k (int): The numerator of the angle fraction.

        n (int): The denominator of the angle fraction, either 4 or 8.
    """
    if power_of_2(n):
        raise ValueError(f'`n` must be a power of 2, got {n}.')
    k = k % (2 * n)
    if k == (n // 2):
        return DyadicComplexNumber([1] + [0] * (n - 1), 0)
    elif k == (3 * n // 2):
        return DyadicComplexNumber([-1] + [0] * (n - 1), 0)
    values = [0] * n
    k_1 = (n // 2 - k)
    k_2 = (n // 2 + k)
    sign_1 = (-1) ** (k_1 < 0) * (-1) ** ((k - n // 2) > n)
    sign_2 = (-1) ** (k_2 > n) * (-1) ** ((k - n // 2) > n)
    values[k_1 % n] += sign_1
    values[k_2 % n] -= sign_2
    return DyadicComplexNumber(values, 1)


def dyadic_cos(k: int, n: int) -> DyadicComplexNumber:
    """
    The dyadic cosine function defined for pi/4 and pi/8.

    Args:
        k (int): The numerator of the angle fraction.

        n (int): The denominator of the angle fraction, either 4 or 8.
    """
    if power_of_2(n):
        raise ValueError(f'`n` must be a power of 2, got {n}.')
    k = k % (2 * n)
    if k == 0:
        return DyadicComplexNumber([1] + [0] * (n - 1), 0)
    elif k == n:
        return DyadicComplexNumber([-1] + [0] * (n - 1), 0)
    k_1 = k % n
    k_2 = (n - k) % n
    values = [0] * n
    sign_1 = (-1) ** (k > n)
    sign_2 = (-1) ** (k > n)
    values[k_1 % n] += sign_1
    values[k_2 % n] -= sign_2
    return DyadicComplexNumber(values, 1)


def floor_log(x: float, b: float = 1 + sqrt(2)) -> tuple[int, float]:
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


def widen_interval(a: float, b: float, eps: float = 1e-8) -> tuple[float, float]:
    """
    Add a small epsilon to the interval [a, b].
    """
    return (a - eps, b + eps)