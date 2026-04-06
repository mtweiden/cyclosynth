"""A module for doing algebraic integer enumeration."""
from typing import Iterator

import math

from algebra import DyadicComplexNumber


# Enumerate integers in the convex shell y \in Z^4 s.t. 2^{k-2} <= ||y||^2 <= 2^{k-1}
# Compute x = (a, b, c, d, -a, b, -c, d)
# Compute (u, ubul) = 2^{-k/2} * Sigma @ x
# Filter
#  1. ||u|| = 1
#  2. ||ubul|| <= 1
#  3. |<u,v>| >= sqrt(1 - epsilon^2)
def _centered_sequence(max_value: int) -> Iterator[int]:
    """Yield integers from -max_value to max_value."""
    yield 0
    for i in range(1, max_value + 1):
        yield i
        yield -i


def enumerate_integers(k: int) -> Iterator[tuple[int, int, int, int]]:
    """
    Enumerate 4-tuples y = (a, b, c, d) that sastisfy 2^{k-2} <= ||y|| <= 2^{k-1}
    """
    if k < 0:
        return
    lower = 1 << max(0, k - 2)  # 2^{k-2}
    upper = 1 << max(0, k - 1)  # 2^{k-1}

    def recurse(depth: int, s: int, acc: list[int]) -> Iterator[tuple[int, ...]]:
        if depth == 4:
            if lower <= s <= upper:
                yield tuple(acc)
            return
        max_allowed = math.isqrt(upper - s)
        for t in _centered_sequence(max_allowed):
            s2 = s + t * t
            # s2 is guaranteed to be <= upper due to max_allowed
            acc.append(t)
            yield from recurse(depth + 1, s2, acc)
            acc.pop()

    yield from recurse(0, 0, [])  # type: ignore



if __name__ == "__main__":
    k = 5 # k = 11 ~ 1 second
    for t in enumerate_integers(k):
        # Compute u and ubul
        u = DyadicComplexNumber(t, k // 2)
        print(u.__repr__())
        print(u.abs())
        # Check that ||u|| = 1
