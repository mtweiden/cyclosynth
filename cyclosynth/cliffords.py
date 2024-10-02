"""A module that defines Clifford gates and operations involving them."""

from cyclosynth.algebra import DyadicComplexNumber
from cyclosynth.algebra import RingRoot2
from cyclosynth.ratio import AlgebraicIntegerOverRoot2
from cyclosynth.matrix import SO3Matrix
from cyclosynth.matrix import U2Matrix


zero = DyadicComplexNumber([0, 0, 0, 0, 0, 0, 0, 0], 0)
one = DyadicComplexNumber([1, 0, 0, 0, 0, 0, 0, 0], 0)
half_one = DyadicComplexNumber([1, 0, 0, 0, 0, 0, 0, 0], 1)
imag = DyadicComplexNumber([0, 0, 0, 0, 1, 0, 0, 0], 0)
half_imag = DyadicComplexNumber([0, 0, 0, 0, 1, 0, 0, 0], 1)
one_over_sqrt2 = DyadicComplexNumber([0, 0, 1, 0, 0, 0 , -1, 0], 1)


I = U2Matrix([one, zero, zero, one])
H = U2Matrix([one_over_sqrt2, one_over_sqrt2, one_over_sqrt2, -one_over_sqrt2])
S = U2Matrix([one, zero, zero, imag])
X = U2Matrix([zero, one, one, zero])
Y = U2Matrix([zero, -imag, imag, zero])
Z = U2Matrix([one, zero, zero, -one])
HX = H * X
HY = H * Y
HZ = H * Z
SX = S * X
SY = S * Y
SZ = S * Z
HS = H * S
HSX = H * S * X
HSY = H * S * Y
HSZ = H * S * Z
SH = S * H
SHX = S * H * X
SHY = S * H * Y
SHZ = S * H * Z
HSH = H * S * H
HSHX = H * S * H * X
HSHY = H * S * H * Y
HSHZ = H * S * H * Z


clifford_keys = {
    (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0): 'I',
    (0.0, 0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, 0.0): 'H',
    (0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0): 'S',
    (1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0): 'X',
    (-1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0): 'Y',
    (-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0): 'Z',
    (0.0, 0.0, 1.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0): 'HX',
    (0.0, 0.0, -1.0, 0.0, -1.0, 0.0, -1.0, 0.0, 0.0): 'HY',
    (0.0, 0.0, -1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0): 'HZ',
    (0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0): 'SX',
    (0.0, -1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, -1.0): 'SY',
    (0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0): 'SZ',
    (0.0, -1.0, 0.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0): 'HS',
    (0.0, -1.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0): 'HSX',
    (0.0, 1.0, 0.0, 0.0, 0.0, -1.0, -1.0, 0.0, 0.0): 'HSY',
    (0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0): 'HSZ',
    (0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0): 'SH',
    (0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, -1.0, 0.0): 'SHX',
    (0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0, -1.0, 0.0): 'SHY',
    (0.0, 0.0, -1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0): 'SHZ',
    (1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0): 'HSH',
    (1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0): 'HSHX',
    (-1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0): 'HSHY',
    (-1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, -1.0, 0.0): 'HSHZ',
}


def cliffords(as_u2: bool = False) -> list[U2Matrix | SO3Matrix]:
    """
    Returns a list of single qubit Clifford group elements.

    Args:
        as_u2 (bool): If True, return the Clifford gates as U(2) matrices.
            Otherwise, return them as SO(3) matrices.

    Note:
        Clifford group is all C = AB
        A in {I, H, S, HS, SH, HSH}
        B in {I, X, Y, Z}
    """
    if as_u2:
        cliffords = [
            I, H, S, X, Y, Z, HX, HY, HZ, SX, SY, SZ, HS, HSX,
            HSY, HSZ, SH, SHX, SHY, SHZ, HSH, HSHX, HSHY, HSHZ,
        ]
    else:
        p = AlgebraicIntegerOverRoot2(RingRoot2([1, 0]), 0)
        m = AlgebraicIntegerOverRoot2(RingRoot2([-1, 0]), 0)
        z = AlgebraicIntegerOverRoot2(RingRoot2([0, 0]), 0)
        cliffords = [
            SO3Matrix([p, z, z, z, p, z, z, z, p]),
            SO3Matrix([z, z, p, z, m, z, p, z, z]),
            SO3Matrix([z, p, z, m, z, z, z, z, p]),
            SO3Matrix([p, z, z, z, m, z, z, z, m]),
            SO3Matrix([m, z, z, z, p, z, z, z, m]),
            SO3Matrix([m, z, z, z, m, z, z, z, p]),
            SO3Matrix([z, z, p, z, p, z, m, z, z]),
            SO3Matrix([z, z, m, z, m, z, m, z, z]),
            SO3Matrix([z, z, m, z, p, z, p, z, z]),
            SO3Matrix([z, p, z, p, z, z, z, z, m]),
            SO3Matrix([z, m, z, m, z, z, z, z, m]),
            SO3Matrix([z, m, z, p, z, z, z, z, p]),
            SO3Matrix([z, m, z, z, z, m, p, z, z]),
            SO3Matrix([z, m, z, z, z, p, m, z, z]),
            SO3Matrix([z, p, z, z, z, m, m, z, z]),
            SO3Matrix([z, p, z, z, z, p, p, z, z]),
            SO3Matrix([z, z, p, p, z, z, z, p, z]),
            SO3Matrix([z, z, p, m, z, z, z, m, z]),
            SO3Matrix([z, z, m, p, z, z, z, m, z]),
            SO3Matrix([z, z, m, m, z, z, z, p, z]),
            SO3Matrix([p, z, z, z, z, p, z, m, z]),
            SO3Matrix([p, z, z, z, z, m, z, p, z]),
            SO3Matrix([m, z, z, z, z, p, z, p, z]),
            SO3Matrix([m, z, z, z, z, m, z, m, z]),
        ]
    return cliffords


def match_clifford(matrix: SO3Matrix) -> str | None:
    """
    Given a matrix in SO(3), return the corresponding Clifford gate.

    If the matrix is not a Clifford gate, None will be returned.
    """
    key = tuple([round(x, 3) for x in matrix.to_float()])
    return clifford_keys.get(key)