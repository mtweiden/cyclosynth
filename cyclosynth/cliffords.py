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
H_dg = H
S_dg = S * S * S
X_dg = X
Y_dg = Y
Z_dg = Z
HX_dg = X * H
HY_dg = Y * H
HZ_dg = Z * H
SX_dg = X * S_dg
SY_dg = Y * S_dg
SZ_dg = Z * S_dg
HS_dg = S_dg * H
HSX_dg = X * S_dg * H
HSY_dg = Y * S_dg * H
HSZ_dg = Z * S_dg * H
SH_dg = H * S_dg
SHX_dg = X * H * S_dg
SHY_dg = Y * H * S_dg
SHZ_dg = Z * H * S_dg
HSH_dg = H * S_dg * H
HSHX_dg = X * H * S_dg * H
HSHY_dg = Y * H * S_dg * H
HSHZ_dg = Z * H * S_dg * H


clifford_gates_to_u2 = {
    'I': I,
    'H': H,
    'S': S,
    'X': X,
    'Y': Y,
    'Z': Z,
    'XH': HX,
    'YH': HY, 
    'ZH': HZ,
    'XS': SX,
    'YS': SY,
    'ZS': SZ,
    'SH': HS,
    'XSH': HSX,
    'YSH': HSY,
    'ZSH': HSZ,
    'HS': SH,
    'XHS': SHX,
    'YHS': SHY,
    'ZHS': SHZ,
    'HSH': HSH,
    'XHSH': HSHX,
    'YHSH': HSHY,
    'ZHSH': HSHZ,
}


clifford_gates_to_invu2 = {
    'I': I,
    'H': H,
    'S': S_dg,
    'X': X,
    'Y': Y,
    'Z': Z,
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


def match_clifford(matrix: U2Matrix) -> str | None:
    """
    Given a matrix in U(2), return the corresponding Clifford gates.

    If the matrix is not a Clifford gate, None will be returned.
    """
    candidates = [
        gates for gates, u in clifford_gates_to_u2.items()
        if matrix.hilbert_schmidt_distance(u) <= 1e-8
    ]
    if len(candidates) == 0:
        return None
    else:
        return candidates[0]


def clifford_str_to_u2(gates: str) -> U2Matrix | None:
    return clifford_gates_to_u2.get(gates)