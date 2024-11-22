"""Make an ellipse and the unit disc simultaneously 1/6-upright."""
from typing import Sequence

from mpmath import mp
from mpmath import sqrt
from mpmath import log
from mpmath import ceil
from mpmath import floor

from numpy import isclose

from cyclosynth.algebra import RingRoot2
from cyclosynth.ratio import AlgebraicIntegerOverRoot2
from cyclosynth.ellipse import Ellipse
from cyclosynth.matrix import Operator

mp.dps = 100


# Useful constants
root2 = sqrt(2)
root2_inv = 1 / root2
lamb = 1 + root2
lamb_inv = root2 - 1
root_lamb_inv = sqrt(lamb_inv)

lambda_ = AlgebraicIntegerOverRoot2(RingRoot2([1, 1]))
lambda_inv = AlgebraicIntegerOverRoot2(RingRoot2([-1, 1]))
lambda_bul = AlgebraicIntegerOverRoot2(RingRoot2([1, -1]))
lambda_inv_bul = AlgebraicIntegerOverRoot2(RingRoot2([-1, -1]))
roothalf = AlgebraicIntegerOverRoot2(RingRoot2([1, 0]), 1)
zero = AlgebraicIntegerOverRoot2(RingRoot2([0, 0]))
one = AlgebraicIntegerOverRoot2(RingRoot2([1, 0]))
root = AlgebraicIntegerOverRoot2(RingRoot2([0, 1]))
two = AlgebraicIntegerOverRoot2(RingRoot2([2, 0]))

identity_op = Operator((one, zero, zero, one))
op_Z = Operator((one, zero, zero, -one))
op_X = Operator((zero, one, one, zero))
op_R = Operator((one, -one, one, one)) * roothalf
op_K = Operator((-lambda_inv, -one, lambda_, one)) * roothalf
op_S = Operator((lambda_, zero, zero, lambda_inv))
op_S_inv = Operator((lambda_inv, zero, zero, lambda_))


def round(number: float) -> int:
    upper, lower = int(ceil(number)), int(floor(number))
    if upper - number < number - lower:
        return upper
    return lower


def apply_op(ell: Ellipse, operator: Operator) -> Ellipse:
    """ G^{dagger} @ D @ G """
    a, b, c, d = [x.to_float() for x in operator.values]
    x, y, z = ell.a, ell.b, ell.d
    aa = a * (a * x + c * y) + c * (a * y + c * z)
    bb = b * (a * x + c * y) + d * (a * y + c * z)
    dd = b * (b * x + d * y) + d * (b * y + d * z)
    return Ellipse([aa, bb, dd], ell.center)


def combine_ops(op1: Operator, op2: Operator) -> Operator:
    return op1 * op2


def skew(ell1: Ellipse, ell2: Ellipse) -> float:
    return ell1.skew() + ell2.skew()


def ellipse_to_bl2z(ell: Ellipse) -> tuple[float, float]:
    b = ell.b
    l2z = ell.d / ell.a
    return b, l2z


def ellipse_to_bz(ell: Ellipse) -> tuple[float, float]:
    b, l2z = ellipse_to_bl2z(ell)
    z = log(l2z) / (2 * log(lamb))
    return b, z


def bias(ell1: Ellipse, ell2: Ellipse) -> float:
    _, z = ellipse_to_bz(ell1)
    _, zeta = ellipse_to_bz(ell2)
    return zeta - z


# The shift lemma
def shift_k_ellipses(
    ell1: Ellipse,
    ell2: Ellipse,
    k: int,
) -> tuple[Ellipse, Ellipse]:
    """
    Given state (D, Delta) such that bias(D, Delta) > 1, shift the state so
    that bias(D', Delta') <= 1.

    Args:
        ell1 (Ellipse): The first ellipse. This is associated with the epsilon
            region.

        ell2 (Ellipse): The second ellipse. This is associated with the unit
            disc in R^2.

        k (int): The shift amount.
    
    Returns:
        (Ellipse): The shifted epsilon region ellipse.

        (Ellipse): The shifted unit disc ellipse.
    """
    def apply_sigma_k(ell: Ellipse) -> Ellipse:
        new_a = ell.a * lamb ** k
        new_d = ell.d * lamb ** -k
        return Ellipse([new_a, ell.b, new_d], ell.center)

    def apply_tau_k(ell: Ellipse) -> Ellipse:
        new_a = ell.a * lamb ** -k
        sign = 1 if k % 2 == 0 else -1
        new_b = sign * ell.b
        new_d = ell.d * lamb ** k
        return Ellipse([new_a, new_b, new_d], ell.center)
    
    return apply_sigma_k(ell1), apply_tau_k(ell2)


def shift_sigma_k(op: Operator, k: int) -> Operator:
    if k >= 0:
        lk = RingRoot2([1, 1]) ** k
        lik = RingRoot2([-1, 1]) ** k
    else:
        lk = RingRoot2([-1, 1]) ** -k
        lik = RingRoot2([1, 1]) ** -k
    lk = AlgebraicIntegerOverRoot2(lk)  # lambda^k
    lik = AlgebraicIntegerOverRoot2(lik)  # lambda^{-k}
    a1, b1, c1, d1 = op.values
    new_op = Operator((lk * a1, b1, c1, lik * d1))
    return new_op


# The R lemma
def R_lemma(
    ell1: Ellipse,
    ell2: Ellipse,
) -> tuple[Ellipse, Ellipse, Operator, Operator]:
    """
    If skew(D, Delta) >= 15, and  -0.8 <= z, zeta <= 0.8, then
    skew(R(D, Delta)) <= 0.9 * skew(D, Delta)
    
    Args:
        ell1 (Ellipse): The first ellipse. This is associated with the epsilon
            region.

        ell2 (Ellipse): The second ellipse. This is associated with the unit
            disc in R^2.
    
    Returns:
        (Ellipse): The R-transformed epsilon region ellipse.

        (Ellipse): The R-transformed unit disc ellipse.

        (Operator): The R operator for the first ellipse.

        (Operator): The R_bul operator for the second ellipse.
    """
    r = roothalf
    def apply_and_compute_R(ell: Ellipse) -> tuple[Ellipse, Operator]:
        op = Operator((r, -r, r, r))
        ell_R = apply_op(ell, op)
        return ell_R, op
    
    def apply_and_compute_R_bul(ell: Ellipse) -> tuple[Ellipse, Operator]:
        op = Operator((-r, r, -r, -r))
        ell_R_bul = apply_op(ell, op)
        return ell_R_bul, op
    
    ell1_R, op_R = apply_and_compute_R(ell1)
    ell2_R, op_R_bul = apply_and_compute_R_bul(ell2)
    return ell1_R, ell2_R, op_R, op_R_bul


# The K lemma
def K_lemma(
    ell1: Ellipse,
    ell2: Ellipse,
    bullet: bool = False,
) -> tuple[Ellipse, Ellipse, Operator, Operator]:
    """
    If skew(D, Delta) >= 15, bias(D, Delta) in [-1, 1] and b, beta >= 0,
    z <= 0.3, zeta >= 0.8, then skew(K(D, Delta)) <= 0.9 * skew(D, Delta).
    
    Args:
        ell1 (Ellipse): The first ellipse. This is associated with the epsilon
            region.

        ell2 (Ellipse): The second ellipse. This is associated with the unit
            disc in R^2.
        
        bullet (bool): If True, apply the K_bul operator instead of K.
    
    Returns:
        (Ellipse): The R-transformed epsilon region ellipse.

        (Ellipse): The R-transformed unit disc ellipse.

        (Operator): The K operator for the first ellipse.

        (Operator): The K_bul operator for the second ellipse.
    """
    assert ell1.b >= 0 and ell2.b >= 0
    r = roothalf

    def apply_and_compute_K(ell: Ellipse) -> tuple[Ellipse, Operator]:
        op = Operator(
            (-lambda_inv * roothalf, -roothalf, lambda_ * roothalf, roothalf)
        )
        ell_K = apply_op(ell, op)
        return ell_K, op
    
    def apply_and_compute_K_bul(ell: Ellipse) -> tuple[Ellipse, Operator]:
        op = Operator((lambda_ * r, -r, -lambda_inv * r, r))
        ell_K = apply_op(ell, op)
        return ell_K, op
    
    if not bullet:
        ell1_K, op_K = apply_and_compute_K(ell1)
        ell2_K, op_K_bul = apply_and_compute_K_bul(ell2)
    else:
        ell1_K, op_K = apply_and_compute_K_bul(ell1)
        ell2_K, op_K_bul = apply_and_compute_K(ell2)
    return ell1_K, ell2_K, op_K, op_K_bul


# The A lemma
def A_lemma(
    ell1: Ellipse,
    ell2: Ellipse,
) -> tuple[Ellipse, Ellipse, Operator, Operator]:
    """
    If skew(D, Delta) >= 15, bias(D, Delta) in [-1, 1] and b, beta >= 0,
    z, zeta >= 0.3, then there exists an n such that
    skew(A^n(D, Delta)) <= 0.9 * skew(D, Delta).
    
    Args:
        ell1 (Ellipse): The first ellipse. This is associated with the epsilon
            region.

        ell2 (Ellipse): The second ellipse. This is associated with the unit
            disc in R^2.
    
    Returns:
        (Ellipse): The R-transformed epsilon region ellipse.

        (Ellipse): The R-transformed unit disc ellipse.

        (Operator): The A operator for the first ellipse.

        (Operator): The A_bul operator for the second ellipse.
    """
    assert ell1.b >= 0 and ell2.b >= 0
    (_, z), (_, zeta) = ellipse_to_bz(ell1), ellipse_to_bz(ell2)
    c = min(z, zeta)
    x = int(floor(lamb ** c / 2))
    n = max(1, x)

    def apply_and_compute_A(ell: Ellipse) -> tuple[Ellipse, Operator]:
        y = AlgebraicIntegerOverRoot2(RingRoot2([-2 * n, 0]))
        op = Operator((one, y, zero, one))
        ell_A = apply_op(ell, op)
        return ell_A, op
    
    def apply_and_compute_A_bul(ell: Ellipse) -> tuple[Ellipse, tuple[float]]:
        return apply_and_compute_A(ell)
    
    ell1_A, op_A = apply_and_compute_A(ell1)
    ell2_A_bul, op_A_bul = apply_and_compute_A_bul(ell2)
    return ell1_A, ell2_A_bul, op_A, op_A_bul


# The B lemma
def B_lemma(
    ell1: Ellipse,
    ell2: Ellipse,
) -> tuple[Ellipse, Ellipse, Operator, Operator]:
    """
    If skew(D, Delta) >= 15, bias(D, Delta) in [-1, 1] and b, beta >= 0,
    z, zeta >= -0.2, then there exists an n such that
    skew(B^n(D, Delta)) <= 0.9 * skew(D, Delta).
    
    Args:
        ell1 (Ellipse): The first ellipse. This is associated with the epsilon
            region.

        ell2 (Ellipse): The second ellipse. This is associated with the unit
            disc in R^2.

    Returns:
        (Ellipse): The R-transformed epsilon region ellipse.

        (Ellipse): The R-transformed unit disc ellipse.

        (Operator): The A operator for the first ellipse.

        (Operator): The B_bul operator for the second ellipse.
    """
    assert ell1.b <= 0 and 0 <= ell2.b
    (_, z), (_, zeta) = ellipse_to_bz(ell1), ellipse_to_bz(ell2)
    c = min(z, zeta)
    x = int(floor(lamb ** c * root2_inv))
    n = max(1, x)
    def apply_and_compute_B(ell: Ellipse) -> tuple[Ellipse, Operator]:
        y = AlgebraicIntegerOverRoot2(RingRoot2([0, n]))
        op = Operator((one, y, zero, one))
        ell_B = apply_op(ell, op)
        return ell_B, op
    
    def apply_and_compute_B_bul(ell: Ellipse) -> tuple[Ellipse, Operator]:
        y = AlgebraicIntegerOverRoot2(RingRoot2([0, -n]))
        op = Operator((one, y, zero, one))
        ell_B_bul = apply_op(ell, op)
        return ell_B_bul, op
    
    ell1_B, op_B = apply_and_compute_B(ell1)
    ell2_B_bul, op_B_bul = apply_and_compute_B_bul(ell2)
    return ell1_B, ell2_B_bul, op_B, op_B_bul


def op_S_k_operator(bias: float) -> Operator:
    if abs(bias) < 2:
        return identity_op
    x = (1 - bias) / 4
    if x >= 0:
        k = int(ceil(x))
        a = RingRoot2([1, 1]) ** k
        b = RingRoot2([-1, 1]) ** k
    else:
        k = int(floor(x))
        a = RingRoot2([-1, 1]) ** -k
        b = RingRoot2([1, 1]) ** -k
    a = AlgebraicIntegerOverRoot2(a)
    b = AlgebraicIntegerOverRoot2(b)
    op = Operator((a, zero, zero, b))
    return op


def compute_shift_amount(ell1: Ellipse, ell2: Ellipse) -> int:
    bias_ = float(bias(ell1, ell2))
    if abs(bias_) <= 1:
        return 0
    x = (1 - bias_) / 2
    k = round(x)
    if x >= 0 and k == 0:
        return 1
    elif x < 0 and k == 0:
        return -1
    return k


def in_R_region(z: float, zeta: float) -> bool:
    return abs(z) <= 0.8 and abs(zeta) <= 0.8


def in_B_region(b: float, beta: float) -> bool:
    return b <= 0 and 0 <= beta


def step_lemma(enclosing_ellipse: Ellipse, unit_disc: Ellipse) -> Operator | None:
    """
    A recursive implementation of the step lemma.

    Ellipse ell1 corresponds to the the epsilon region, and ell2 corresponds to
    the unit disc in R^2.
    """
    def wlog_using(ell1: Ellipse, ell2: Ellipse, op: Operator) -> Operator | None:
        # Apply the operator to ell1 and ell2
        ella, ellb = apply_op(ell1, op), apply_op(ell2, op.conj())
        # Recursively call step_lemma on the transformed ellipses
        op2 = step_lemma(ella, ellb)
        # Combine the current operator with the recursive result
        if op2 is None:
            return op
        return op * op2
    
    def with_shift(ell1: Ellipse, ell2: Ellipse,  k: int) -> Operator | None:
        # Apply the operator to ell1 and ell2
        ella, ellb = shift_k_ellipses(ell1, ell2, k)
        # Recursively call step_lemma on the transformed ellipses
        op2 = step_lemma(ella, ellb)
        # Combine the current operator with the recursive result
        if op2 is None:
            return None
        return shift_sigma_k(op2, k)

    # Compute case deciding variables
    b, z = ellipse_to_bz(enclosing_ellipse)
    beta, zeta = ellipse_to_bz(unit_disc)

    if skew(enclosing_ellipse, unit_disc) <= 15:
        return None
    
    bias_ = bias(enclosing_ellipse, unit_disc)

    if beta < 0:
        return wlog_using(enclosing_ellipse, unit_disc, op_Z)
    elif z + zeta < 0:
        return wlog_using(enclosing_ellipse, unit_disc, op_X)
    elif abs(bias_) > 2:
        op_S_k = op_S_k_operator(bias_)
        return wlog_using(enclosing_ellipse, unit_disc, op_S_k)
    elif abs(bias_) > 1:
        k = compute_shift_amount(enclosing_ellipse, unit_disc)
        return with_shift(enclosing_ellipse, unit_disc, k)
    # Case 1.1 and 2.1 - apply R
    elif abs(z) <= 0.8 and abs(zeta) <= 0.8:
        return op_R
    # Case 1.2 - apply K
    elif 0 <= b and z <= 0.3 and 0.8 <= zeta:
        return op_K
    # Case 1.3 - apply A
    elif 0 <= b and 0.3 <= z and 0.3 <= zeta:
        c = min(z, zeta)
        x = int(floor(lamb ** c / 2))
        n = max(1, x)
        y = AlgebraicIntegerOverRoot2(RingRoot2([-2 * n, 0]))
        op_A = Operator((one, y, zero, one))
        return op_A
    # Case 1.4 - apply K_bul
    elif 0 <= b and 0.8 <= z and zeta <= 0.3:
        return op_K.conj()
    # Case 2.2 - apply B
    elif b < 0 and -0.2 <= z and -0.2 <= zeta:
        c = min(z, zeta)
        x = int(floor(lamb ** c * root2_inv))
        n = max(1, x)
        y = AlgebraicIntegerOverRoot2(RingRoot2([0, n]))
        op_B = Operator((one, y, zero, one))
        return op_B
    else:
        raise RuntimeError('No case matched')
    

def reduce_normalized_ellipses(
    enclosing_ellipse: Ellipse,
    unit_disc: Ellipse,
) -> Operator:
    """
    Reduce the given ellipse so that it and the transformed unit disc are
    both at least 1/6 upright.

    Args:
        enclosing_ellipse (Ellipse): An ellipse enclosing the epsilon region.

        unit_disc (Ellipse): An ellipse representing the unit disc in R^2.
    
    Returns:
        (Operator): The total grid operator.
    """
    op = step_lemma(enclosing_ellipse, unit_disc)
    if op is None:
        return identity_op
    new_ellipse = apply_op(enclosing_ellipse, op)
    new_disc = apply_op(unit_disc, op.conj())
    return op * reduce_normalized_ellipses(new_ellipse, new_disc)


def reduce(enclosing_ellipse: Ellipse) -> Operator:
    """
    Reduce the given ellipse so that it and the transformed unit disc are
    both at least 1/6 upright.

    Args:
        enclosing_ellipse (Ellipse): An ellipse enclosing the epsilon region.
    
    Returns:
        (Operator): A grid operator that makes the ellipse and the unit disc
            simultaneously 1/6-upright.
    """
    norm = sqrt(enclosing_ellipse.det())
    enc_ell = Ellipse(
        [
            enclosing_ellipse.a / norm,
            enclosing_ellipse.b / norm, 
            enclosing_ellipse.d / norm,
        ],
        enclosing_ellipse.center,
    )
    unit_disc = Ellipse([1, 0, 1], (0, 0))
    assert isclose(abs(float(enc_ell.det())), 1.0)
    assert isclose(abs(float(unit_disc.det())), 1.0)
    return reduce_normalized_ellipses(enc_ell, unit_disc)