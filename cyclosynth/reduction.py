"""Make an ellipse and the unit disc simultaneously 1/6-upright."""
from typing import Sequence

from mpmath import mp
from mpmath import sqrt
from mpmath import log
from mpmath import ceil
from mpmath import floor

from cyclosynth.ellipse import Ellipse

mp.dps = 100


# Useful constants
root2 = sqrt(2)
root2_inv = 1 / root2
lamb = 1 + root2
lamb_inv = root2 - 1
root_lamb_inv = sqrt(lamb_inv)

def apply_op(ell: Ellipse, operator: Sequence[float]) -> Ellipse:
    """ G^{dagger} @ D @ G """
    a, b, c, d = operator
    x, y, z = ell.a, ell.b, ell.d
    aa = a * (a * x + c * y) + c * (a * y + c * z)
    bb = b * (a * x + c * y) + d * (a * y + c * z)
    dd = b * (b * x + d * y) + d * (b * y + d * z)
    return Ellipse([aa, bb, dd], ell.center)

def combine_ops(op1: Sequence[float], op2: Sequence[float]) -> tuple[float]:
    a1, b1, c1, d1 = op1
    a2, b2, c2, d2 = op2
    a = a1 * a2 + b1 * c2
    b = a1 * b2 + b1 * d2
    c = c1 * a2 + d1 * c2
    d = c1 * b2 + d1 * d2
    return a, b, c, d

# A state is a pair of symmetric positive definite matrices of determinant 1.
# Given state (D, Delta) with
#     D = [[e * lambda^{-z}, b], [b, e * lambda^z]]
#     Delta = [[ee * lambda^{-zeta}, beta], [beta, lambda^{zeta}]]
# where labmda = 1 + √2. We define skew(D, Delta) = b^2 + beta^2, and 
# bias(D, Delta) = zeta - z. The value skew(D, Delta) is small when both
# ellipses are upright.
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


# The action of a grid operator on a state is defined as:
#    (D, Delta) -G-> (G^{\\dagger} @ D @ G , G_bul^{\\dagger} @ Delta @ G_bul)
# where G_bul is the elementwise Galois conjugate of G.


# The step lemma
# Finds grid operator G such that G(D, Delta) has skew <= 15.
# We associate with each state (D, Delta) a pair (z, zeta). The step lemma
# proceeds by associating actions with pairs (z, zeta).

# The shift lemma
def shift_k(
    ell1: Ellipse,
    ell2: Ellipse,
    k: int,
    return_operators: bool = True,
) -> tuple[Ellipse] | tuple[Ellipse, Ellipse, tuple[float], tuple[float]]:
    """
    Given state (D, Delta) such that bias(D, Delta) > 1, shift the state so
    that bias(D', Delta') <= 1.

    Args:
        ell1 (Ellipse): The first ellipse. This is associated with the epsilon
            region.

        ell2 (Ellipse): The second ellipse. This is associated with the unit
            disc in R^2.

        k (int): The shift amount.

        return_operators (bool): If True, return the shift operators for the
            first and second ellipses. (Default: False)
    
    Returns:
        (Ellipse): The shifted epsilon region ellipse.

        (Ellipse): The shifted unit disc ellipse.

        (Optional[tuple[float]]): The shift operator for the first ellipse.
            If return_operators is False, this is not returned.

        (Optional[tuple[float]]): The shift operator for the second ellipse.
            If return_operators is False, this is not returned.
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
    
    def compute_sigma_tau_k() -> tuple[tuple[float]]:
        factor = root_lamb_inv ** k
        lambk = lamb ** k
        sign = 1 if k % 2 == 0 else -1
        lambk_factor = factor * lambk
        sig = tuple([lambk_factor, 0, 0, factor])
        tau = tuple([factor, 0, 0, sign * lambk_factor])
        return sig, tau
    
    if return_operators:
        sig, tau = compute_sigma_tau_k()
        return apply_sigma_k(ell1), apply_tau_k(ell2), sig, tau
    return apply_sigma_k(ell1), apply_tau_k(ell2)


# The R lemma
def R_lemma(
    ell1: Ellipse,
    ell2: Ellipse,
    return_operators: bool = True,
) -> tuple[Ellipse] | tuple[Ellipse, Ellipse, tuple[float], tuple[float]]:
    """
    If skew(D, Delta) >= 15, and  -0.8 <= z, zeta <= 0.8, then
    skew(R(D, Delta)) <= 0.9 * skew(D, Delta)
    
    Args:
        ell1 (Ellipse): The first ellipse. This is associated with the epsilon
            region.

        ell2 (Ellipse): The second ellipse. This is associated with the unit
            disc in R^2.

        return_operators (bool): If True, return the R operators for the
            first and second ellipses. (Default: False)
    
    Returns:
        (Ellipse): The R-transformed epsilon region ellipse.

        (Ellipse): The R-transformed unit disc ellipse.

        (Optional[tuple[float]]): The R operator for the first ellipse.
            If return_operators is False, this is not returned.

        (Optional[tuple[float]]): The R_bul operator for the second ellipse.
            If return_operators is False, this is not
    """
    r = root2_inv
    def apply_and_compute_R(ell: Ellipse) -> tuple[Ellipse, tuple[float]]:
        op = (r, -r, r, r)
        ell_R = apply_op(ell, op)
        return ell_R, op
    
    def apply_and_compute_R_bul(ell: Ellipse) -> tuple[Ellipse, tuple[float]]:
        op = (-r, r, -r, -r)
        ell_R_bul = apply_op(ell, op)
        return ell_R_bul, op
    
    ell1_R, op_R = apply_and_compute_R(ell1)
    ell2_R, op_R_bul = apply_and_compute_R_bul(ell2)
    if return_operators:
        return ell1_R, ell2_R, op_R, op_R_bul
    return ell1_R, ell2_R


# The K lemma
def K_lemma(
    ell1: Ellipse,
    ell2: Ellipse,
    return_operators: bool = True,
    bullet: bool = False,
) -> tuple[Ellipse] | tuple[Ellipse, Ellipse, tuple[float], tuple[float]]:
    """
    If skew(D, Delta) >= 15, bias(D, Delta) in [-1, 1] and b, beta >= 0,
    z <= 0.3, zeta >= 0.8, then skew(K(D, Delta)) <= 0.9 * skew(D, Delta).
    
    Args:
        ell1 (Ellipse): The first ellipse. This is associated with the epsilon
            region.

        ell2 (Ellipse): The second ellipse. This is associated with the unit
            disc in R^2.

        return_operators (bool): If True, return the R operators for the
            first and second ellipses. (Default: False)
        
        bullet (bool): If True, apply the K_bul operator instead of K.
    
    Returns:
        (Ellipse): The R-transformed epsilon region ellipse.

        (Ellipse): The R-transformed unit disc ellipse.

        (Optional[tuple[float]]): The K operator for the first ellipse.
            If return_operators is False, this is not returned.

        (Optional[tuple[float]]): The K_bul operator for the second ellipse.
            If return_operators is False, this is not
    """
    assert ell1.b >= 0 and ell2.b >= 0
    r = root2_inv

    def apply_and_compute_K(ell: Ellipse) -> tuple[Ellipse, tuple[float]]:
        op = (-lamb_inv * r, -r, lamb * r, r)
        ell_K = apply_op(ell, op)
        return ell_K, op
    
    def apply_and_compute_K_bul(ell: Ellipse) -> tuple[Ellipse, tuple[float]]:
        op = (lamb * r, -r, -lamb_inv * r, r)
        ell_K = apply_op(ell, op)
        return ell_K, op
    
    if not bullet:
        ell1_K, op_K = apply_and_compute_K(ell1)
        ell2_K, op_K_bul = apply_and_compute_K_bul(ell2)
    else:
        ell1_K, op_K = apply_and_compute_K_bul(ell1)
        ell2_K, op_K_bul = apply_and_compute_K(ell2)
    if return_operators:
        return ell1_K, ell2_K, op_K, op_K_bul
    return ell1_K, ell2_K


# The A lemma
def A_lemma(
    ell1: Ellipse,
    ell2: Ellipse,
    return_operators: bool = True,
) -> tuple[Ellipse] | tuple[Ellipse, Ellipse, tuple[float], tuple[float]]:
    """
    If skew(D, Delta) >= 15, bias(D, Delta) in [-1, 1] and b, beta >= 0,
    z, zeta >= 0.3, then there exists an n such that
    skew(A^n(D, Delta)) <= 0.9 * skew(D, Delta).
    
    Args:
        ell1 (Ellipse): The first ellipse. This is associated with the epsilon
            region.

        ell2 (Ellipse): The second ellipse. This is associated with the unit
            disc in R^2.

        return_operators (bool): If True, return the R operators for the
            first and second ellipses. (Default: False)
    
    Returns:
        (Ellipse): The R-transformed epsilon region ellipse.

        (Ellipse): The R-transformed unit disc ellipse.

        (Optional[tuple[float]]): The A operator for the first ellipse.
            If return_operators is False, this is not returned.

        (Optional[tuple[float]]): The A_bul operator for the second ellipse.
            If return_operators is False, this is not
    """
    assert ell1.b >= 0 and ell2.b >= 0
    (_, z), (_, zeta) = ellipse_to_bz(ell1), ellipse_to_bz(ell2)
    c = min(z, zeta)
    x = int(floor(lamb ** c / 2))
    n = max(1, x)

    def apply_and_compute_A(ell: Ellipse) -> tuple[Ellipse, tuple[float]]:
        op = (1, -2 * n, 0, 1)
        ell_A = apply_op(ell, op)
        return ell_A, op
    
    def apply_and_compute_A_bul(ell: Ellipse) -> tuple[Ellipse, tuple[float]]:
        op = (1, -2 * n, 0, 1)
        ell_A_bul = apply_op(ell, op)
        return ell_A_bul, op
    
    ell1_A, op_A = apply_and_compute_A(ell1)
    ell2_A_bul, op_A_bul = apply_and_compute_A_bul(ell2)
    if return_operators:
        return ell1_A, ell2_A_bul, op_A, op_A_bul
    return ell1_A, ell2_A_bul


# The B lemma
def B_lemma(
    ell1: Ellipse,
    ell2: Ellipse,
    return_operators: bool = True,
) -> tuple[Ellipse] | tuple[Ellipse, Ellipse, tuple[float], tuple[float]]:
    """
    If skew(D, Delta) >= 15, bias(D, Delta) in [-1, 1] and b, beta >= 0,
    z, zeta >= -0.2, then there exists an n such that
    skew(B^n(D, Delta)) <= 0.9 * skew(D, Delta).
    
    Args:
        ell1 (Ellipse): The first ellipse. This is associated with the epsilon
            region.

        ell2 (Ellipse): The second ellipse. This is associated with the unit
            disc in R^2.

        return_operators (bool): If True, return the R operators for the
            first and second ellipses. (Default: False)
    
    Returns:
        (Ellipse): The R-transformed epsilon region ellipse.

        (Ellipse): The R-transformed unit disc ellipse.

        (Optional[tuple[float]]): The A operator for the first ellipse.
            If return_operators is False, this is not returned.

        (Optional[tuple[float]]): The B_bul operator for the second ellipse.
            If return_operators is False, this is not
    """
    assert ell1.b <= 0 and 0 <= ell2.b
    (_, z), (_, zeta) = ellipse_to_bz(ell1), ellipse_to_bz(ell2)
    c = min(z, zeta)
    x = int(floor(lamb ** c * root2_inv))
    n = max(1, x)
    def apply_and_compute_B(ell: Ellipse) -> tuple[Ellipse, tuple[float]]:
        op = (1, root2 * n, 0, 1)
        ell_B = apply_op(ell, op)
        return ell_B, op
    
    def apply_and_compute_B_bul(ell: Ellipse) -> tuple[Ellipse, tuple[float]]:
        op = (1, -root2 * n, 0, 1)
        ell_B_bul = apply_op(ell, op)
        return ell_B_bul, op
    
    ell1_B, op_B = apply_and_compute_B(ell1)
    ell2_B_bul, op_B_bul = apply_and_compute_B_bul(ell2)
    if return_operators:
        return ell1_B, ell2_B_bul, op_B, op_B_bul
    return ell1_B, ell2_B_bul


def X_operator(
    ell1: Ellipse,
    ell2: Ellipse,
    op1: tuple[float] | None = None,
    op2: tuple[float] | None = None,
) -> tuple[Ellipse] | tuple[Ellipse, Ellipse, tuple[float], tuple[float]]:
    """
    Swap the diagonal elements of the ellipses.
    """
    if op1 is not None and op2 is None or op1 is None and op2 is not None:
        raise ValueError("Both or neither operators must be provided.")
    ell1_X = Ellipse([ell1.d, ell1.b, ell1.a], ell1.center)
    ell2_X = Ellipse([ell2.d, ell2.b, ell2.a], ell2.center)
    if op1 is None and op2 is None:
        return ell1_X, ell2_X
    op1_X = (op1[1], op1[0], op1[3], op1[2])
    op2_X = (op2[1], op2[0], op2[3], op2[2])
    return ell1_X, ell2_X, op1_X, op2_X


def Z_operator(
    ell1: Ellipse,
    ell2: Ellipse,
    op1: tuple[float] | None = None,
    op2: tuple[float] | None = None,
) -> tuple[Ellipse] | tuple[Ellipse, Ellipse, tuple[float], tuple[float]]:
    """
    Multiply the off diagonal elements of the ellipses by -1.
    """
    if op1 is not None and op2 is None or op1 is None and op2 is not None:
        raise ValueError("Both or neither operators must be provided.")
    ell1_Z = Ellipse([ell1.a, -ell1.b, ell1.d], ell1.center)
    ell2_Z = Ellipse([ell2.a, -ell2.b, ell2.d], ell2.center)
    if op1 is None and op2 is None:
        return ell1_Z, ell2_Z
    op1_Z = (op1[0], -op1[1], op1[2], -op1[3])
    op2_Z = (op2[0], -op2[1], op2[2], -op2[3])
    return ell1_Z, ell2_Z, op1_Z, op2_Z


def reduce(
    enclosing_ellipse: Ellipse,
) -> tuple[Ellipse, Ellipse, tuple[float], tuple[float]]:
    """
    Reduce the given ellipse so that it and the transformed unit disc are
    both at least 1/6 upright.

    Args:
        enclosing_ellipse (Ellipse): An ellipse enclosing the epsilon region.
    
    Returns:
        (Ellipse): The transformed epsilon region ellipse.

        (Ellipse): The transformed unit disc.
    """

    eps_ell = enclosing_ellipse.copy()
    root_det_D = sqrt(enclosing_ellipse.det())
    eps_ell = Ellipse(
        [
            enclosing_ellipse.a / root_det_D,
            enclosing_ellipse.b / root_det_D,
            enclosing_ellipse.d / root_det_D,
        ],
        enclosing_ellipse.center,
    )  
    eps_ell_copy = eps_ell.copy()
    unit_disc = Ellipse([1, 0, 1], (0, 0))

    def fix_bias(ell1: Ellipse, ell2: Ellipse) -> tuple[Ellipse, tuple[float]]:
        bias_ = float(bias(ell1, ell2))
        if abs(bias_) > 1:
            x = (1 - bias_) / 2
            if x < 0:
                k = int(floor(x))
            else:
                k = int(floor(x))
            ell1, ell2, op1, op2 = shift_k(ell1, ell2, k, return_operators=True)
        else:
            op1, op2 = (1, 0, 0, 1), (1, 0, 0, 1)
        return ell1, ell2, op1, op2

    def in_R_region(z: float, zeta: float) -> bool:
        return abs(z) <= 0.8 and abs(zeta) <= 0.8

    def in_B_region(b: float, beta: float) -> bool:
        return b <= 0 and 0 <= beta

    skews = [skew(eps_ell, unit_disc)]
    counter = 0
    op_D, op_Delta = (1, 0, 0, 1), (1, 0, 0, 1)

    # from numpy import isclose
    # def check_match(op_D: Sequence[float]) -> bool:
    #     aa = float(apply_op(eps_ell_copy, op_D).a)
    #     return isclose(aa, float(eps_ell.a), atol=1e-5)

    while skew(eps_ell, unit_disc) >= 15:
        eps_ell, unit_disc, op1, op2 = fix_bias(eps_ell, unit_disc)
        op_D, op_Delta = combine_ops(op_D, op1), combine_ops(op_Delta, op2)
        # if not check_match(op_D):
        #     import pdb; pdb.set_trace()

        # For determining case
        b, z = ellipse_to_bz(eps_ell)
        beta, zeta = ellipse_to_bz(unit_disc)

        # Apply X_operator when not in R and zeta < -z
        if not in_R_region(z, zeta) and zeta < -z or b < 0 and zeta < -z:
            eps_ell, unit_disc, op_D, op_Delta = X_operator(
                eps_ell, unit_disc, op_D, op_Delta,
            )
            z, zeta = -z, -zeta
        # if not check_match(op_D):
        #     import pdb; pdb.set_trace()
        
        # TODO: Check for when to apply Z
        # Apply Z if beta <= 0 <= b
        if beta < 0 and b >= 0 or not in_B_region(b, beta) and b < 0:
            eps_ell, unit_disc, op_D, op_Delta = Z_operator(
                eps_ell, unit_disc, op_D, op_Delta,
            )
            b, beta = -b, -beta
        # if not check_match(op_D):
        #     import pdb; pdb.set_trace()

        # Case 1.1 - apply R
        if b >= 0 and in_R_region(z, zeta):
            eps_ell, unit_disc, op1, op2 = R_lemma(eps_ell, unit_disc)
        # Case 1.2 - apply K
        elif b >= 0 and z <= 0.3 and 0.8 <= zeta:
            eps_ell, unit_disc, op1, op2 = K_lemma(eps_ell, unit_disc)
        # Case 1.3 - apply A
        elif b >= 0 and 0.3 <= z and 0.3 <= zeta:
            eps_ell, unit_disc, op1, op2 = A_lemma(eps_ell, unit_disc)
        # Case 1.4 - apply K_bul
        elif b >= 0 and 0.8 <= z and zeta <= 0.3:
            eps_ell, unit_disc, op1, op2 = K_lemma(
                eps_ell, unit_disc, bullet=True,
            )
        # Case 2.1 0 apply R
        elif b < 0 and abs(z) <= 0.8 and abs(zeta) <= 0.8:
            eps_ell, unit_disc, op1, op2 = R_lemma(eps_ell, unit_disc)
        # Case 2.2 - apply B
        elif b < 0 and -0.2 <= z and -0.2 <= zeta:
            eps_ell, unit_disc, op1, op2 = B_lemma(eps_ell, unit_disc)
        else:
            raise ValueError("No case matched.")

        skews.append(skew(eps_ell, unit_disc))
        
        op_D, op_Delta = combine_ops(op_D, op1), combine_ops(op_Delta, op2)

        # if not check_match(op_D):
        #     import pdb; pdb.set_trace()

        counter += 1
    
    eps_ell = Ellipse(
        [
            eps_ell.a * root_det_D,
            eps_ell.b * root_det_D,
            eps_ell.d * root_det_D,
        ],
        eps_ell.center,
    )
    return eps_ell, unit_disc, op_D, op_Delta