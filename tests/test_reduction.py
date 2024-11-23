from mpmath import mp
from mpmath import sqrt
from mpmath import pi

from numpy import isclose

from random import randint
from random import uniform

from cyclosynth.ellipse import Ellipse
from cyclosynth.matrix import Vector
from cyclosynth.reduction import apply_op
from cyclosynth.reduction import bias
from cyclosynth.reduction import skew
from cyclosynth.reduction import shift_k_ellipses
from cyclosynth.reduction import R_lemma
from cyclosynth.reduction import K_lemma
from cyclosynth.reduction import A_lemma
from cyclosynth.reduction import B_lemma
from cyclosynth.reduction import reduce

mp.dps = 100


def random_state(
    z: float | None = None,
    zeta: float | None = None,
    bias: float | None = None,
    b_sign: int = 1,
) -> tuple[Ellipse, Ellipse]:
    if zeta is not None and bias is not None:
        raise ValueError("Cannot specify both zeta and bias.")
    if b_sign not in [-1, 1]:
        raise ValueError("b_sign must be +/- 1.")

    lamb = 1 + sqrt(2)

    if z is None:
        z = randint(2, 50)
    b = b_sign * randint(10, 50)
    e = sqrt(b ** 2 + 1)

    if bias is None:
        if zeta is None:
            sign = 1 if randint(0, 1) == 0 else -1
            zeta = z + sign * randint(2, 10)
    else:
        zeta = z + bias

    beta = randint(10, 50)
    epsi = sqrt(beta ** 2 + 1)

    ell1 = Ellipse([e * lamb ** -z, b, e * lamb ** z])
    ell2 = Ellipse([epsi * lamb ** -zeta, beta, epsi * lamb ** zeta])

    return ell1, ell2


class TestReduction:

    num_trials = 100

    def test_bias(self) -> None:
        for _ in range(self.num_trials):
            sign = 1 if randint(0, 1) == 0 else -1
            bias_ = sign * uniform(2.0, 10.0)
            ell1, ell2 = random_state(bias=bias_)
            assert isclose(bias(ell1, ell2), bias_, rtol=1e-5)

    def test_shift_k(self) -> None:
        # Bias(D, Delta) > 1 -> zeta > 1 + z
        for _ in range(self.num_trials):
            # Setup
            k = randint(1, 4)
            sign = 1 if randint(0, 1) == 0 else -1
            bias_ = sign * uniform(2.0, 10.0)
            ell1, ell2 = random_state(bias=bias_)
            bias_1 = float(bias(ell1, ell2))
            skew_1 = float(skew(ell1, ell2))
            # Apply operator
            ella, ellb = shift_k_ellipses(ell1, ell2, -sign * k)
            bias_2 = float(bias(ella, ellb))
            skew_2 = float(skew(ella, ellb))
            assert isclose(skew_1, skew_2, rtol=1e-5)
            assert isclose(bias_2 + sign * 2 * k, bias_1, rtol=1e-5)
            # Comparing to grid operator requires shifting back
            ellaa, ellbb = shift_k_ellipses(ella, ellb, sign * k)
            bias_3 = float(bias(ellaa, ellbb))
            skew_3 = float(skew(ellaa, ellbb))
            assert isclose(skew_1, skew_3, rtol=1e-5)
            assert isclose(bias_3, bias_1, rtol=1e-5)
            assert(float(ell1.a) == float(ellaa.a))
            assert(float(ell1.b) == float(ellaa.b))
            assert(float(ell1.d) == float(ellaa.d))
            assert(float(ell2.a) == float(ellbb.a))
            assert(float(ell2.b) == float(ellbb.b))
            assert(float(ell2.d) == float(ellbb.d))
    
    def test_R_lemma(self) -> None:
        for _ in range(self.num_trials):
            # Setup
            z = uniform(-0.8, 0.8)
            zeta = uniform(-0.8, 0.8)
            ell1, ell2 = random_state(z=z, zeta=zeta)
            skew1 = float(skew(ell1, ell2))
            # Apply operator
            ella, ellb, R, R_bul = R_lemma(ell1, ell2)
            skew2 = float(skew(ella, ellb))
            assert skew2 <= 0.9 * skew1
            # Manually check operator
            ellaa = apply_op(ell1, R)
            ellbb = apply_op(ell2, R_bul)
            assert isclose(float(ellaa.a), float(ella.a), rtol=1e-5)
            assert isclose(float(ellaa.b), float(ella.b), rtol=1e-5)
            assert isclose(float(ellaa.d), float(ella.d), rtol=1e-5)
            assert isclose(float(ellbb.a), float(ellb.a), rtol=1e-5)
            assert isclose(float(ellbb.b), float(ellb.b), rtol=1e-5)
            assert isclose(float(ellbb.d), float(ellb.d), rtol=1e-5)
    
    def test_K_lemma(self) -> None:
        # bias(D, Delta) in [-1, 1], zeta >= 0.8 and z <= 0.3 imply that
        # z + 1 in [-0.2, 1.3] and zeta - 1 in [-0.2, 1.3]
        for _ in range(self.num_trials):
            # Setup
            z = uniform(-1.2, 0.3)
            zeta = uniform(0.8, 2.3)
            ell1, ell2 = random_state(z=z, zeta=zeta)
            skew1 = float(skew(ell1, ell2))
            # Apply operator
            ella, ellb, K, K_bul = K_lemma(ell1, ell2)
            skew2 = float(skew(ella, ellb))
            assert skew2 <= 0.9 * skew1
            # Manually check operator
            ellaa = apply_op(ell1, K)
            ellbb = apply_op(ell2, K_bul)
            assert isclose(float(ellaa.a), float(ella.a), rtol=1e-5)
            assert isclose(float(ellaa.b), float(ella.b), rtol=1e-5)
            assert isclose(float(ellaa.d), float(ella.d), rtol=1e-5)
            assert isclose(float(ellbb.a), float(ellb.a), rtol=1e-5)
            assert isclose(float(ellbb.b), float(ellb.b), rtol=1e-5)
            assert isclose(float(ellbb.d), float(ellb.d), rtol=1e-5)
    
    def test_A_lemma(self) -> None:
        # bias(D, Delta) in [-1, 1], and z, zeta >= 0.3
        for _ in range(self.num_trials):
            # Setup
            z = uniform(0.3, 100)
            bias_ = uniform(-1.0, 1.0)
            if z + bias_ < 0.3:
                bias_ = 0.3 - z
            ell1, ell2 = random_state(z=z, bias=bias_)
            skew1 = float(skew(ell1, ell2))
            # Apply operator
            ella, ellb, A, A_bul = A_lemma(ell1, ell2)
            skew2 = float(skew(ella, ellb))
            assert skew2 <= 0.9 * skew1
            # Manually check operator
            ellaa = apply_op(ell1, A)
            ellbb = apply_op(ell2, A_bul)
            assert isclose(float(ellaa.a), float(ella.a), rtol=1e-5)
            assert isclose(float(ellaa.b), float(ella.b), rtol=1e-5)
            assert isclose(float(ellaa.d), float(ella.d), rtol=1e-5)
            assert isclose(float(ellbb.a), float(ellb.a), rtol=1e-5)
            assert isclose(float(ellbb.b), float(ellb.b), rtol=1e-5)
            assert isclose(float(ellbb.d), float(ellb.d), rtol=1e-5)
    
    def test_B_lemma(self) -> None:
        # bias(D, Delta) in [-1, 1], b <= 0 <= beta, and z, zeta >= -0.2
        for _ in range(self.num_trials):
            # Setup
            z = uniform(-0.2, 100)
            bias_ = uniform(-1.0, 1.0)
            if z + bias_ < -0.2:
                bias_ = -0.2 - z
            ell1, ell2 = random_state(z=z, bias=bias_, b_sign=-1)
            skew1 = float(skew(ell1, ell2))
            # Apply operator
            ella, ellb, B, B_bul = B_lemma(ell1, ell2)
            skew2 = float(skew(ella, ellb))
            assert skew2 <= 0.9 * skew1
            # Manually check operator
            ellaa = apply_op(ell1, B)
            ellbb = apply_op(ell2, B_bul)
            assert isclose(float(ellaa.a), float(ella.a), rtol=1e-5)
            assert isclose(float(ellaa.b), float(ella.b), rtol=1e-5)
            assert isclose(float(ellaa.d), float(ella.d), rtol=1e-5)
            assert isclose(float(ellbb.a), float(ellb.a), rtol=1e-5)
            assert isclose(float(ellbb.b), float(ellb.b), rtol=1e-5)
            assert isclose(float(ellbb.d), float(ellb.d), rtol=1e-5)
    
    def test_reduce(self) -> None:
        for _ in range(self.num_trials):
            # Set up
            angle = uniform(0, 2 * pi)
            epsilon = 1e-5
            ell = Ellipse.find_ellipse(angle, epsilon)
            disc = Ellipse([1, 0, 1], (0, 0))

            norm = sqrt(ell.det())
            ell_norm = Ellipse(
                [ell.a / norm, ell.b / norm, ell.d / norm],
                ell.center,
            )
            # Apply reduction
            opG = reduce(ell)
            GellG = apply_op(ell_norm, opG)
            GdiscG = apply_op(disc, opG.conj())
            assert GellG.skew() < 15
            assert GdiscG.skew() < 15

            # Check that a known solution for the original problem is also a
            # solution for the reduced problem
            u = Vector(ell.center)
            assert ell.check_inclusion(u)
            assert disc.check_inclusion(u)