from mpmath import mp
from mpmath import sqrt
from mpmath import pi

from numpy import isclose

from random import randint
from random import uniform

from cyclosynth.ellipse import Ellipse

from cyclosynth.reduction import apply_op
from cyclosynth.reduction import bias
from cyclosynth.reduction import skew
from cyclosynth.reduction import shift_k
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
            assert isclose(bias(ell1, ell2), bias_, atol=1e-5)

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
            ella, ellb = shift_k(ell1, ell2, -sign * k, False)
            bias_2 = float(bias(ella, ellb))
            skew_2 = float(skew(ella, ellb))
            assert isclose(skew_1, skew_2, atol=1e-5)
            assert isclose(bias_2 + sign * 2 * k, bias_1, atol=1e-5)
            # Manually check operator
            _, _, sigmak, tauk = shift_k(ell1, ell2, -sign * k, True)
            ellaa = apply_op(ell1, sigmak)
            ellbb = apply_op(ell2, tauk)
            assert isclose(float(ellaa.a), float(ella.a), atol=1e-5)
            assert isclose(float(ellaa.b), float(ella.b), atol=1e-5)
            assert isclose(float(ellaa.d), float(ella.d), atol=1e-5)
            assert isclose(float(ellbb.a), float(ellb.a), atol=1e-5)
            assert isclose(float(ellbb.b), float(ellb.b), atol=1e-5)
            assert isclose(float(ellbb.d), float(ellb.d), atol=1e-5)
    
    def test_R_lemma(self) -> None:
        for _ in range(self.num_trials):
            # Setup
            z = uniform(-0.8, 0.8)
            zeta = uniform(-0.8, 0.8)
            ell1, ell2 = random_state(z=z, zeta=zeta)
            skew1 = float(skew(ell1, ell2))
            # Apply operator
            ella, ellb = R_lemma(ell1, ell2, False)
            skew2 = float(skew(ella, ellb))
            assert skew2 <= 0.9 * skew1
            # Manually check operator
            _, _, R, R_bul = R_lemma(ell1, ell2, True)
            ellaa = apply_op(ell1, R)
            ellbb = apply_op(ell2, R_bul)
            assert isclose(float(ellaa.a), float(ella.a), atol=1e-5)
            assert isclose(float(ellaa.b), float(ella.b), atol=1e-5)
            assert isclose(float(ellaa.d), float(ella.d), atol=1e-5)
            assert isclose(float(ellbb.a), float(ellb.a), atol=1e-5)
            assert isclose(float(ellbb.b), float(ellb.b), atol=1e-5)
            assert isclose(float(ellbb.d), float(ellb.d), atol=1e-5)
    
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
            ella, ellb = K_lemma(ell1, ell2, False)
            skew2 = float(skew(ella, ellb))
            assert skew2 <= 0.9 * skew1
            # Manually check operator
            _, _, K, K_bul = K_lemma(ell1, ell2, True)
            ellaa = apply_op(ell1, K)
            ellbb = apply_op(ell2, K_bul)
            assert isclose(float(ellaa.a), float(ella.a), atol=1e-5)
            assert isclose(float(ellaa.b), float(ella.b), atol=1e-5)
            assert isclose(float(ellaa.d), float(ella.d), atol=1e-5)
            assert isclose(float(ellbb.a), float(ellb.a), atol=1e-5)
            assert isclose(float(ellbb.b), float(ellb.b), atol=1e-5)
            assert isclose(float(ellbb.d), float(ellb.d), atol=1e-5)
    
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
            ella, ellb = A_lemma(ell1, ell2, False)
            skew2 = float(skew(ella, ellb))
            assert skew2 <= 0.9 * skew1
            # Manually check operator
            _, _, A, A_bul = A_lemma(ell1, ell2, True)
            ellaa = apply_op(ell1, A)
            ellbb = apply_op(ell2, A_bul)
            assert isclose(float(ellaa.a), float(ella.a), atol=1e-5)
            assert isclose(float(ellaa.b), float(ella.b), atol=1e-5)
            assert isclose(float(ellaa.d), float(ella.d), atol=1e-5)
            assert isclose(float(ellbb.a), float(ellb.a), atol=1e-5)
            assert isclose(float(ellbb.b), float(ellb.b), atol=1e-5)
            assert isclose(float(ellbb.d), float(ellb.d), atol=1e-5)
    
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
            ella, ellb = B_lemma(ell1, ell2, False)
            skew2 = float(skew(ella, ellb))
            assert skew2 <= 0.9 * skew1
            # Manually check operator
            _, _, B, B_bul = B_lemma(ell1, ell2, True)
            ellaa = apply_op(ell1, B)
            ellbb = apply_op(ell2, B_bul)
            assert isclose(float(ellaa.a), float(ella.a), atol=1e-5)
            assert isclose(float(ellaa.b), float(ella.b), atol=1e-5)
            assert isclose(float(ellaa.d), float(ella.d), atol=1e-5)
            assert isclose(float(ellbb.a), float(ellb.a), atol=1e-5)
            assert isclose(float(ellbb.b), float(ellb.b), atol=1e-5)
            assert isclose(float(ellbb.d), float(ellb.d), atol=1e-5)
    
    def test_reduce(self) -> None:
        for _ in range(self.num_trials):
            # Set up
            angle = uniform(0, 2 * pi)
            epsilon = 1e-5
            ell1 = Ellipse.find_ellipse(angle, epsilon)

            # Apply reduction
            ell2, disc, opell, opdisc = reduce(ell1)
            norm = sqrt(ell2.det())
            ell2_normalized = Ellipse(
                [ell2.a / norm, ell2.b / norm, ell2.d / norm],
                ell2.center,
            )
            assert ell2_normalized.skew() < 15
            assert disc.skew() < 15

            # Manually check operators
            ell_manual = apply_op(ell1, opell)
            disc_manual = apply_op(Ellipse([1, 0, 1]), opdisc)

            assert isclose(float(disc_manual.a), float(disc.a), atol=1e-5)
            assert isclose(float(disc_manual.b), float(disc.b), atol=1e-5)
            assert isclose(float(disc_manual.d), float(disc.d), atol=1e-5)

            assert isclose(float(ell_manual.a), float(ell2.a), atol=1e-5)
            assert isclose(float(ell_manual.b), float(ell2.b), atol=1e-5)
            assert isclose(float(ell_manual.d), float(ell2.d), atol=1e-5)