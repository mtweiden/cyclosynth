from random import random
from random import uniform
from random import randint

from mpmath import mp
from mpmath import acos
from mpmath import cos
from mpmath import sin
from mpmath import pi

from cyclosynth.algebra import RingRoot2
from cyclosynth.ratio import IntegerRatio
from cyclosynth.gridsynth import find_ellipse
from cyclosynth.gridsynth import enumerate_points_for_k

# from random import seed
# seed(42)

mp.dps = 100


class TestGridsynth:

    num_trials = 1000

    def test_find_ellipse(self) -> None:
        angle = 2 * pi * random()
        epsilon = 1e-8
        ellipse = find_ellipse(angle, epsilon)

        def random_point_in_epsilon_region() -> tuple[float, float]:
            # Sample a point in the unrotated epsilon region
            d = 1 - (epsilon**2 / 2)
            y_lim = sin(acos(d))
            y = uniform(-y_lim, y_lim)
            x_max = max(d, 1 - y**2 - epsilon**3)
            x = uniform(d, x_max)
            # Rotate the epsilon region by the angle
            x_ = x * cos(angle / 2) + y * sin(angle / 2)
            y_ = x * -sin(angle / 2) + y * cos(angle / 2)
            return x_, y_
        
        for _ in range(self.num_trials):
            p = random_point_in_epsilon_region()
            assert ellipse.check_inclusion(p)


    def test_enumerate_points_for_k(self) -> None:
        for _ in range(self.num_trials):
            a, b, c, d = [randint(-100, 100) for _ in range(4)]
            if (c - a) % 2 == 0:
                alpha = RingRoot2([d, (c - a) // 2])
                beta = RingRoot2([b, (c + a) // 2])
            else:
                alpha = RingRoot2([d, (c - a - 1) // 2])
                beta = RingRoot2([b, (c + a - 1) // 2])
            
            epsilon = 1e-3
            x_lo = alpha.to_float() - epsilon
            x_hi = alpha.to_float() + epsilon
            y_lo = beta.to_float() - epsilon
            y_hi = beta.to_float() + epsilon
            k = 20
            solutions = enumerate_points_for_k(x_lo, x_hi, y_lo, y_hi, k)
            for s in solutions:
                assert s is not None
                break