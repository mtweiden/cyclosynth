from random import random
from random import uniform

from numpy import array
from numpy.linalg import eigvals
from numpy import isclose

from mpmath import acos
from mpmath import cos
from mpmath import pi
from mpmath import sin
from mpmath import sqrt

from cyclosynth.ellipse import Ellipse


from random import seed
seed(42)

def random_ellipse(random_center: bool = False) -> Ellipse:
    a, b, d = 1000 * random(), 1000 * random(), 1000 * random()
    mat = array([[a, b], [b, d]])
    mat = (mat @ mat.T)
    assert all(eigvals(mat) > 0)
    a, b, d = mat[0, 0], mat[0, 1], mat[1, 1]
    center = (random() * 1000, random() * 1000) if random_center else (0, 0)
    return Ellipse([a, b, d], center)


class TestEllipse:

    num_trials = 1000

    def test_find_ellipse(self) -> None:
        angle = 2 * pi * random()
        epsilon = 1e-8
        ellipse = Ellipse.find_ellipse(angle, epsilon)

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

    def test_make_upright(self) -> None:
        for _ in range(self.num_trials):
            e = random_ellipse()
            if e.uprightness() >= 0.5:
                continue
            skew = e.skew()
            new_e = e.make_upright()
            assert new_e.skew() <= 0.5 * skew
            assert new_e.uprightness() >= 0.5
    
    def test_upright_operators(self) -> None:
        for _ in range(self.num_trials):
            e = random_ellipse()
            if e.uprightness() >= 0.5:
                continue
            new_e, op = e.make_upright(return_operator=True)
            manual_e = e.copy()
            manual_e = manual_e.apply_operator(op)
            assert new_e.is_close(manual_e)
    
    def test_bounding_box(self) -> None:
        for _ in range(self.num_trials):
            e = random_ellipse(random_center=True)
            (x_lo, x_hi), (y_lo, y_hi) = e.bounding_box()
            assert x_lo <= x_hi
            assert y_lo <= y_hi

            p1, p2 = e.center
            a, b, d = e.a, e.b, e.d

            def descriminant_x_bounds(x: float) -> float:
                beta = (2 * p2 * d - 2 * b * x) / d
                alpha = (a * x ** 2 - 2 * b * x * p2 + d * p2 ** 2 - 1) / d
                return beta ** 2 - 4 * alpha
            
            def descriminant_y_bounds(y: float) -> float:
                beta = (2 * p1 * a - 2 * b * y) / a
                alpha = (a * p1 ** 2 - 2 * b * p1 * y + d * y ** 2 - 1) / a
                return beta ** 2 - 4 * alpha

            # Test x bounds
            desc_x_lo = descriminant_x_bounds(x_lo - p1)
            desc_x_hi = descriminant_x_bounds(x_hi - p1)
            assert isclose(desc_x_lo, 0, atol=1e-3)
            assert isclose(desc_x_hi, 0, atol=1e-3)

            # Test y bounds
            desc_y_lo = descriminant_y_bounds(y_lo - p2)
            desc_y_hi = descriminant_y_bounds(y_hi - p2)
            assert isclose(desc_y_lo, 0, atol=1e-3)
            assert isclose(desc_y_hi, 0, atol=1e-3)