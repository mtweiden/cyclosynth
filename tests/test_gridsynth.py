from random import random
from random import uniform
from random import randint

from mpmath import mp
from mpmath import acos
from mpmath import cos
from mpmath import sin
from mpmath import pi
from mpmath import sqrt

from numpy import isclose

from cyclosynth.algebra import RingRoot2
from cyclosynth.gridsynth import find_ellipse
from cyclosynth.gridsynth import in_epsilon_region
from cyclosynth.gridsynth import solve_grid_problem_1d
from cyclosynth.gridsynth import solve_scaled_grid_problem_1d

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
    
    def test_solve_grid_problem_1d_a(self) -> None:
        x_lo, x_hi = 0, 8.25
        y_lo, y_hi = -1, 1
        solutions = [s for s in solve_grid_problem_1d(x_lo, x_hi, y_lo, y_hi)]
        assert len(solutions) == 7
        assert isclose(solutions[0].to_float(), 0.0)
        assert isclose(solutions[1].to_float(), 1.0)
        assert isclose(solutions[2].to_float(), float(1 + sqrt(2)))
        assert isclose(solutions[3].to_float(), float(2 + sqrt(2)))
        assert isclose(solutions[4].to_float(), float(2 + 2 * sqrt(2)))
        assert isclose(solutions[5].to_float(), float(3 + 2 * sqrt(2)))
        assert isclose(solutions[6].to_float(), float(4 + 3 * sqrt(2)))
    
    def test_solve_grid_problem_1d_b(self) -> None:
        x_lo, x_hi = -5, 5
        y_lo, y_hi = -3, 3
        solutions = [s for s in solve_grid_problem_1d(x_lo, x_hi, y_lo, y_hi)]
        assert len(solutions) == 22
        for solution in solutions:
            s, sc = solution.to_float(), solution.conj().to_float()
            assert x_lo <= s and s <= x_hi
            assert y_lo <= sc and sc <= y_hi
    
    def test_solve_scaled_grid_problem_1d(self) -> None:
        k = 20
        solutions_check_limit = 10
        for _ in range(self.num_trials):
            x_lo, x_hi = uniform(-100, 100), uniform(-100, 100)
            if x_lo > x_hi:
                x_lo, x_hi = x_hi, x_lo
            elif x_lo == x_hi:
                x_lo -= 1
            y_lo, y_hi = -1, 1
            args = (x_lo, x_hi, y_lo, y_hi, k)
            for i, c in enumerate(solve_scaled_grid_problem_1d(*args)):
                if i >= solutions_check_limit:
                    break
                assert c is not None
                c_float = c.to_float()
                assert x_lo <= c_float and c_float <= x_hi

    def test_solve_scaled_grid_problem_1d_failure(self) -> None:
        k = 1
        x_lo, x_hi = -0.1, 0.1
        y_lo, y_hi = 0.1, 1
        args = (x_lo, x_hi, y_lo, y_hi, k)
        solutions = [c for c in solve_scaled_grid_problem_1d(*args)]
        assert len(solutions) == 0
    
    def test_solve_scaled_grid_problem_2d(self) -> None:
        ...

    # def test_in_epsilon_region(self) -> None:
    #     epsilon = 1e-20
    #     for _ in range(self.num_trials):
    #         angle = 2 * pi * random()
    #         ellipse = find_ellipse(angle, epsilon)
    #         (x_lo, x_hi), (y_lo, y_hi) = ellipse.bounding_box()
    #         k = 40
    #         candidates = solve_scaled_grid_problem_1d(x_lo, x_hi, y_lo, y_hi, k)
    #         something_works = False
    #         for s in candidates:
    #             if in_epsilon_region(angle, epsilon, )
    