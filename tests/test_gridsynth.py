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
from cyclosynth.gridsynth import in_epsilon_region
from cyclosynth.gridsynth import gridpoints_1d
from cyclosynth.gridsynth import scaled_gridpoints_1d
from cyclosynth.ratio import AlgebraicIntegerOverRoot2

# from random import seed
# seed(42)

mp.dps = 100


class TestGridsynth:

    num_trials = 1000
    
    def test_gridpoints_1d_a(self) -> None:
        x_lo, x_hi = 0, 8.25
        y_lo, y_hi = -1, 1
        solutions = [s for s in gridpoints_1d(x_lo, x_hi, y_lo, y_hi)]
        assert len(solutions) == 7
        assert isclose(solutions[0].to_float(), 0.0)
        assert isclose(solutions[1].to_float(), 1.0)
        assert isclose(solutions[2].to_float(), float(1 + sqrt(2)))
        assert isclose(solutions[3].to_float(), float(2 + sqrt(2)))
        assert isclose(solutions[4].to_float(), float(2 + 2 * sqrt(2)))
        assert isclose(solutions[5].to_float(), float(3 + 2 * sqrt(2)))
        assert isclose(solutions[6].to_float(), float(4 + 3 * sqrt(2)))
    
    def test_gridpoints_1d_b(self) -> None:
        x_lo, x_hi = -3, 3
        y_lo, y_hi = -3, 3 + 1e-8
        solutions = [s for s in gridpoints_1d(x_lo, x_hi, y_lo, y_hi)]
        assert len(solutions) == 15
        assert isclose(solutions[0].to_float(), -3.0)
        assert isclose(solutions[1].to_float(), float(-2 * sqrt(2)))
        assert isclose(solutions[2].to_float(), -1 + float(-1 * sqrt(2)))
        assert isclose(solutions[3].to_float(), -2)
        assert isclose(solutions[4].to_float(), float(-1 * sqrt(2)))
        assert isclose(solutions[5].to_float(), -1)
        assert isclose(solutions[6].to_float(), 1 + float(-1 * sqrt(2)))
        assert isclose(solutions[7].to_float(), 0.0)
        assert isclose(solutions[8].to_float(), -1 + float(1 * sqrt(2)))
        assert isclose(solutions[9].to_float(), 1)
        assert isclose(solutions[10].to_float(), float(1 * sqrt(2)))
        assert isclose(solutions[11].to_float(), 2)
        assert isclose(solutions[12].to_float(), 1 + float(1 * sqrt(2)))
        assert isclose(solutions[13].to_float(), float(2 * sqrt(2)))
        assert isclose(solutions[14].to_float(), 3.0)
    
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
            for i, c in enumerate(scaled_gridpoints_1d(*args)):
                if i >= solutions_check_limit:
                    break
                assert c is not None
                c_float = c.to_float()
                assert x_lo <= c_float and c_float <= x_hi

    # def test_solve_scaled_grid_problem_1d_failure(self) -> None:
    #     k = 1
    #     x_lo, x_hi = -0.1, 0.1
    #     y_lo, y_hi = 0.1, 1
    #     args = (x_lo, x_hi, y_lo, y_hi, k)
    #     solutions = [c for c in solve_scaled_grid_problem_1d(*args)]
    #     assert len(solutions) == 0
    
    # def test_solve_scaled_parity_grid_problem_1d(self) -> None:
    #     # TODO: Figure out what to put for beta
    #     k = 20
    #     solutions_check_limit = 10
    #     for _ in range(self.num_trials):
    #         x_lo, x_hi = uniform(-100, 100), uniform(-100, 100)
    #         if x_lo > x_hi:
    #             x_lo, x_hi = x_hi, x_lo
    #         elif x_lo == x_hi:
    #             x_lo -= 1
    #         y_lo, y_hi = -1, 1

    #         # IDK what this should be
    #         beta = AlgebraicIntegerOverRoot2(RingRoot2([1, 1]), k)

    #         args = (x_lo, x_hi, y_lo, y_hi, k, beta)
    #         for i, c in enumerate(solve_scaled_parity_grid_problem_1d(*args)):
    #             if i >= solutions_check_limit:
    #                 break
    #             import pdb; pdb.set_trace()
    #             assert c is not None
    #             c_float = c.to_float()
    #             assert x_lo <= c_float and c_float <= x_hi
    #             c.simplify()
    #             print(c.denominator_power)
    #             assert c.denominator_power <= k - 1

    
    # def test_solve_scaled_grid_problem_2d(self) -> None:
    #     ...

    # # def test_in_epsilon_region(self) -> None:
    # #     epsilon = 1e-20
    # #     for _ in range(self.num_trials):
    # #         angle = 2 * pi * random()
    # #         ellipse = find_ellipse(angle, epsilon)
    # #         (x_lo, x_hi), (y_lo, y_hi) = ellipse.bounding_box()
    # #         k = 40
    # #         candidates = solve_scaled_grid_problem_1d(x_lo, x_hi, y_lo, y_hi, k)
    # #         something_works = False
    # #         for s in candidates:
    # #             if in_epsilon_region(angle, epsilon, )
    