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
from cyclosynth.convex import ConvexSet
from cyclosynth.ellipse import Ellipse
from cyclosynth.ratio import AlgebraicIntegerOverRoot2
from cyclosynth.ratio import IntegerRatio
from cyclosynth.operator import Operator
from cyclosynth.gridsynth import gridpoints_1d
from cyclosynth.gridsynth import gridpoints_2d

# from random import seed
# seed(42)

mp.dps = 100


def random_ratio(k: int) -> AlgebraicIntegerOverRoot2:
    a, b = [randint(-100, 100) for _ in range(2)]
    return AlgebraicIntegerOverRoot2(RingRoot2([a, b]), k)

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
        k = 10
        solutions_check_limit = 10
        offset = 0.05
        for _ in range(self.num_trials):
            x_hi = uniform(-1 + offset, 1.0)
            x_lo = x_hi - offset
            y_lo, y_hi = -1, 1
            args = (x_lo, x_hi, y_lo, y_hi, k)
            no_solutions = True
            for i, c in enumerate(gridpoints_1d(*args)):
                no_solutions = False
                if i >= solutions_check_limit:
                    break
                assert c is not None
                c_float = c.to_float()
                assert x_lo <= c_float and c_float <= x_hi
            assert not no_solutions

    def test_solve_scaled_grid_problem_1d_failure(self) -> None:
        k = 1
        x_lo, x_hi = -0.1, 0.1
        y_lo, y_hi = 0.1, 1
        args = (x_lo, x_hi, y_lo, y_hi, k)
        solutions = [c for c in gridpoints_1d(*args)]
        assert len(solutions) == 0
    
    def test_solve_scaled_grid_problem_1d_parity(self) -> None:
        k = 11
        solutions_check_limit = 10
        offset = 0.05
        for _ in range(self.num_trials):
            beta = random_ratio(k + 1)
            x_hi = uniform(-1 + offset, 1.0)
            x_lo = x_hi - offset
            y_lo, y_hi = -1, 1
            args = (x_lo, x_hi, y_lo, y_hi, k, beta)
            no_solutions = True
            for i, c in enumerate(gridpoints_1d(*args)):
                no_solutions = False
                if i >= solutions_check_limit:
                    break
                assert c is not None
                c_float = c.to_float()
                assert x_lo <= c_float and c_float <= x_hi
                c = AlgebraicIntegerOverRoot2.from_integer_ratio(c)
                assert c.denominator_power <= k - 1
            assert not no_solutions
    
    def test_solve_grid_problem_2d(self) -> None:
        ell = Ellipse([1/2, 0, 1/2], (0, 0))
        one = IntegerRatio(RingRoot2([1, 0]))
        zero = IntegerRatio(RingRoot2([0, 0]))
        opG = Operator([one, zero, zero, one])
        set_A, set_B = ConvexSet(ell), ConvexSet(ell)
        solutions = [_ for _ in gridpoints_2d(set_A, set_B, opG, 0)]
        assert len(solutions) == 17