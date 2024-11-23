from typing import Sequence

from random import randint

from math import sqrt
from math import isclose

from cyclosynth.algebra import RingRoot2
from cyclosynth.algebra import RingRootRoot2Plus2
from cyclosynth.algebra import DyadicComplexNumber
from cyclosynth.ratio import IntegerRatio
from cyclosynth.ratio import AlgebraicIntegerOverRoot2
from cyclosynth.ratio import AlgebraicIntegerOverRootRoot2Plus2


eps = 1e-6


def rand_integer_values(
    n: int,
    min_val: int = -1_000_000_000_000,
    max_val: int = 1_000_000_000_000,
) -> list[int]:
    return [randint(min_val, max_val) for _ in range(n)]


def random_ringroot2() -> RingRoot2:
    return RingRoot2(rand_integer_values(2))


def random_ringrootroot2plus2() -> RingRootRoot2Plus2:
    return RingRootRoot2Plus2(rand_integer_values(4))


def random_dyadic_coefficients(n: int) -> list[int]:
    values = [0] * (2 * n)
    # values[0] = rand_integer_values(1, -1_000_000, 1_000_000)[0]
    values[0] = rand_integer_values(1)[0]
    for i in range(1, 4):
        # coeff = rand_integer_values(1, -1_000_000, 1_000_000)[0]
        coeff = rand_integer_values(1)[0]
        values[2 * i] = coeff
        values[2 * n - 2 * i] = -coeff
    return values


class TestRatios:

    num_trials = 1000

    def test_simplify(self) -> None:
        for _ in range(self.num_trials):
            x = RingRootRoot2Plus2([1, 0, 0, 0])
            power = randint(0, 100)
            for _ in range(power):
                x = x * RingRootRoot2Plus2([0, 0, 1, 0])
            ratio = AlgebraicIntegerOverRootRoot2Plus2(x, power)
            ratio.simplify()
            assert ratio.numerator.values == [1, 0, 0, 0]
            assert ratio.denominator_power == 0

    def test_add(self) -> None:
        for _ in range(self.num_trials):
            x, y = random_ringrootroot2plus2(), random_ringrootroot2plus2()
            x_power, y_power = rand_integer_values(2, 0, 32)
            x_ratio = AlgebraicIntegerOverRootRoot2Plus2(x, x_power)
            y_ratio = AlgebraicIntegerOverRootRoot2Plus2(y, y_power)
            rr2p2 = RingRootRoot2Plus2([0, 0, 1, 0])
            rr2p2 = rr2p2.to_float()
            x_float = x.to_float() / (rr2p2 ** x_power)
            y_float = y.to_float() / (rr2p2 ** y_power)
            z_float = x_float + y_float
            z_ratio = x_ratio + y_ratio
            assert isclose(z_ratio.to_float(), z_float, rel_tol=eps)

    def test_mul(self) -> None:
        for _ in range(self.num_trials):
            x, y = random_ringrootroot2plus2(), random_ringrootroot2plus2()
            x_power, y_power = rand_integer_values(2, 0, 32)
            x_ratio = AlgebraicIntegerOverRootRoot2Plus2(x, x_power)
            y_ratio = AlgebraicIntegerOverRootRoot2Plus2(y, y_power)
            rr2p2 = RingRootRoot2Plus2([0, 0, 1, 0])
            rr2p2 = rr2p2.to_float()
            x_float = x.to_float() / (rr2p2 ** x_power)
            y_float = y.to_float() / (rr2p2 ** y_power)
            z_float = x_float * y_float
            z_ratio = x_ratio * y_ratio
            assert isclose(z_ratio.to_float(), z_float, rel_tol=eps)
            
    def test_simplify_r2(self) -> None:
        for _ in range(self.num_trials):
            x = RingRoot2([1, 0])
            power = randint(0, 100)
            for _ in range(power):
                x = x * RingRoot2([0, 1])
            ratio = AlgebraicIntegerOverRoot2(x, power)
            ratio.simplify()
            assert ratio.numerator.values == [1, 0]
            assert ratio.denominator_power == 0

    def test_add_r2(self) -> None:
        for _ in range(self.num_trials):
            x, y = random_ringroot2(), random_ringroot2()
            x_power, y_power = rand_integer_values(2, 0, 32)
            x_ratio = AlgebraicIntegerOverRoot2(x, x_power)
            y_ratio = AlgebraicIntegerOverRoot2(y, y_power)
            r2 = RingRoot2([0, 1])
            r2 = r2.to_float()
            x_float = x.to_float() / (r2 ** x_power)
            y_float = y.to_float() / (r2 ** y_power)
            z_float = x_float + y_float
            z_ratio = x_ratio + y_ratio
            assert isclose(z_ratio.to_float(), z_float, rel_tol=eps)

    def test_mul_r2(self) -> None:
        for _ in range(self.num_trials):
            x, y = random_ringroot2(), random_ringroot2()
            x_power, y_power = rand_integer_values(2, 0, 32)
            x_ratio = AlgebraicIntegerOverRoot2(x, x_power)
            y_ratio = AlgebraicIntegerOverRoot2(y, y_power)
            r2 = RingRoot2([0, 1])
            r2 = r2.to_float()
            x_float = x.to_float() / (r2 ** x_power)
            y_float = y.to_float() / (r2 ** y_power)
            z_float = x_float * y_float
            z_ratio = x_ratio * y_ratio
            assert isclose(z_ratio.to_float(), z_float, rel_tol=eps)

    def test_add_mixed(self) -> None:
        for i in range(self.num_trials):
            x, y = random_ringrootroot2plus2(), random_ringroot2()
            x_power, y_power = rand_integer_values(2, 0, 32)
            x_ratio = AlgebraicIntegerOverRootRoot2Plus2(x, x_power)
            y_ratio = AlgebraicIntegerOverRoot2(y, y_power)
            r2 = RingRoot2([0, 1])
            r2 = r2.to_float()
            rr2p2 = RingRootRoot2Plus2([0, 0, 1, 0])
            rr2p2 = rr2p2.to_float()
            x_float = x.to_float() / (rr2p2 ** x_power)
            y_float = y.to_float() / (r2 ** y_power)
            z_float = x_float + y_float
            if i % 2 == 0:
                z_ratio = x_ratio + y_ratio
            else:
                z_ratio = y_ratio + x_ratio
            assert isclose(z_ratio.to_float(), z_float, rel_tol=eps)

    def test_mul_mixed(self) -> None:
        for i in range(self.num_trials):
            x, y = random_ringrootroot2plus2(), random_ringroot2()
            x_power, y_power = rand_integer_values(2, 0, 32)
            x_ratio = AlgebraicIntegerOverRootRoot2Plus2(x, x_power)
            y_ratio = AlgebraicIntegerOverRoot2(y, y_power)
            r2 = RingRoot2([0, 1])
            r2 = r2.to_float()
            rr2p2 = RingRootRoot2Plus2([0, 0, 1, 0])
            rr2p2 = rr2p2.to_float()
            x_float = x.to_float() / (rr2p2 ** x_power)
            y_float = y.to_float() / (r2 ** y_power)
            z_float = x_float * y_float
            if i % 2 == 0:
                z_ratio = x_ratio * y_ratio
            else:
                z_ratio = y_ratio * x_ratio
            assert isclose(z_ratio.to_float(), z_float, rel_tol=eps)
    
    def test_from_dyadic_rr2p2(self) -> None:
        n = 8
        for _ in range(self.num_trials):
            values = random_dyadic_coefficients(n)
            power = rand_integer_values(1, 0, 50)[0]
            dyadic = DyadicComplexNumber(values, power)
            ratio = AlgebraicIntegerOverRootRoot2Plus2.from_dyadic(dyadic)
            diff = dyadic.to_complex().real - ratio.to_float()
            assert isclose(diff, 0.0, abs_tol=1e-3)
        
    def test_from_dyadic_r2(self) -> None:
        n = 4
        for _ in range(self.num_trials):
            values = random_dyadic_coefficients(n)
            power = rand_integer_values(1, 0, 50)[0]
            dyadic = DyadicComplexNumber(values, power)
            ratio = AlgebraicIntegerOverRoot2.from_dyadic(dyadic)
            diff = dyadic.to_complex().real - ratio.to_float()
            assert isclose(diff, 0.0, abs_tol=1e-3)
    
    def test_generic_ratio(self) -> None:
        for _ in range(self.num_trials):
            # int int
            x1, x2, y1, y2 = rand_integer_values(4)
            x = IntegerRatio(x1, x2)
            y = IntegerRatio(y1, y2)
            assert isclose((x + y).to_float(), ((x1 / x2) + (y1 / y2)))
            assert isclose((x - y).to_float(), ((x1 / x2) - (y1 / y2)))
            assert isclose((x * y).to_float(), ((x1 / x2) * (y1 / y2)))
            assert isclose((x * y.inverse()).to_float(), (x1 * y2 / x2 / y1))

            # aint int
            x1, x2 = random_ringroot2(), random_ringroot2()
            x = IntegerRatio(x1, x2)
            x1, x2 = x1.to_float(), x2.to_float()
            assert isclose((x + y).to_float(), ((x1 / x2) + (y1 / y2)))
            assert isclose((x - y).to_float(), ((x1 / x2) - (y1 / y2)))
            assert isclose((x * y).to_float(), ((x1 / x2) * (y1 / y2)))
            assert isclose((x * y.inverse()).to_float(), (x1 * y2 / x2 / y1))

            # int aint
            y1, y2 = random_ringrootroot2plus2(), random_ringrootroot2plus2()
            y = IntegerRatio(y1, y2)
            y1, y2 = y1.to_float(), y2.to_float()
            assert isclose((x + y).to_float(), ((x1 / x2) + (y1 / y2)))
            assert isclose((x - y).to_float(), ((x1 / x2) - (y1 / y2)))
            assert isclose((x * y).to_float(), ((x1 / x2) * (y1 / y2)))
            assert isclose((x * y.inverse()).to_float(), (x1 * y2 / x2 / y1))

            # aint aint
            x1, x2 = random_ringroot2(), random_ringroot2()
            y1, y2 = random_ringrootroot2plus2(), random_ringrootroot2plus2()
            x = IntegerRatio(x1, x2)
            y = IntegerRatio(y1, y2)
            x1, x2 = x1.to_float(), x2.to_float()
            y1, y2 = y1.to_float(), y2.to_float()
            assert isclose((x + y).to_float(), ((x1 / x2) + (y1 / y2)))
            assert isclose((x - y).to_float(), ((x1 / x2) - (y1 / y2)))
            assert isclose((x * y).to_float(), ((x1 / x2) * (y1 / y2)))
            assert isclose((x * y.inverse()).to_float(), (x1 * y2 / x2 / y1))