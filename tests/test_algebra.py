from typing import Sequence

from random import randint

from cmath import exp
from cmath import isclose

from math import sqrt
from math import pi

from cyclosynth.algebra import RingRoot2
from cyclosynth.algebra import RingRootRoot2Plus2
from cyclosynth.algebra import DyadicComplexNumber


def rand_integer_values(
    n: int,
    min_val: int = -1_000_000_000_000_000,
    max_val: int = 1_000_000_000_000_000,
) -> list[int]:
    return [randint(min_val, max_val) for _ in range(n)]


def to_float(values: Sequence[int]) -> float:
    f_value = 0.0
    for i in range(len(values)):
        if i == 0:
            rad = 1
        elif i == 1:
            rad = sqrt(2)
        elif i == 2:
            rad = sqrt(sqrt(2) + 2)
        elif i == 3:
            rad = sqrt(2) * sqrt(sqrt(2) + 2)
        else:
            raise ValueError(f'Unknown radical for {i}.')
        f_value += values[i] * rad
    return f_value


class TestAlgebra:

    num_trials = 1000

    def test_ringroot2(self) -> None:
        for _ in range(self.num_trials):
            values = rand_integer_values(2)
            RingRoot2(values)

    def test_ringrootroot2p2(self) -> None:
        for _ in range(self.num_trials):
            values = rand_integer_values(4)
            RingRootRoot2Plus2(values)

    def test_ringroot2_add(self) -> None:
        for _ in range(self.num_trials):
            n = 2
            values_a = rand_integer_values(n)
            values_b = rand_integer_values(n)
            values_c = [values_a[i] + values_b[i] for i in range(n)]
            int_a = RingRoot2(values_a)
            int_b = RingRoot2(values_b)
            int_c = int_a + int_b
            float_values = to_float(values_c)
            float_int = int_c.to_float().real
            assert isclose(float_values, float_int, rel_tol=1e-6)

    def test_ringroot2_mul(self) -> None:
        for _ in range(self.num_trials):
            n = 2
            values_a = rand_integer_values(n)
            values_b = rand_integer_values(n)
            int_a = RingRoot2(values_a)
            int_b = RingRoot2(values_b)
            int_c = int_a * int_b
        # TODO: Check a real example

    def test_ringrootroot2p2_add(self) -> None:
        n = 4
        for _ in range(self.num_trials):
            values_a = rand_integer_values(n)
            values_b = rand_integer_values(n)
            values_c = [values_a[i] + values_b[i] for i in range(n)]
            int_a = RingRootRoot2Plus2(values_a)
            int_b = RingRootRoot2Plus2(values_b)
            int_c = int_a + int_b
            float_values = to_float(values_c)
            float_int = int_c.to_float().real
            assert isclose(float_values, float_int, rel_tol=1e-6)

    def test_ringrootroot2p2_mul(self) -> None:
        n = 4
        for _ in range(self.num_trials):
            values_a = rand_integer_values(n)
            values_b = rand_integer_values(n)
            int_a = RingRootRoot2Plus2(values_a)
            int_b = RingRootRoot2Plus2(values_b)
            int_c = int_a * int_b
            # TODO: Check a real example

    def test_dyadic_add(self) -> None:
        n, m = 8, 16
        for _ in range(self.num_trials):
            # Same base
            values_a = rand_integer_values(n)
            values_b = rand_integer_values(n)
            power_a = rand_integer_values(1, 0, 50)[0]
            power_b = rand_integer_values(1, 0, 50)[0]
            total_a, total_b = 0 * 1j, 0 * 1j
            for i, (coeff_a, coeff_b) in enumerate(zip(values_a, values_b)):
                total_a += coeff_a * exp(1j * pi * i / n)
                total_b += coeff_b * exp(1j * pi * i / n)
            total_a = total_a / (2 ** power_a)
            total_b = total_b / (2 ** power_b)
            total_c = total_a + total_b
            int_a = DyadicComplexNumber(values_a, power_a)
            int_b = DyadicComplexNumber(values_b, power_b)
            int_c = int_a + int_b
            float_c = int_c.to_complex()
            assert isclose(total_c, float_c, rel_tol=1e-6)

            # Mixed base
            values_b = rand_integer_values(m)
            power_b = rand_integer_values(1, 0, 50)[0]
            total_b = 0 * 1j
            for i, coeff_b in enumerate(values_b):
                total_b += coeff_b * exp(1j * pi * i / m)
            total_b = total_b / (2 ** power_b)
            total_c = total_a + total_b
            int_b = DyadicComplexNumber(values_b, power_b)
            int_c = int_a + int_b
            float_c = int_c.to_complex()
            assert isclose(total_c, float_c, rel_tol=1e-6)

    def test_dyadic_sub(self) -> None:
        n, m = 8, 16
        for _ in range(self.num_trials):
            # Same base
            values_a = rand_integer_values(n)
            values_b = rand_integer_values(n)
            power_a = rand_integer_values(1, 0, 50)[0]
            power_b = rand_integer_values(1, 0, 50)[0]
            total_a, total_b = 0 * 1j, 0 * 1j
            for i, (coeff_a, coeff_b) in enumerate(zip(values_a, values_b)):
                total_a += coeff_a * exp(1j * pi * i / n)
                total_b += coeff_b * exp(1j * pi * i / n)
            total_a = total_a / (2 ** power_a)
            total_b = total_b / (2 ** power_b)
            total_c = total_a - total_b
            int_a = DyadicComplexNumber(values_a, power_a)
            int_b = DyadicComplexNumber(values_b, power_b)
            int_c = int_a - int_b
            float_c = int_c.to_complex()
            assert isclose(total_c, float_c, rel_tol=1e-6)

            # Mixed base
            values_b = rand_integer_values(m)
            power_b = rand_integer_values(1, 0, 50)[0]
            total_b = 0 * 1j
            for i, coeff_b in enumerate(values_b):
                total_b += coeff_b * exp(1j * pi * i / m)
            total_b = total_b / (2 ** power_b)
            total_c = total_a - total_b
            int_b = DyadicComplexNumber(values_b, power_b)
            int_c = int_a - int_b
            float_c = int_c.to_complex()
            assert isclose(total_c, float_c, rel_tol=1e-6)

    def test_dyadic_mul(self) -> None:
        n, m = 8, 16
        for _ in range(self.num_trials):
            # Same base
            values_a = rand_integer_values(n)
            values_b = rand_integer_values(n)
            power_a = rand_integer_values(1, 0, 50)[0]
            power_b = rand_integer_values(1, 0, 50)[0]
            total_a, total_b = 0 * 1j, 0 * 1j
            for i, (coeff_a, coeff_b) in enumerate(zip(values_a, values_b)):
                total_a += coeff_a * exp(1j * pi * i / n)
                total_b += coeff_b * exp(1j * pi * i / n)
            total_a = total_a / (2 ** power_a)
            total_b = total_b / (2 ** power_b)
            total_c = total_a * total_b
            int_a = DyadicComplexNumber(values_a, power_a)
            int_b = DyadicComplexNumber(values_b, power_b)
            int_c = int_a * int_b
            float_c = int_c.to_complex()
            assert isclose(total_c, float_c, rel_tol=1e-6)

            # Mixed base
            values_b = rand_integer_values(m)
            power_b = rand_integer_values(1, 0, 50)[0]
            total_b = 0 * 1j
            for i, coeff_b in enumerate(values_b):
                total_b += coeff_b * exp(1j * pi * i / m)
            total_b = total_b / (2 ** power_b)
            total_c = total_a * total_b
            int_b = DyadicComplexNumber(values_b, power_b)
            int_c = int_a * int_b
            float_c = int_c.to_complex()
            assert isclose(total_c, float_c, rel_tol=1e-6)
    
    def test_dyadic_conjugate(self) -> None:
        for _ in range(self.num_trials):
            for n in range(2, 10):
                n = 2 ** n
                values = [randint(-100, 100)] * n
                denominator_power = randint(0, 8)
                dyadic = DyadicComplexNumber(values, denominator_power)
                conj_n = dyadic.to_complex().conjugate()
                conj = dyadic.conj().to_complex()
                assert isclose(conj_n, conj, rel_tol=1e-6)