from typing import Sequence

from random import randint

from math import cos
from math import isclose
from math import pi
from math import sin

from cyclosynth.algebra import RingRoot2
from cyclosynth.algebra import RingRootRoot2Plus2

from cyclosynth.utils import discrete_cos
from cyclosynth.utils import discrete_sin
from cyclosynth.utils import dyadic_cos
from cyclosynth.utils import dyadic_sin
from cyclosynth.utils import is_divisible_by_root2
from cyclosynth.utils import is_divisible_by_rootroot2plus2


def rand_integer_values(
    n: int,
    min_val: int = -1_000_000_000_000_000,
    max_val: int = 1_000_000_000_000_000,
) -> list[int]:
    return [randint(min_val, max_val) for _ in range(n)]


class TestUtils:

    def test_fixed_divisibility_rootroot2plus2(self) -> None:
        number_1 = RingRootRoot2Plus2([1, 0, 0, 0])  # 1
        number_2 = RingRootRoot2Plus2([0, 1, 0, 0])  # sqrt(2)
        number_3 = RingRootRoot2Plus2([0, 0, -1, 1])  # /(sqrt(2+sqrt(2)))
        number_4 = RingRootRoot2Plus2([-1, 1, 0, 0])  # /(sqrt(2+sqrt(2)))^2
        assert not is_divisible_by_rootroot2plus2(number_1)
        assert is_divisible_by_rootroot2plus2(number_2)
        assert is_divisible_by_rootroot2plus2(number_3)
        assert not is_divisible_by_rootroot2plus2(number_4)

    def test_divisibility_rootroot2plus2(self) -> None:
        values = rand_integer_values(4)
        number = RingRootRoot2Plus2(values)
        factor_1 = RingRootRoot2Plus2([0, 0, randint(1, 100), 0])
        test_num_1 = number * factor_1
        factor_2 = RingRootRoot2Plus2([0, 0, 0, randint(1, 100)])
        test_num_2 = number * factor_2
        assert is_divisible_by_rootroot2plus2(test_num_1)
        assert is_divisible_by_rootroot2plus2(test_num_2)

    def test_fixed_divisibility_root2(self) -> None:
        number_1 = RingRootRoot2Plus2([1, 0, 0, 0])  # 1
        number_2 = RingRootRoot2Plus2([0, 1, 0, 0])  # sqrt(2)
        number_3 = RingRootRoot2Plus2([0, 0, -1, 1])  # /(sqrt(2+sqrt(2)))
        number_4 = RingRootRoot2Plus2([-1, 1, 0, 0])  # /(sqrt(2+sqrt(2)))^2
        assert not is_divisible_by_rootroot2plus2(number_1)
        assert is_divisible_by_rootroot2plus2(number_2)
        assert is_divisible_by_rootroot2plus2(number_3)
        assert not is_divisible_by_rootroot2plus2(number_4)
        number_5 = RingRoot2([1, 0])  # 1
        number_6 = RingRoot2([0, 1])  # sqrt(2)
        number_7 = RingRoot2([3, 0])  # 3
        number_8 = RingRoot2([4, 1])  # 4 + sqrt(2)
        assert not is_divisible_by_root2(number_5)
        assert is_divisible_by_root2(number_6)
        assert not is_divisible_by_root2(number_7)
        assert is_divisible_by_root2(number_8)
    
    def test_discrete_sin(self) -> None:
        sin4_f = sin(pi / 4)
        sin8_f = sin(pi / 8)
        sin4_d = discrete_sin(4).to_float().real
        sin8_d = discrete_sin(8).to_float().real
        assert isclose(sin4_f, sin4_d, abs_tol=1e-6)
        assert isclose(sin8_f, sin8_d, abs_tol=1e-6)
    
    def test_discrete_cos(self) -> None:
        cos4_f = cos(pi / 4)
        cos8_f = cos(pi / 8)
        cos4_d = discrete_cos(4).to_float().real
        cos8_d = discrete_cos(8).to_float().real
        assert isclose(cos4_f, cos4_d, abs_tol=1e-6)
        assert isclose(cos8_f, cos8_d, abs_tol=1e-6)
    
    def test_dyadic_sin(self) -> None:
        for k in range(-32, 32):
            for n in range(2, 10):
                n = 2 ** n
                dsin = dyadic_sin(k, n)
                dy = dsin.to_complex().real
                np = sin(k * pi / n)
                assert isclose(dy, np, abs_tol=1e-6)

    def test_dyadic_cos(self) -> None:
        for k in range(-32, 32):
            for n in range(2, 10):
                n = 2 ** n
                dcos = dyadic_cos(k, n)
                dy = dcos.to_complex().real
                np = cos(k * pi / n)
                assert isclose(dy, np, abs_tol=1e-6)