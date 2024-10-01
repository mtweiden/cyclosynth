from random import seed
from random import randint

from math import isclose

from cyclosynth.algebra import RingRoot2
from cyclosynth.algebra import RingRootRoot2Plus2
from cyclosynth.general_ratio import GeneralIntegerRatio


eps = 1e-6


def rand_integer_values(
    n: int,
    min_val: int = -1_000_000_000_000_000,
    max_val: int = 1_000_000_000_000_000,
) -> list[int]:
    return [randint(min_val, max_val) for _ in range(n)]


def random_ringroot2() -> RingRoot2:
    return RingRoot2(rand_integer_values(2))


def random_ringrootroot2plus2() -> RingRootRoot2Plus2:
    return RingRootRoot2Plus2(rand_integer_values(4))


class TestGeneralIntegerRatios:

    num_trials = 1000

    # format: numerator - denominator - number
    def test_ratio_mul_root2_none_int(self) -> None:
        for _ in range(self.num_trials):
            numerator = random_ringroot2()
            number = randint(-1_000_000, 1_000_000)
            ratio = GeneralIntegerRatio(numerator)
            ratio = ratio * number
            n_num = numerator.to_float().real
            n_numb = float(number)
            n_ratio = n_num * n_numb
            assert isclose(ratio.to_float().real, n_ratio, rel_tol=eps)

    def test_ratio_mul_root2_none_root2(self) -> None:
        for _ in range(self.num_trials):
            numerator = random_ringroot2()
            number = random_ringroot2()
            ratio = GeneralIntegerRatio(numerator)
            ratio = ratio * number
            n_num = numerator.to_float()
            n_numb = float(number)
            n_ratio = n_num * n_numb
            assert isclose(ratio.to_float().real, n_ratio, rel_tol=eps)

    def test_ratio_mul_root2_root2_int(self) -> None:
        for _ in range(self.num_trials):
            numerator = random_ringroot2()
            denominator = random_ringroot2()
            number = randint(-1_000_000, 1_000_000)
            n_num = numerator.to_float()
            n_den = denominator.to_float()
            n_numb = float(number)
            ratio = GeneralIntegerRatio(numerator, denominator)
            ratio = ratio * number
            n_ratio = n_num / n_den * n_numb
            assert isclose(ratio.to_float().real, n_ratio, rel_tol=eps)

    def test_ratio_mul_root2_root2_root2(self) -> None:
        for _ in range(self.num_trials):
            numerator = random_ringroot2()
            denominator = random_ringroot2()
            number = random_ringroot2()
            ratio = GeneralIntegerRatio(numerator, denominator)
            n_num = numerator.to_float()
            n_den = denominator.to_float()
            n_numb = float(number)
            n_ratio = n_num / n_den * n_numb
            ratio = ratio * number
            assert isclose(ratio.to_float().real, n_ratio, rel_tol=eps)

    def test_ratio_mul_root2_rr2p2_int(self) -> None:
        for _ in range(self.num_trials):
            numerator = random_ringroot2()
            denominator = random_ringrootroot2plus2()
            number = randint(-1_000_000, 1_000_000)
            ratio = GeneralIntegerRatio(numerator, denominator)
            ratio = ratio * number
            n_num = numerator.to_float()
            n_den = denominator.to_float()
            n_numb = float(number)
            n_ratio = n_num / n_den * n_numb
            assert isclose(ratio.to_float().real, n_ratio, rel_tol=eps)

    def test_ratio_mul_root2_rr2p2_root2(self) -> None:
        for _ in range(self.num_trials):
            numerator = random_ringroot2()
            denominator = random_ringrootroot2plus2()
            number = random_ringroot2()
            ratio = GeneralIntegerRatio(numerator, denominator)
            ratio = ratio * number
            n_num = numerator.to_float()
            n_den = denominator.to_float()
            n_numb = float(number)
            n_ratio = n_num / n_den * n_numb
            assert isclose(ratio.to_float().real, n_ratio, rel_tol=eps)

    # format: numerator - denominator - number
    def test_ratio_mult_rr2p2_none_int(self) -> None:
        for _ in range(self.num_trials):
            numerator = random_ringrootroot2plus2()
            number = randint(-1_000_000, 1_000_000)
            ratio = GeneralIntegerRatio(numerator)
            ratio = ratio * number
            n_num = numerator.to_float()
            n_numb = float(number)
            n_ratio = n_num * n_numb
            assert isclose(ratio.to_float().real, n_ratio, rel_tol=eps)

    def test_ratio_mult_rr2p2_none_root2(self) -> None:
        for _ in range(self.num_trials):
            numerator = random_ringrootroot2plus2()
            number = random_ringroot2()
            ratio = GeneralIntegerRatio(numerator)
            ratio = ratio * number
            n_num = numerator.to_float()
            n_numb = float(number)
            n_ratio = n_num * n_numb
            assert isclose(ratio.to_float().real, n_ratio, rel_tol=eps)

    def test_ratio_mul_rr2p2_none_rr2p2(self) -> None:
        for _ in range(self.num_trials):
            numerator = random_ringrootroot2plus2()
            number = random_ringrootroot2plus2()
            ratio = GeneralIntegerRatio(numerator)
            ratio = ratio * number
            n_num = numerator.to_float()
            n_numb = float(number)
            n_ratio = n_num * n_numb
            assert isclose(ratio.to_float().real, n_ratio, rel_tol=eps)

    def test_ratio_mult_rr2p2_root2_int(self) -> None:
        for _ in range(self.num_trials):
            numerator = random_ringrootroot2plus2()
            denominator = random_ringroot2()
            number = randint(-1_000_000, 1_000_000)
            ratio = GeneralIntegerRatio(numerator, denominator)
            ratio = ratio * number
            n_num = numerator.to_float()
            n_den = denominator.to_float()
            n_numb = float(number)
            n_ratio = n_num / n_den * n_numb
            assert isclose(ratio.to_float().real, n_ratio, rel_tol=eps)

    def test_ratio_mult_rr2p2_root2_root2(self) -> None:
        for _ in range(self.num_trials):
            numerator = random_ringrootroot2plus2()
            denominator = random_ringroot2()
            number = random_ringroot2()
            ratio = GeneralIntegerRatio(numerator, denominator)
            ratio = ratio * number
            n_num = numerator.to_float()
            n_den = denominator.to_float()
            n_numb = float(number)
            n_ratio = n_num / n_den * n_numb
            assert isclose(ratio.to_float().real, n_ratio, rel_tol=eps)

    def test_ratio_mult_rr2p2_root2_rr2p2(self) -> None:
        for _ in range(self.num_trials):
            numerator = random_ringrootroot2plus2()
            denominator = random_ringroot2()
            number = random_ringrootroot2plus2()
            ratio = GeneralIntegerRatio(numerator, denominator)
            ratio = ratio * number
            n_num = numerator.to_float()
            n_den = denominator.to_float()
            n_numb = float(number)
            n_ratio = n_num / n_den * n_numb
            assert isclose(ratio.to_float().real, n_ratio, rel_tol=eps)

    def test_ratio_mult_rr2p2_rr2p2_int(self) -> None:
        for _ in range(self.num_trials):
            numerator = random_ringrootroot2plus2()
            denominator = random_ringrootroot2plus2()
            number = randint(-1_000_000, 1_000_000)
            ratio = GeneralIntegerRatio(numerator, denominator)
            ratio = ratio * number
            n_num = numerator.to_float()
            n_den = denominator.to_float()
            n_numb = float(number)
            n_ratio = n_num / n_den * n_numb
            assert isclose(ratio.to_float().real, n_ratio, rel_tol=eps)

    def test_ratio_mult_rr2p2_rr2p2_root2(self) -> None:
        for _ in range(self.num_trials):
            numerator = random_ringrootroot2plus2()
            denominator = random_ringrootroot2plus2()
            number = random_ringroot2()
            ratio = GeneralIntegerRatio(numerator, denominator)
            ratio = ratio * number
            n_num = numerator.to_float()
            n_den = denominator.to_float()
            n_numb = float(number)
            n_ratio = n_num / n_den * n_numb
            assert isclose(ratio.to_float().real, n_ratio, rel_tol=eps)

    def test_ratio_mult_rr2p2_rr2p2_rr2p2(self) -> None:
        for _ in range(self.num_trials):
            numerator = random_ringrootroot2plus2()
            denominator = random_ringrootroot2plus2()
            number = random_ringrootroot2plus2()
            ratio = GeneralIntegerRatio(numerator, denominator)
            ratio = ratio * number
            n_num = numerator.to_float()
            n_den = denominator.to_float()
            n_numb = number.to_float()
            n_ratio = n_num / n_den * n_numb
            assert isclose(ratio.to_float().real, n_ratio, rel_tol=eps)

    # format: numerator - denominator - number
    def test_ratio_mul_root2_none_int(self) -> None:
        for _ in range(self.num_trials):
            numerator = random_ringroot2()
            number = randint(-1_000_000, 1_000_000)
            ratio = GeneralIntegerRatio(numerator)
            ratio = ratio / number
            n_num = numerator.to_float().real
            n_numb = float(number)
            n_ratio = n_num / n_numb
            assert isclose(ratio.to_float().real, n_ratio, rel_tol=eps)

    def test_ratio_mul_root2_none_root2(self) -> None:
        for _ in range(self.num_trials):
            numerator = random_ringroot2()
            number = random_ringroot2()
            ratio = GeneralIntegerRatio(numerator)
            ratio = ratio / number
            n_num = numerator.to_float()
            n_numb = float(number)
            n_ratio = n_num / n_numb
            assert isclose(ratio.to_float().real, n_ratio, rel_tol=eps)

    def test_ratio_div_root2_root2_int(self) -> None:
        for _ in range(self.num_trials):
            numerator = random_ringroot2()
            denominator = random_ringroot2()
            number = randint(-1_000_000, 1_000_000)
            n_num = numerator.to_float()
            n_den = denominator.to_float()
            n_numb = float(number)
            ratio = GeneralIntegerRatio(numerator, denominator)
            ratio = ratio / number
            n_ratio = n_num / n_den / n_numb
            assert isclose(ratio.to_float().real, n_ratio, rel_tol=eps)

    def test_ratio_div_root2_root2_root2(self) -> None:
        for _ in range(self.num_trials):
            numerator = random_ringroot2()
            denominator = random_ringroot2()
            number = random_ringroot2()
            ratio = GeneralIntegerRatio(numerator, denominator)
            ratio = ratio / number
            n_num = numerator.to_float()
            n_den = denominator.to_float()
            n_numb = float(number)
            n_ratio = n_num / n_den / n_numb
            assert isclose(ratio.to_float().real, n_ratio, rel_tol=eps)

    def test_ratio_mul_root2_rr2p2_int(self) -> None:
        for _ in range(self.num_trials):
            numerator = random_ringroot2()
            denominator = random_ringrootroot2plus2()
            number = randint(-1_000_000, 1_000_000)
            ratio = GeneralIntegerRatio(numerator, denominator)
            ratio = ratio / number
            n_num = numerator.to_float()
            n_den = denominator.to_float()
            n_numb = float(number)
            n_ratio = n_num / n_den / n_numb
            assert isclose(ratio.to_float().real, n_ratio, rel_tol=eps)

    def test_ratio_mul_root2_rr2p2_root2(self) -> None:
        for _ in range(self.num_trials):
            numerator = random_ringroot2()
            denominator = random_ringrootroot2plus2()
            number = random_ringroot2()
            ratio = GeneralIntegerRatio(numerator, denominator)
            ratio = ratio / number
            n_num = numerator.to_float()
            n_den = denominator.to_float()
            n_numb = float(number)
            n_ratio = n_num / n_den / n_numb
            assert isclose(ratio.to_float().real, n_ratio, rel_tol=eps)

    # format: numerator - denominator - number
    def test_ratio_div_rr2p2_none_int(self) -> None:
        for _ in range(self.num_trials):
            numerator = random_ringrootroot2plus2()
            number = randint(-1_000_000, 1_000_000)
            ratio = GeneralIntegerRatio(numerator)
            ratio = ratio / number
            n_num = numerator.to_float()
            n_numb = float(number)
            n_ratio = n_num / n_numb
            assert isclose(ratio.to_float().real, n_ratio, rel_tol=eps)

    def test_ratio_div_rr2p2_none_root2(self) -> None:
        for _ in range(self.num_trials):
            numerator = random_ringrootroot2plus2()
            number = random_ringroot2()
            ratio = GeneralIntegerRatio(numerator)
            ratio = ratio / number
            n_num = numerator.to_float()
            n_numb = float(number)
            n_ratio = n_num / n_numb
            assert isclose(ratio.to_float().real, n_ratio, rel_tol=eps)

    def test_ratio_mul_rr2p2_none_rr2p2(self) -> None:
        for _ in range(self.num_trials):
            numerator = random_ringrootroot2plus2()
            number = random_ringrootroot2plus2()
            ratio = GeneralIntegerRatio(numerator)
            ratio = ratio / number
            n_num = numerator.to_float()
            n_numb = float(number)
            n_ratio = n_num / n_numb
            assert isclose(ratio.to_float().real, n_ratio, rel_tol=eps)

    def test_ratio_div_rr2p2_root2_int(self) -> None:
        for _ in range(self.num_trials):
            numerator = random_ringrootroot2plus2()
            denominator = random_ringroot2()
            number = randint(-1_000_000, 1_000_000)
            ratio = GeneralIntegerRatio(numerator, denominator)
            ratio = ratio / number
            n_num = numerator.to_float()
            n_den = denominator.to_float()
            n_numb = float(number)
            n_ratio = n_num / n_den / n_numb
            assert isclose(ratio.to_float().real, n_ratio, rel_tol=eps)

    def test_ratio_div_rr2p2_root2_root2(self) -> None:
        for _ in range(self.num_trials):
            numerator = random_ringrootroot2plus2()
            denominator = random_ringroot2()
            number = random_ringroot2()
            ratio = GeneralIntegerRatio(numerator, denominator)
            ratio = ratio / number
            n_num = numerator.to_float()
            n_den = denominator.to_float()
            n_numb = float(number)
            n_ratio = n_num / n_den / n_numb
            assert isclose(ratio.to_float().real, n_ratio, rel_tol=eps)

    def test_ratio_div_rr2p2_rr2p2_int(self) -> None:
        for _ in range(self.num_trials):
            numerator = random_ringrootroot2plus2()
            denominator = random_ringrootroot2plus2()
            number = randint(-1_000_000, 1_000_000)
            ratio = GeneralIntegerRatio(numerator, denominator)
            ratio = ratio / number
            n_num = numerator.to_float()
            n_den = denominator.to_float()
            n_numb = float(number)
            n_ratio = n_num / n_den / n_numb
            assert isclose(ratio.to_float().real, n_ratio, rel_tol=eps)

    def test_ratio_div_rr2p2_rr2p2_root2(self) -> None:
        for _ in range(self.num_trials):
            numerator = random_ringrootroot2plus2()
            denominator = random_ringrootroot2plus2()
            number = random_ringroot2()
            ratio = GeneralIntegerRatio(numerator, denominator)
            ratio = ratio / number
            n_num = numerator.to_float()
            n_den = denominator.to_float()
            n_numb = float(number)
            n_ratio = n_num / n_den / n_numb
            assert isclose(ratio.to_float().real, n_ratio, rel_tol=eps)

    def test_ratio_div_rr2p2_rr2p2_rr2p2(self) -> None:
        for _ in range(self.num_trials):
            numerator = random_ringrootroot2plus2()
            denominator = random_ringrootroot2plus2()
            number = random_ringrootroot2plus2()
            ratio = GeneralIntegerRatio(numerator, denominator)
            ratio = ratio / number
            n_num = numerator.to_float()
            n_den = denominator.to_float()
            n_numb = number.to_float()
            n_ratio = n_num / n_den / n_numb
            assert isclose(ratio.to_float().real, n_ratio, rel_tol=eps)

    def test_ratio_simplify(self) -> None:
        for _ in range(self.num_trials):
            numerator = random_ringrootroot2plus2()
            denominator = random_ringrootroot2plus2()
            multiplier = randint(1, 1_000)
            numerator = numerator * multiplier
            denominator = denominator * multiplier
            ratio = GeneralIntegerRatio(numerator, denominator)
            f_ratio = ratio.to_float()

            ratio.simplify()
            f_simplified = ratio.to_float()

            assert isclose(f_ratio, f_simplified, rel_tol=eps)
