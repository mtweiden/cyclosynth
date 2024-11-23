from random import randint

from cyclosynth.algebra import RingRoot2
from cyclosynth.ratio import IntegerRatio
from cyclosynth.operator import Operator


def random_integer_ratio() -> int:
    a, b = randint(-1000000, 1000000), randint(-1000000, 1000000)
    x, y = randint(-1000000, 1000000), randint(-1000000, 1000000)
    w = RingRoot2([a, b])
    v = RingRoot2([x, y])
    return IntegerRatio(w, v)


def random_operator() -> Operator:
    vals = [random_integer_ratio() for _ in range(4)]
    return Operator(vals)


class TestOperator:

    num_trials = 100

    def test_operator(self) -> None:
        for _ in range(self.num_trials):
            a = random_operator()
            a_inv = a.inv()
            identity = a * a_inv
            assert identity[0, 0].to_float() == 1
            assert identity[0, 1].to_float() == 0
            assert identity[1, 1].to_float() == 1
            assert identity[1, 0].to_float() == 0
