from random import randint

from cyclosynth.algebra import RingRoot2
from cyclosynth.ratio import IntegerRatio
from cyclosynth.operator_ import Operator
from cyclosynth.matrix import Vector

from numpy import isclose


def random_integer_ratio() -> int:
    a, b = randint(-1000000, 1000000), randint(-1000000, 1000000)
    x, y = randint(-1000000, 1000000), randint(-1000000, 1000000)
    w = RingRoot2([a, b])
    v = RingRoot2([x, y])
    return IntegerRatio(w, v)


def random_operator() -> Operator:
    vals = [random_integer_ratio() for _ in range(4)]
    return Operator(vals)


def random_vector() -> Vector:
    vals = [random_integer_ratio() for _ in range(2)]
    return Vector(vals)


def numeric_mat_mul(operator: Operator, vector: Vector) -> tuple[float]:
    a, b, c, d = [float(v.to_float()) for v in operator.values]
    x, y = [float(v.to_float()) for v in vector.values]
    w = a * x + b * y
    z = c * x + d * y
    return [w, z]

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

    def test_vector_act_on(self) -> None:
        for _ in range(self.num_trials):
            op = random_operator()
            vec = random_vector()
            result = op.act_on(vec)
            numeric = numeric_mat_mul(op, vec)
            assert isclose(result.values[0].to_float(), numeric[0], rtol=1e-4)
            assert isclose(result.values[1].to_float(), numeric[1], rtol=1e-4)