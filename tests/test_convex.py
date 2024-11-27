from random import random
from random import randint

from mpmath import pi

from numpy import isclose

from cyclosynth.algebra import RingRoot2
from cyclosynth.ratio import IntegerRatio
from cyclosynth.convex import ConvexSet
from cyclosynth.ellipse import Ellipse
from cyclosynth.matrix import Vector


from random import seed
seed(42)


def random_ratios(n: int) -> list[IntegerRatio]:
    ratios = []
    d = randint(-1000000, 1000000)
    for _ in range(n):
        a, b = [randint(-1000000, 1000000) for _ in range(2)]
        _a, _b = RingRoot2([a, d]), RingRoot2([b, d])
        ratios.append(IntegerRatio(_a, _b))
    return ratios


class TestConvex:

    num_trials = 1000

    def test_intersection(self) -> None:
        one = RingRoot2([1, 0])
        zero = RingRoot2([0, 0])

        def assert_intersection(sec: tuple[float], expected: bool) -> None:
            if sec is None:
                assert not expected
            else:
                assert expected

        for _ in range(self.num_trials):
            x_lo, x_hi, y_lo, y_hi = random_ratios(4)
            if x_lo.to_float() > x_hi.to_float():
                x_lo, x_hi = x_hi, x_lo
            if y_lo.to_float() > y_hi.to_float():
                y_lo, y_hi = y_hi, y_lo

            convex = ConvexSet.from_bounding_box((x_lo, x_hi), (y_lo, y_hi))

            hor_length = x_hi - x_lo
            ver_length = y_hi - y_lo

            right = Vector((one, zero))
            up = Vector((zero, one))
            angled = Vector((hor_length, ver_length))

            # p = (p0, p1) is a point on the plane
            # v = (v0, v1) is a vector pointing in the direction of the line
            center = Vector(((x_hi + x_lo) / 2, (y_hi + y_lo) / 2))
            outside = center + Vector((hor_length, ver_length))

            intersection = convex.intersection(right, center)
            assert_intersection(intersection, True)
            intersection = convex.intersection(up, center)
            assert_intersection(intersection, True)
            intersection = convex.intersection(angled, center)
            assert_intersection(intersection, True)

            intersection = convex.intersection(right, outside)
            assert_intersection(intersection, False)
            intersection = convex.intersection(up, outside)
            assert_intersection(intersection, False)
            intersection = convex.intersection(angled, outside)
            assert_intersection(intersection, True)