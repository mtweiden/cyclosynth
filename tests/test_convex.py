from random import random

from mpmath import pi

from cyclosynth.convex import ConvexSet


from random import seed
seed(42)


class TestConvex:

    num_trials = 1000

    def test_line_intersection(self) -> None:
        for _ in range(self.num_trials):
            angle = 2 * pi * random()
            precision = 1e-3
            convex = ConvexSet(angle, precision)

            (x_lo, x_hi), (y_lo, y_hi) = convex.bounding_box()

            m = (y_hi - y_lo) / (x_hi - x_lo)
            eps = 0.1 * abs(y_hi - y_lo)

            def assert_intersection(
                intersection: tuple[tuple[float]] | None,
                is_included: bool,
            ) -> None:
                if intersection is None:
                    assert not is_included
                else:
                    (x_lo_, x_hi_), (y_lo_, y_hi_) = intersection
                    assert x_lo <= x_lo_ and x_hi_ <= x_hi
                    assert y_lo <= y_lo_ and y_hi_ <= y_hi
            
            p0, p1 = convex.ellipse().center
            half_len = (y_hi - y_lo) / 2

            # Success cases
            # case 1: left to top
            slope = m
            offset = p1 - slope * p0
            intercept = offset + half_len - eps
            intersection = convex.line_intersection(slope, intercept)
            assert_intersection(intersection, True)
            # case 2: left to right
            slope = eps
            offset = p1 - slope * p0
            intercept = offset
            intersection = convex.line_intersection(slope, intercept)
            assert_intersection(intersection, True)
            # case 3: bottom to right
            slope = m
            offset = p1 - slope * p0
            intercept = offset - half_len - eps
            intersection = convex.line_intersection(slope, intercept)
            assert_intersection(intersection, True)
            # case 4: top to right
            slope = -m
            offset = p1 - slope * p0
            intercept = offset + half_len + eps
            intersection = convex.line_intersection(slope, intercept)
            assert_intersection(intersection, True)
            # case 5: left to right
            slope = -eps
            offset = p1 - slope * p0
            intercept = offset
            intersection = convex.line_intersection(slope, intercept)
            assert_intersection(intersection, True)
            # case 6: right to bottom
            slope = -m
            offset = p1 - slope * p0
            intercept = offset - half_len + eps
            intersection = convex.line_intersection(slope, intercept)
            assert_intersection(intersection, True)

            # Failure cases
            # case 7: above top
            slope = m
            offset = p1 - slope * p0
            intercept = offset + half_len + eps
            intersection = convex.line_intersection(slope, intercept)
            assert_intersection(intersection, False)
            # case 8: below bottom
            slope = -m
            offset = p1 - slope * p0
            intercept = offset - half_len - eps
            intersection = convex.line_intersection(slope, intercept)
            assert_intersection(intersection, False)