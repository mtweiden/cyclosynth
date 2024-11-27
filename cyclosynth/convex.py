"""A module defining convex sets in the real plane."""
from __future__ import annotations

from mpmath import mpf
from mpmath import sqrt
from mpmath import cos
from mpmath import sin

from cyclosynth.algebra import AlgebraicInteger
from cyclosynth.ratio import IntegerRatio
from cyclosynth.ellipse import Ellipse
from cyclosynth.operator import Operator
from cyclosynth.matrix import Vector
from cyclosynth.reduction import reduce


class ConvexSet:

    def __init__(
        self,
        enclosing_ellipse: Ellipse,
        angle: float | mpf | None = None,
        precision: float | mpf | None = None,
    ) -> None:
        self.ell = enclosing_ellipse
        self.angle = angle
        self.precision = precision
    
    @staticmethod
    def from_angle(angle: float | mpf, precision: float | mpf) -> ConvexSet:
        ell = Ellipse.find_ellipse(angle, precision)
        return ConvexSet(ell, angle, precision)
    
    @staticmethod
    def from_bounding_box(
        x_range: tuple[float | mpf | IntegerRatio | AlgebraicInteger],
        y_range: tuple[float | mpf | IntegerRatio | AlgebraicInteger],
    ) -> ConvexSet:
        x_lo, x_hi = x_range
        y_lo, y_hi = y_range

        if isinstance(x_lo, IntegerRatio):
            x_lo = x_lo.to_float()
        if isinstance(x_hi, IntegerRatio):
            x_hi = x_hi.to_float()
        if isinstance(y_lo, IntegerRatio):
            y_lo = y_lo.to_float()
        if isinstance(y_hi, IntegerRatio):
            y_hi = y_hi.to_float()

        w, h = x_hi - x_lo, y_hi - y_lo
        center = ((x_lo + x_hi) / 2, (y_lo + y_hi) / 2)
        return ConvexSet(Ellipse([2 / w ** 2, 0, 2 / h ** 2], center))
    
    def transform(self, operator: Operator) -> ConvexSet:
        """
        Return a new set corresponding to a transform on this set's ellipse.
        """
        return ConvexSet(self.ell.copy().apply_operator(operator))

    def ellipse(self) -> Ellipse:
        return self.ell
    
    def make_upright(self) -> Operator:
        return reduce(self.ell)
    
    def apply_operator(self, operator: Operator) -> ConvexSet:
        self.ell = self.ell.apply_operator(operator)
        return self
    
    def bounding_box(self) -> tuple[tuple[float]]:
        return self.ell.bounding_box()
    
    def check_inclusion(
        self,
        alpha: IntegerRatio | float,
        beta: IntegerRatio | float,
    ) -> bool:
        if self.angle is None or self.precision is None:
            return self.check_inclusion_ellipse(alpha, beta)
        else:
            return self.check_inclusion_epsilon_region(alpha, beta)

    def check_inclusion_epsilon_region(
        self,
        alpha: IntegerRatio | float,
        beta: IntegerRatio | float,
    ) -> bool:
        if self.angle is None or self.precision is None:
            raise ValueError('Angle and precision must be set.')
        if isinstance(alpha, IntegerRatio):
            alpha = alpha.to_float()
        if isinstance(beta, IntegerRatio):
            beta = beta.to_float()
        # TODO: Change this to Hilbert-Schmidt distance?
        x, y = cos(self.angle), sin(self.angle)
        eucliean_dist_2 = (alpha - x) ** 2 + (beta - y) ** 2
        return eucliean_dist_2 <= self.precision ** 2
    
    def check_inclusion_ellipse(
        self,
        alpha: IntegerRatio | float,
        beta: IntegerRatio | float,
    ) -> bool:
        if isinstance(alpha, IntegerRatio):
            alpha = alpha.to_float()
        if isinstance(beta, IntegerRatio):
            beta = beta.to_float()
        return self.ell.check_inclusion((alpha, beta))
    
    def check_inclusion_bounding_box(
        self,
        alpha: IntegerRatio | float,
        beta: IntegerRatio | float,
    ) -> bool:
        if isinstance(alpha, IntegerRatio):
            alpha = alpha.to_float()
        if isinstance(beta, IntegerRatio):
            beta = beta.to_float()
        (x_lo, x_hi), (y_lo, y_hi) = self.bounding_box()
        return x_lo <= alpha <= x_hi and y_lo <= beta <= y_hi
    
    def line_intersection_ellipse(
        self,
        slope: Vector,
        intercept: Vector,
    ) -> tuple[float] | None:
        """
        Return the range of values (t0, t1) where the line intersects the
        ellipse of this ConvexSet.

        Given p + tv, where p, v are vectors, find the range of t where the
        line intersects the set.
        """
        v, p = slope, intercept

        # Radius
        # TODO: What if the radius is not 1?
        # What if the ellipse is not a circle?
        s = 1

        # Compute quadratic coefficients
        a = v.innerprod(v)
        b = 2 * v.innerprod(p)
        c = p.innerprod(p) - s

        # Solve the quadratic equation
        discriminant = b * b - 4 * a * c
        if discriminant < 0:
            return None
        sqrt_discriminant = sqrt(discriminant.to_float())
        x1 = (-b - sqrt_discriminant) / (2 * a)
        x2 = (-b + sqrt_discriminant) / (2 * a)
        roots = (min(x1, x2), max(x1, x2))

        # If no real roots, there is no intersection
        if roots is None:
            return None

        t0, t1 = roots
        return (t0, t1)

    def intersection(
        self,
        direction: Vector,
        point: Vector,
    ) -> tuple[float] | None:
        """
        Return the range of values (t0, t1) where the line intersects the
        bounding box of this ConvexSet.

        Given p + tv, where p, v are vectors, find the range of t where the
        line intersects the set.
        """
        def _convert_value(value: float | mpf | IntegerRatio | int) -> float:
            if isinstance(value, IntegerRatio):
                return value.to_float()
            return value

        (px, py), (vx, vy) = point.values, direction.values

        px, py = _convert_value(px), _convert_value(py)
        vx, vy = _convert_value(vx), _convert_value(vy)

        (x0, x1), (y0, y1) = self.bounding_box()

        # Line is vertical and intersects the rectangle
        if vx == 0 and x0 <= px <= x1:
            t0y = (y0 - py) / vy if vy != 0 else float('-inf')
            t1y = (y1 - py) / vy if vy != 0 else float('inf')
            return (min(t0y, t1y), max(t0y, t1y))
        # Line is vertical but outside the rectangle
        elif vx == 0:
            return None
        # Line is horizontal and intersects the rectangle
        elif vy == 0 and y0 <= py <= y1:
            t0x = (x0 - px) / vx if vx != 0 else float('-inf')
            t1x = (x1 - px) / vx if vx != 0 else float('inf')
            return (min(t0x, t1x), max(t0x, t1x))
        # Line is horizontal but outside the rectangle
        elif vy == 0:
            return None
        # General case: compute the intersection
        t0x = (x0 - px) / vx
        t1x = (x1 - px) / vx
        t0y = (y0 - py) / vy
        t1y = (y1 - py) / vy
        t0 = max(min(t0x, t1x), min(t0y, t1y))
        t1 = min(max(t0x, t1x), max(t0y, t1y))

        # Check if the intervals overlap
        if t0 > t1:
            return None  # No intersection
        return (t0, t1)
