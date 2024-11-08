"""A module for creating and manipulating ellipses."""
from __future__ import annotations

from typing import Sequence

from mpmath import sqrt
from mpmath import pi
from mpmath import matrix
from mpmath import floor
from mpmath import ceil

from cyclosynth.algebra import AlgebraicInteger


class Ellipse:
    """
    An ellipse in the real plane.

    TODO: Are these actually integers?
    """
    def __init__(
        self,
        mat: Sequence[float] | matrix,
        center: Sequence[float] = None,
    ) -> None:
        """
        Initialize an ellipse with the given parameters.

        E = {u in R^2 | (u - p).dagger D (u - p) <= 1}

        Args:
            d_parameters (Sequence[float] | ndarray): A length 3 sequence of
                parameters for the defining matrix D.
            
            p_parameters (Sequence[float]): A length 2 sequence of parameters
                specifying the center of the ellipse.
        """
        if isinstance(mat, matrix):
            self.a, self.b, self.d = mat[0,0], mat[0,1], mat[1,1]
        else:
            self.a, self.b, self.d = mat
        self.p = tuple(center) or (0.0, 0.0)
    
    @property
    def center(self) -> tuple[float]:
        return self.p
    
    @property
    def mat(self) -> tuple[tuple[float]]:
        return ((self.a, self.b), (self.b, self.d))
    
    def det(self) -> float:
        d = self.a * self.d - self.b * self.b
        return d
    
    def area(self) -> float:
        return pi / sqrt(self.det())
    
    def area_of_bounding_box(self) -> float:
        return 4 * sqrt(self.a * self.d) / self.det()
    
    def uprightness(self) -> float:
        return self.area() / self.area_of_bounding_box()
    
    def skew(self) -> float:
        return self.b * self.b
    
    def apply_operator(self, operator: GridOperator) -> Ellipse:
        inv_op = operator.inverse()
        w, x, y, z = inv_op.a, inv_op.b, inv_op.c, inv_op.d
        a, b, d = self.a, self.b, self.d
        aa = w * (a * w + b * y) + y * (b * w + d * y)
        bb = w * (a * x + b * z) + y * (b * x + d * z)
        dd = x * (a * x + b * z) + z * (b * x + d * z)
        return Ellipse([aa, bb, dd], self.p)
    
    def check_inclusion(self, point: Sequence[float]) -> bool:
        """See if point is in the ellipse."""
        x, y = point[0] - self.p[0], point[1] - self.p[1]
        a, b, d = self.a, self.b, self.d
        return a * x * x + 2 * b * x * y + d * y * y <= 1
    
    def make_upright(self) -> Ellipse:
        """
        Apply special grid operators until the ellipse is at least 1/2-upright.

        TODO:
          - Write a function that returns the grid operator tha does this.
        """
        def apply_op(ellipse, n, operator_a):
            if operator_a:
                op = GridOperator([1, n, 0, 1])
            else:
                op = GridOperator([1, 0, n, 1])
            return ellipse.apply_operator(op)
            
        ellipse = self
        while ellipse.uprightness() < 1/2:
            # If a<= d - A.dagger^n D A^n: b -> na + b
            if ellipse.a <= ellipse.d:
                operator_a = True
                rhs = (ellipse.a - 2 * ellipse.b) / (2 * ellipse.a)
                if rhs < 0:
                    n = -int(floor(rhs))
                else:
                    n = int(ceil(rhs))
            # If d < a - B.dagger^n D B^n: b -> nd + b
            else:
                operator_a = False
                rhs = (ellipse.d - 2 * ellipse.b) / (2 * ellipse.d)
                if rhs < 0:
                    n = -int(floor(rhs))
                else:
                    n = int(ceil(rhs))

            def measure(n):
                new_up = apply_op(ellipse, n, operator_a).uprightness()
                return new_up - ellipse.uprightness()
            
            if measure(n) > 0:
                ellipse = apply_op(ellipse, n, operator_a)
            else:
                ellipse = apply_op(ellipse, -n, operator_a)
        return ellipse


class GridOperator:
    """
    An integer matrix that operates on the integer lattice.
    """
    def __init__(self, values: Sequence[int]) -> None:
        self.a, self.b, self.c, self.d = values
    
    def det(self) -> int:
        return self.a * self.d - self.b * self.c
    
    def inverse(self) -> GridOperator | None:
        if abs(self.det()) != 1:
            return None
        det = self.det()
        a = self.d / det
        b = -self.b / det
        c = -self.c / det
        d = self.a / det
        return GridOperator([a, b, c, d])
    
    def compose(self, other: GridOperator) -> GridOperator:
        """
        TODO: Check if multiplying on correct side.
        """
        a = self.a * other.a + self.b * other.c
        b = self.a * other.b + self.b * other.d
        c = self.c * other.a + self.d * other.c
        d = self.c * other.b + self.d * other.d
        return GridOperator([a, b, c, d])