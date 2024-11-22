"""
A module for creating and manipulating ellipses.

TODO: Change to symbolic computations.
"""
from __future__ import annotations

from typing import Sequence

from numpy import isclose
from mpmath import sqrt
from mpmath import pi
from mpmath import matrix
from mpmath import floor
from mpmath import ceil
from mpmath import cos
from mpmath import sin
from mpmath import power
from mpmath import diag
from mpmath import inverse


class Ellipse:
    """
    An ellipse in the real plane.
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
        self.center = (0.0, 0.0) if center is None else tuple(center)
    
    def copy(self) -> Ellipse:
        return Ellipse([self.a, self.b, self.d], self.center)
    
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
        return Ellipse([aa, bb, dd], self.center)
    
    def check_inclusion(self, point: Sequence[float]) -> bool:
        """See if point is in the ellipse."""
        x, y = point[0] - self.center[0], point[1] - self.center[1]
        a, b, d = self.a, self.b, self.d
        dist = a * x * x + 2 * b * x * y + d * y * y
        return dist <= 1
    
    def make_upright(
        self,
        return_operator: bool = False,
    ) -> Ellipse | tuple[Ellipse, Sequence[GridOperator]]:
        """
        Apply special grid operators until the ellipse is at least 1/2-upright.

        Args:
            return_operators (bool): If True, return the sequence of operators
                that were applied to the ellipse.
        
        Returns:
            upright_ellipse (Ellipse): The upright ellipse.

            operators (Sequence[GridOperator]): The sequence of operators that
                were applied to the ellipse.
        """

        def get_op(n: int, operator_a: bool) -> GridOperator:
            if operator_a:
                return GridOperator([1, n, 0, 1])
            else:
                return GridOperator([1, 0, n, 1])
        
        operator = GridOperator([1, 0, 0, 1])
            
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
                op = get_op(n, operator_a)
                new_up = ellipse.apply_operator(op).uprightness()
                return new_up - ellipse.uprightness()
            
            if measure(n) > 0:
                op = get_op(n, operator_a)
            else:
                op = get_op(-n, operator_a)
            ellipse = ellipse.apply_operator(op)
            operator = op.compose(operator)
        
        if return_operator:
            return ellipse, operator
        return ellipse
    
    def bounding_box(self) -> tuple[tuple[float]]:
        """Return smallest bounding box of the ellipse."""
        x, y = self.center
        sqrt_det = sqrt(self.det())
        w = sqrt(self.d) / sqrt_det
        h = sqrt(self.a) / sqrt_det
        return ((x - w, x + w), (y - h, y + h))
    
    def __eq__(self, other: Ellipse) -> bool:
        return self.mat == other.mat and self.center == other.center
    
    def is_close(self, other: Ellipse, precision: float = 1e-3) -> bool:
        da = self.a - other.a
        db = self.b - other.b
        dd = self.d - other.d
        dp0 = self.center[0] - other.center[0]
        dp1 = self.center[1] - other.center[1]
        return (
            isclose(da, 0,  atol=precision) and
            isclose(db, 0,  atol=precision) and
            isclose(dd, 0,  atol=precision) and
            isclose(dp0, 0, atol=precision) and
            isclose(dp1, 0, atol=precision)
        )
    
    def __repr__(self) -> str:
        if self.center != (0.0, 0.0):
            return f"Ellipse({self.mat}, {self.center})"
        else:
            return f"Ellipse({self.mat})"

    @staticmethod
    def find_ellipse(angle: float, epsilon: float) -> Ellipse:
        """
        Find the ellipse matrix for the given angle and precision.

        Returns:
            (Ellipse): An ellipse centered at and rotated along the epsilon
                region of the given angle.
        """
        d = 1 - (epsilon ** 2 / 2)  # distance from origin of e-region center
        zx = cos(-angle / 2)  # width of e-region
        zy = sin(-angle / 2)  # height of e-region
        center = (d * zx, d * zy)

        # Eigenvalues for scaling the determining matrix
        r1 = 4 / power(epsilon, 4)
        r2 = 1 / (power(epsilon, 2) * (1 - power(epsilon, 2) / 4))
        
        # Construct the determining matrix
        bmat = matrix([[zx, -zy], [zy, zx]])
        mmat = diag([r1, r2])
        mat = bmat @ mmat @ inverse(bmat)

        return Ellipse(mat, center)


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
        Right matrix multiplication.
        """
        a = self.a * other.a + self.b * other.c
        b = self.a * other.b + self.b * other.d
        c = self.c * other.a + self.d * other.c
        d = self.c * other.b + self.d * other.d
        return GridOperator([a, b, c, d])
    
    def __repr__(self) -> str:
        return f"GridOperator({self.a}, {self.b}, {self.c}, {self.d})"
    
    def apply_to_point(self, point: Sequence[float]) -> tuple[float]:
        x, y = point
        return (self.a * x + self.b * y, self.c * x + self.d * y)
