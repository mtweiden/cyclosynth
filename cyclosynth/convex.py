"""A module defining convex sets in the real plane."""
from __future__ import annotations

from typing import Sequence

from cyclosynth.ellipse import Ellipse
from cyclosynth.ellipse import GridOperator
from cyclosynth.ratio import AlgebraicIntegerOverRoot2


class ConvexSet:

    def __init__(self, angle: float | None, epsilon: float | None) -> None:
        self.angle = angle
        self.epsilon = epsilon
        if angle is not None and epsilon is not None:
            self.ell = Ellipse.find_ellipse(angle, epsilon)
        else:
            self.ell = None
    
    @staticmethod
    def from_ellipse(ell: Ellipse) -> ConvexSet:
        convex = ConvexSet(None, None)
        convex.ell = ell
        return convex

    def ellipse(self) -> Ellipse:
        return self.ell
    
    def make_upright(self) -> tuple[ConvexSet, GridOperator]:
        ell, operator = self.ell.make_upright(return_operator=True)
        self.ell = ell
        return self, operator
    
    def apply_operator(self, operator: GridOperator) -> ConvexSet:
        self.ell = self.ell.apply_operator(operator)
        return self
    
    def bounding_box(self) -> tuple[tuple[float]]:
        """
        TODO: Do I care about the bounding box or the uprighted bounding box?
        """
        return self.ell.bounding_box()

    def check_inclusion_epsilon_region(self, p: Sequence[float]) -> bool:
        ...
    
    def check_inclusion_ellipse(self, p: Sequence[float]) -> bool:
        return self.ell.check_inclusion(p)
    
    def check_inclusion_bounding_box(self, p: Sequence[float]) -> bool:
        (x_lo, x_hi), (y_lo, y_hi) = self.bounding_box()
        x, y = p
        return x_lo <= x <= x_hi and y_lo <= y <= y_hi
    
    def line_intersection(
        self,
        slope: AlgebraicIntegerOverRoot2 | float,
        intercept: AlgebraicIntegerOverRoot2 | float,
    ) -> tuple[tuple[float]] | None:
        """
        Return the range of (x, y) values where the line intersects the set.

        The method computes the intersection of the line with the bounding box
        of the ellipse.

        TODO:
          - What format should the slope and intercept be in?
          - Should the return value be 2D?
          - Do I care abound the bounding box or the uprighted bounding box?
        """
        (x_lo, x_hi), (y_lo, y_hi) = self.bounding_box()

        def compute_y_on_line(x: AlgebraicIntegerOverRoot2 | float) -> float:
            if isinstance(x, AlgebraicIntegerOverRoot2):
                x = x.to_float()
            return slope * x + intercept 
        
        def compute_x_on_line(y: AlgebraicIntegerOverRoot2 | float) -> float:
            if isinstance(y, AlgebraicIntegerOverRoot2):
                y = y.to_float()
            return (y - intercept) / slope

        def in_range_vertical(y: float) -> bool:
            return y_lo <= y <= y_hi
        
        def in_range_horizontal(x: float) -> bool:
            return x_lo <= x <= x_hi
        
        y0, y1 = compute_y_on_line(x_lo), compute_y_on_line(x_hi)
        x0, x1 = compute_x_on_line(y_lo), compute_x_on_line(y_hi)
        iny0, iny1 = in_range_vertical(y0), in_range_vertical(y1)
        inx0, inx1 = in_range_horizontal(x0), in_range_horizontal(x1)

        if not any([iny0, iny1, inx0, inx1]):
            return None
        
        x_lo_, x_hi_ = max(x_lo, x0), min(x_hi, x1)
        y_lo_, y_hi_ = max(y_lo, y0), min(y_hi, y1)
        return ((x_lo_, x_hi_), (y_lo_, y_hi_))