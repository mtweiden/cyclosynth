from __future__ import annotations

from typing import Optional

from math import gcd

from cyclosynth.algebra import AlgebraicInteger


class GeneralIntegerRatio:
    """
    A ratio of two algebraic integers.
    """

    def __init__(
        self,
        numerator: AlgebraicInteger,
        denominator: Optional[AlgebraicInteger] = None,
    ) -> None:
        """
        Construct a ratio of algebraic integers.

        Args:
            numerator (AlgebraicInteger): The numerator of the ratio.

            denominator (Optional[AlgebraicInteger]): The denominator of the
                ratio. If None, then the ratio is a "whole" number.
                (Default: None)
        """
        self.numerator = numerator.copy()
        self.denominator = None if denominator is None else denominator.copy()

    def __mul__(self, number: int | AlgebraicInteger) -> GeneralIntegerRatio:
        """
        Multiply the ratio by a Rational or Algebraic integer.
        """
        # Convert number to an algebraic integer is needed
        number = self.numerator.int_to_algebraic_int(number)
        new_numerator = self.numerator * number
        new_ratio = GeneralIntegerRatio(new_numerator, self.denominator)
        new_ratio.simplify()
        return new_ratio

    def __truediv__(self, number: AlgebraicInteger) -> GeneralIntegerRatio:
        """
        Divide the ratio by a Rational or Algebraic integer.
        """
        # Convert number to an algebraic integer is needed
        # Assign to denominator if working with whole number
        if self.denominator is None:
            number = self.numerator.int_to_algebraic_int(number)
            new_denominator = number
        else:
            number = self.denominator.int_to_algebraic_int(number)
            new_denominator = self.denominator * number
        new_ratio = GeneralIntegerRatio(self.numerator, new_denominator)
        new_ratio.simplify()
        return new_ratio

    def to_float(self) -> complex:
        n = self.numerator.to_float()
        if self.denominator is None:
            d = 1
        else:
            d = self.denominator.to_float()
        return n / d

    def simplify(self) -> None:
        """
        Divide out any common scalar factors from numerator and denominator.
        """
        if self.denominator is None:
            return

        n_factor = self.numerator.greatest_divisor()
        d_factor = self.denominator.greatest_divisor()
        gcf = gcd(n_factor, d_factor)

        self.numerator.values = [n // gcf for n in self.numerator.values]
        self.denominator.values = [d // gcf for d in self.denominator.values]

    def __repr__(self) -> str:
        n = self.numerator.__repr__()
        if self.denominator is not None:
            d = self.denominator.__repr__()
            return f'{n} / {d}'
        else:
            return f'{n}'