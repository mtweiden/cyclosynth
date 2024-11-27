from __future__ import annotations

from math import gcd
from math import log2

from mpmath import sqrt
from mpmath import mpf

from cyclosynth.algebra import AlgebraicInteger
from cyclosynth.algebra import DyadicComplexNumber
from cyclosynth.algebra import RingRoot2
from cyclosynth.algebra import RingRootRoot2Plus2


class IntegerRatio:
    """
    A ratio of two algebraic integers.

    The denominator is of the form `denominator` ** `denominator_power`.
    """

    def __init__(
        self,
        numerator: AlgebraicInteger | int,
        denominator: AlgebraicInteger | int = 1,
    ) -> None:
        """
        Construct a ratio of algebraic integers.

        Args:
            numerator (AlgebraicInteger): The numerator of the ratio.

            denominator (AlgebraicInteger): The denominator of the ratio.
        """
        if isinstance(numerator, AlgebraicInteger):
            self.numerator = numerator.copy()
        else:
            self.numerator = numerator  # int

        if isinstance(denominator, AlgebraicInteger):
            assert denominator.to_float() != 0, 'Denominator cannot be zero.'
            self.denominator = denominator.copy()
        else:
            assert denominator != 0, 'Denominator cannot be zero.'
            self.denominator = denominator
        
    def _combine_integers(
        self,
        d1: AlgebraicInteger | int,
        d2: AlgebraicInteger | int,
    ) -> AlgebraicInteger | int:
        if isinstance(d1, int) and not isinstance(d2, int):
            d3 = d2 * d1
        elif isinstance(d2, int) and not isinstance(d1, int):
            d3 = d1 * d2
        else:
            d3 = d1 * d2
        return d3

    def __mul__(
        self,
        other: IntegerRatio | AlgebraicInteger | int,
    ) -> IntegerRatio:
        """
        Multiply the ratio by an IntegerRatio.
        """
        if isinstance(other, int):
            new_numerator = self.numerator * other
            new_ratio = IntegerRatio(new_numerator, self.denominator)
        else:
            if isinstance(other, AlgebraicInteger):
                other = IntegerRatio(other)
            new_numerator = self._combine_integers(self.numerator, other.numerator)
            new_denominator = self._combine_integers(
                self.denominator, other.denominator,
            )
            new_ratio = IntegerRatio(new_numerator, new_denominator)
        new_ratio.simplify()
        return new_ratio

    def __add__(self, other: IntegerRatio | int) -> IntegerRatio:
        """
        Add to the ratio by an IntegerRatio.
        """
        a = self._combine_integers(other.numerator, self.denominator)
        b = self._combine_integers(self.numerator, other.denominator)
        if isinstance(a, int):
            new_numerator = b + a
        else:
            new_numerator = a + b
        new_denominator = self._combine_integers(
            self.denominator, other.denominator,
        )
        new_ratio = IntegerRatio(new_numerator, new_denominator)
        new_ratio.simplify()
        return new_ratio
    
    def __sub__(self, other: IntegerRatio) -> IntegerRatio:
        """
        Subtract an IntegerRatio from the ratio.
        """
        return self + (-other)
    
    def __truediv__(self, other: IntegerRatio) -> IntegerRatio:
        new_num = self.numerator * other.denominator
        new_denom = self.denominator * other.numerator
        new_ratio = IntegerRatio(new_num, new_denom)
        new_ratio.simplify()
        return new_ratio

    def simplify(self) -> None:
        """
        Divide by the denominator as many times as possible.
        """
        if isinstance(self.numerator, AlgebraicInteger):
            if all(v == 0 for v in self.numerator.values):
                self.denominator = 1
        if self.denominator == 1:
            return
        if isinstance(self.denominator, int):
            denom_vals = [self.denominator]
        else:
            denom_vals = self.denominator.values
        if isinstance(self.numerator, int):
            numer_vals = [self.numerator]
        else:
            numer_vals = self.numerator.values

        num_gcd = gcd(*numer_vals)
        den_gcd = gcd(*denom_vals)
        full_gcd = gcd(num_gcd, den_gcd)

        if isinstance(self.numerator, int):
            self.numerator = self.numerator // full_gcd
        else:
            self.numerator.values = [
                v // full_gcd for v in self.numerator.values
            ]
        if isinstance(self.denominator, int):
            self.denominator = self.denominator // full_gcd
        else:
            self.denominator.values = [
                v // full_gcd for v in self.denominator.values
            ]
    
    def conj(self) -> IntegerRatio:
        """
        Return the conjugate of the ratio.
        """
        if not isinstance(self.numerator, int):
            new_numerator = self.numerator.conj()
        else:
            new_numerator = self.numerator
        if not isinstance(self.denominator, int):
            new_denominator = self.denominator.conj()
        else:
            new_denominator = self.denominator
        return IntegerRatio(new_numerator, new_denominator)
    
    def inverse(self) -> IntegerRatio:
        """
        Return the inverse of the ratio.
        """
        return IntegerRatio(self.denominator, self.numerator)

    def to_float(self) -> complex:
        if isinstance(self.numerator, AlgebraicInteger):
            n = self.numerator.to_float()
        else:
            n = self.numerator
        if isinstance(self.denominator, AlgebraicInteger):
            d = self.denominator.to_float()
        else:
            d = self.denominator
        return n / d

    def __repr__(self) -> str:
        n = self.numerator.__repr__()
        d = self.denominator.__repr__()
        return f'({n}) / ({d})'
    
    def __neg__(self) -> IntegerRatio:
        if isinstance(self.numerator, AlgebraicInteger):
            new_numerator = self.numerator.copy()
        else:
            new_numerator = self.numerator
        new_numerator = new_numerator * -1
        return IntegerRatio(new_numerator, self.denominator)


class AlgebraicIntegerOverRootRoot2Plus2(IntegerRatio):
    """
    Some AlgebraicInteger over a power of sqrt(sqrt(2)+2).
    """
    def __init__(
        self,
        integer: AlgebraicInteger,
        power: int = 0,
    ) -> None:
        """
        Construct an AlgebraicInteger over a power of sqrt(sqrt(2)+2).

        Args:
            integer (AlgebraicInteger): The algebraic integer.

            power (int): The power of sqrt(sqrt(2)+2) dividing `integer`.
                (Default: 0)
        
        Raises:
            ValueError: If `power` is negative.
        """
        rr2p2 = RingRootRoot2Plus2([0, 0, 1, 0])
        super().__init__(integer, rr2p2)
        self.denominator_power = power

    def __add__(self, other: IntegerRatio) -> IntegerRatio:
        # Set denominators equal
        if isinstance(other, AlgebraicIntegerOverRoot2):
            other = other.to_rr2p2()
        rr2p2 = RingRootRoot2Plus2([0, 0, 1, 0])
        if self.denominator_power < other.denominator_power:
            for _ in range(other.denominator_power - self.denominator_power):
                self.numerator = self.numerator * rr2p2
            self.denominator_power = other.denominator_power
        elif self.denominator_power > other.denominator_power:
            for _ in range(self.denominator_power - other.denominator_power):
                other.numerator = other.numerator * rr2p2
            other.denominator_power = self.denominator_power
        # Perform addition
        new_integer = self.numerator + other.numerator
        # Simplify if possible
        new_ratio = AlgebraicIntegerOverRootRoot2Plus2(
            new_integer, self.denominator_power,
        )
        new_ratio.simplify()
        return new_ratio
    
    def __mul__(self, other: IntegerRatio) -> IntegerRatio:
        if isinstance(other, AlgebraicIntegerOverRoot2):
            other = other.to_rr2p2()
        # Perform multiplication
        new_integer = self.numerator * other.numerator
        new_power = self.denominator_power + other.denominator_power
        # Simplify if possible
        new_ratio = AlgebraicIntegerOverRootRoot2Plus2(new_integer, new_power)
        new_ratio.simplify()
        return new_ratio
    
    def simplify(self) -> None:
        """
        Simplify the ratio by dividing by powers of the denominator.

        Division algorithm based on `utils.is_divisible_by_rootroot2plus2`.
        """
        if isinstance(self.numerator, AlgebraicInteger):
            if all(v == 0 for v in self.numerator.values):
                self.denominator_power = 0
        gamma = RingRootRoot2Plus2([0, 0, 2, -1])
        result = self.numerator.copy()
        for _ in range(self.denominator_power):
            new_result = result * gamma
            if all(v % 2 == 0 for v in new_result.values):
                result.values = [v // 2 for v in new_result.values]
                self.denominator_power -= 1
            else:
                break
        self.numerator = result
    
    def to_float(self) -> float:
        numerator = self.numerator.to_float().real
        denominator = sqrt(2 + sqrt(2)) ** self.denominator_power
        return numerator / denominator
    
    def __repr__(self) -> str:
        numerator = self.numerator.__repr__()
        rr2p2 = 'sqrt(2 + sqrt(2))'
        denominator = rr2p2 if self.denominator_power == 1 else \
                f'{rr2p2}^{self.denominator_power}'
        return f'{numerator} / {denominator}'
    
    def __neg__(self) -> AlgebraicIntegerOverRootRoot2Plus2:
        new_numerator = self.numerator.copy()
        new_numerator = new_numerator * -1
        return AlgebraicIntegerOverRootRoot2Plus2(
            new_numerator, self.denominator_power,
        )
    
    @staticmethod
    def from_dyadic(
        dyadic: DyadicComplexNumber,
    ) -> AlgebraicIntegerOverRootRoot2Plus2:
        """
        Convert a number in Z[e^{i*pi/n}, 1/2] to an IntegerRatio.

        Args:
            dyadic (DyadicComplexNumber): The DyadicComplexNumber to convert
                to an AlgebraicIntegerOverRoot2.
        """
        assert len(dyadic.values) == 16
        dyadic.simplify()
        k = dyadic.denominator_exponent
        one_plus_root2 = RingRootRoot2Plus2([1, 1, 0, 0])
        scale = RingRootRoot2Plus2([1, 0, 0, 0])
        for _ in range(2 * k):
            scale = scale * one_plus_root2
        scale = AlgebraicIntegerOverRootRoot2Plus2(scale, 4 * k)
        c0 = 2 * dyadic.values[2]
        c1 = dyadic.values[2] + dyadic.values[6]
        c2 = dyadic.values[0]
        c3 = dyadic.values[4]
        dyadic_as_alg_int = RingRootRoot2Plus2([c0, c1, c2, c3])
        number = AlgebraicIntegerOverRootRoot2Plus2(dyadic_as_alg_int, 1)
        number = scale * number
        return number


class AlgebraicIntegerOverRoot2(IntegerRatio):
    """
    Some AlgebraicInteger over a power of sqrt(2).
    """
    def __init__(
        self,
        integer: AlgebraicInteger,
        power: int = 0,
    ) -> None:
        """
        Construct an AlgebraicInteger over a power of sqrt(2).

        Args:
            integer (AlgebraicInteger): The algebraic integer.

            power (int): The power of sqrt(2) dividing `integer`.
                (Default: 0)
        
        Raises:
            ValueError: If `power` is negative.
        """
        r2 = RingRoot2([0, 1])
        super().__init__(integer, r2)
        self.denominator_power = power 

    def __add__(self, other: IntegerRatio) -> IntegerRatio:
        if isinstance(other, AlgebraicIntegerOverRootRoot2Plus2):
            new_self = self.to_rr2p2()
            return new_self + other
        # Set denominators equal
        r2 = RingRoot2([0, 1])
        if self.denominator_power < other.denominator_power:
            for _ in range(other.denominator_power - self.denominator_power):
                self.numerator = self.numerator * r2
            self.denominator_power = other.denominator_power
        elif self.denominator_power > other.denominator_power:
            for _ in range(self.denominator_power - other.denominator_power):
                other.numerator = other.numerator * r2
            other.denominator_power = self.denominator_power
        # Perform addition
        new_integer = self.numerator + other.numerator
        # Simplify if possible
        new_ratio = AlgebraicIntegerOverRoot2(
            new_integer, self.denominator_power,
        )
        new_ratio.simplify()
        return new_ratio
    
    def __mul__(self, other: IntegerRatio | float) -> IntegerRatio | float:
        if isinstance(other, AlgebraicIntegerOverRootRoot2Plus2):
            new_self = self.to_rr2p2()
            return new_self * other
        if isinstance(other, float) or isinstance(other, mpf):
            return self.to_float() * other
        # Perform multiplication
        new_integer = self.numerator * other.numerator
        new_power = self.denominator_power + other.denominator_power
        # Simplify if possible
        new_ratio = AlgebraicIntegerOverRoot2(new_integer, new_power)
        new_ratio.simplify()
        return new_ratio
    
    def __neg__(self) -> AlgebraicIntegerOverRoot2:
        new_numerator = self.numerator.copy()
        new_numerator = new_numerator * -1
        return AlgebraicIntegerOverRoot2(new_numerator, self.denominator_power)
    
    def simplify(self) -> None:
        """
        Simplify the ratio by dividing by powers of the denominator.

        Division algorithm based on `utils.is_divisible_by_root2`.
        """
        if isinstance(self.numerator, AlgebraicInteger):
            if all(v == 0 for v in self.numerator.values):
                self.denominator_power = 0
        if isinstance(self.numerator, int):
            result = RingRoot2([self.numerator, 0])
        else:
            result = self.numerator.copy()
        gamma = RingRoot2([0, 1])
        for _ in range(self.denominator_power):
            new_result = gamma * result
            if all(v % 2 == 0 for v in new_result.values):
                result.values = [v // 2 for v in new_result.values]
                self.denominator_power -= 1
            else:
                break
        self.numerator = result
    
    def __repr__(self) -> str:
        numerator = self.numerator.__repr__()
        r2 = 'sqrt(2)'
        if self.denominator_power == 0:
            return numerator
        elif self.denominator_power == 1:
            denominator = r2
        else:
            denominator = f'{r2}^{self.denominator_power}'
        return f'{numerator} / {denominator}'
    
    def to_float(self) -> float:
        numerator = self.numerator.to_float().real
        denominator = sqrt(2) ** self.denominator_power
        return numerator / denominator
    
    def to_rr2p2(self) -> AlgebraicIntegerOverRootRoot2Plus2:
        """
        Convert a ratio over sqrt(2) to a ratio over sqrt(sqrt(2)+2).

        1 / sqrt(2) = (1 + sqrt(2)) / (sqrt(2) + 2) ** 2

        Returns:
            (AlgebraicIntegerOverRootRoot2Plus2): An equivalent ratio over
                some power of sqrt(sqrt(2)+2).
        """
        if self.denominator_power == 0:
            return AlgebraicIntegerOverRootRoot2Plus2(self.numerator, 0)
        conversion_numerator = RingRootRoot2Plus2([1, 1, 0, 0])
        factor = conversion_numerator.copy()
        for _ in range(self.denominator_power - 1):
            conversion_numerator = conversion_numerator * factor
        new_numerator = conversion_numerator * self.numerator
        new_power = self.denominator_power * 2
        ratio = AlgebraicIntegerOverRootRoot2Plus2(new_numerator, new_power)
        ratio.simplify()
        return ratio
    
    @staticmethod
    def from_dyadic(dyadic: DyadicComplexNumber) -> AlgebraicIntegerOverRoot2:
        """
        Convert a number in Z[e^{i*pi/n}, 1/2] to an IntegerRatio.

        Args:
            dyadic (DyadicComplexNumber): The DyadicComplexNumber to convert
                to an AlgebraicIntegerOverRoot2.
        """
        assert len(dyadic.values) == 8
        dyadic.simplify()
        k = dyadic.denominator_exponent
        c0 = dyadic.values[0]
        c1 = dyadic.values[2]
        dyadic_as_alg_int = RingRoot2([c0, c1])
        number = AlgebraicIntegerOverRoot2(dyadic_as_alg_int, 2 * k)
        return number
    
    def conj(self) -> AlgebraicIntegerOverRoot2:
        """
        Return the conjugate of the ratio.
        """
        new_numerator = self.numerator.conj()
        ratio = AlgebraicIntegerOverRoot2(new_numerator, self.denominator_power)
        if self.denominator_power % 2 == 1:
            ratio = -ratio
        return ratio
    
    @staticmethod
    def from_integer_ratio(ratio: IntegerRatio) -> AlgebraicIntegerOverRoot2:
        """
        Convert an IntegerRatio to an AlgebraicIntegerOverRoot2.

        Args:
            integer_ratio (IntegerRatio): The IntegerRatio to convert.
        """
        denom = ratio.denominator
        if isinstance(denom, int):
            denom = RingRoot2([denom, 0])
        if sum(1 for x in denom.values if x != 0) != 1 or not \
                any(power_of_2(v) for v in denom.values):
            m = 'Denominator must be a power of sqrt(2) and non zero.'
            raise ValueError(m)

        if denom.values[0] == 0:
            power = 2 * int(log2(denom.values[1])) + 1
        else:
            power = 2 * int(log2(denom.values[0]))
        
        new_ratio = AlgebraicIntegerOverRoot2(ratio.numerator, power)
        new_ratio.simplify()
        return new_ratio


def power_of_2(n: int) -> bool:
    if n == 0:  # Not actually a power of 2
        return True
    return log2(n) == int(log2(n))