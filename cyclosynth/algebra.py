from __future__ import annotations

from abc import ABC
from abc import abstractmethod

from typing import Sequence

from cmath import exp

from math import gcd
from math import log2
from math import sqrt
from math import pi


class AlgebraicInteger(ABC):
    """
    An abstract class representing algebraic integers.

    The methods for this class define general ring operations that are valid
    for all algebraic integers.
    """

    def __init__(self, values: int | Sequence[int]) -> None:
        """
        Construct an algebraic integer.

        Args:
            value (int | Sequence[int]): The value of the algebraic integer.
        """
        if isinstance(values, int):
            values = [values]
        self.values = list(values).copy()
    
    @abstractmethod
    def __add__(self, other: int | AlgebraicInteger) -> AlgebraicInteger:
        m = 'Define an __add__ method.'
        raise NotImplementedError(m)

    @abstractmethod
    def __sub__(self, other: int | AlgebraicInteger) -> AlgebraicInteger:
        m = 'Define a __sub__ method.'
        raise NotImplementedError(m)

    @abstractmethod
    def __mul__(self, other: int | AlgebraicInteger) -> AlgebraicInteger:
        m = 'Define a __mul__ method.'
        raise NotImplementedError(m)

    @abstractmethod
    def int_to_algebraic_int(
        self,
        number: int | AlgebraicInteger,
    ) -> AlgebraicInteger:
        m = 'Define how to convert integers to this type of Algebraic Integer.'
        raise NotImplementedError(m)

    @abstractmethod
    def to_float(self) -> float:
        m = 'Define how to convert this type of Algebraic Integer to a float.'
        raise NotImplementedError(m)

    def __float__(self) -> float:
        return float(self.to_float().real)

    def greatest_divisor(self) -> int:
        return gcd(*[abs(n) for n in self.values])
    
    def copy(self) -> AlgebraicInteger:
        return self.__class__(self.values)
    
    def conj(self) -> AlgebraicInteger:
        raise NotImplementedError('Define a conjugate method.')
    

class RingRoot2(AlgebraicInteger):

    def __init__(self, values: Sequence[int]) -> None:
        """
        An algebraic integer in Z[sqrt(2)]

        Args:
            values (Sequence[int]): A Sequence of length two in the form
                values[0] + values[1] * sqrt(2).

        Raises:
            ValueError: If `values` is not of length 2.
        """
        if len(values) != 2:
            raise ValueError('`values` must be of length 2.')
        super().__init__(values)

    def __add__(self, other: int |AlgebraicInteger) -> AlgebraicInteger:
        if isinstance(other, RingRootRoot2Plus2):
            return other + self
        elif isinstance(other, int):
            x, y = other, 0
        else:
            x, y = other.values
        a = self.values[0] + x
        b = self.values[1] + y
        new_integer = RingRoot2([a, b])
        return new_integer

    def __sub__(self, other: int | AlgebraicInteger) -> AlgebraicInteger:
        if isinstance(other, RingRootRoot2Plus2):
            return other - self
        elif isinstance(other, int):
            x, y = other, 0
        else:
            x, y = other.values
        a = self.values[0] - x
        b = self.values[1] - y
        new_integer = RingRoot2([a, b])
        return new_integer

    def __mul__(self, other: int | AlgebraicInteger) -> AlgebraicInteger:
        if isinstance(other, RingRootRoot2Plus2):
            return other * self
        elif isinstance(other, int):
            x, y = other, 0
        else:
            x, y = other.values
        a, b = self.values
        new_a = a * x + 2 * b * y
        new_b = a * y + b * x
        new_integer = RingRoot2([new_a, new_b])
        return new_integer

    def int_to_algebraic_int(
        self,
        number: int | AlgebraicInteger,
    ) -> AlgebraicInteger:
        if isinstance(number, AlgebraicInteger):
            return number
        assert isinstance(number, int), '`number` must be an int!'
        n = len(self.values)
        number_values = [number] + [0] * (n - 1)
        return RingRoot2(number_values)

    def to_float(self) -> float:
        a = self.values[0]
        b = self.values[1] * sqrt(2)
        return a + b

    def __repr__(self) -> str:
        s = f'{self.values[0]} + {self.values[1]}*sqrt(2)'
        return s
    
    def conj(self) -> RingRoot2:
        a, b = self.values
        return RingRoot2([a, -b])
    
    def __pow__(self, n: int) -> RingRoot2:
        if n < 0:
            raise ValueError('`n` must be a non-negative integer.')
        elif n == 0:
            return RingRoot2([1, 0])
        if n == 1:
            return self
        return self * self ** (n - 1)
    
    def __neg__(self) -> RingRoot2:
        return RingRoot2([-v for v in self.values])


class RingRootRoot2Plus2(AlgebraicInteger):

    def __init__(self, values: Sequence[int]) -> None:
        """
        An algebraic integer in Z[sqrt(sqrt(2)+2)]

        Args:
            values (Sequence[int]): A Sequence of length four in the form
                values[0] + values[1] * sqrt(2) 
                + values[2] * sqrt(sqrt(2)+2)
                + values[3] * sqrt(2)sqrt(sqrt(2)+2).

        Raises:
            ValueError: If `values` is not of length 4.
        """
        if len(values) != 4:
            raise ValueError('`values` must be of length 4.')
        super().__init__(values)

    def __add__(self, other: AlgebraicInteger) -> AlgebraicInteger:
        if isinstance(other, int):
            w, x, y, z = other, 0, 0, 0
        elif isinstance(other, RingRoot2):
            w, x, y, z = other.values[0], other.values[1], 0, 0
        else:
            w, x, y, z = other.values
        a = self.values[0] + w
        b = self.values[1] + x
        c = self.values[2] + y
        d = self.values[3] + z
        new_integer = RingRootRoot2Plus2([a, b, c, d])
        return new_integer

    def __sub__(self, other: int | AlgebraicInteger) -> AlgebraicInteger:
        if isinstance(other, int):
            w, x, y, z = other, 0, 0, 0
        elif isinstance(other, RingRoot2):
            w, x, y, z = other.values[0], other.values[1], 0, 0
        else:
            w, x, y, z = other.values
        a = self.values[0] - w
        b = self.values[1] - x
        c = self.values[2] - y
        d = self.values[3] - z
        new_integer = RingRootRoot2Plus2([a, b, c, d])
        return new_integer

    def __mul__(self, other: int | AlgebraicInteger) -> AlgebraicInteger:
        if isinstance(other, int):
            w, x, y, z = other, 0, 0, 0
        elif isinstance(other, RingRoot2):
            w, x, y, z = other.values[0], other.values[1], 0, 0
        else:
            w, x, y, z = other.values
        a, b, c, d = self.values
        aa = a*w + 2*b*x + 2*c*y + 2*c*z + 2*d*y + 4*d*z
        bb = a*x + b*w + c*y + 2*c*z + 2*d*y + 2*d*z
        cc = a*y + 2*b*z + c*w + 2*d*x
        dd = a*z + b*y + c*x + d*w
        new_integer = RingRootRoot2Plus2([aa, bb, cc, dd])
        return new_integer

    def int_to_algebraic_int(
        self,
        number: int | AlgebraicInteger,
    ) -> AlgebraicInteger:
        if isinstance(number, AlgebraicInteger):
            return number
        assert isinstance(number, int), '`number` must be an int!'
        n = len(self.values)
        number_values = [number] + [0] * (n - 1)
        return RingRootRoot2Plus2(number_values)

    def to_float(self) -> float:
        a = self.values[0]
        b = self.values[1] * sqrt(2)
        c = self.values[2] * sqrt(sqrt(2) + 2)
        d = self.values[3] * sqrt(2) * sqrt(sqrt(2) + 2)
        return a + b + c + d
    
    @staticmethod
    def convert_from_root2(root2: RingRoot2) -> RingRootRoot2Plus2:
        """
        Convert an algebraic integer in Z[sqrt(2)] to Z[sqrt(sqrt(2)+2)].

        Returns:
            (RingRootRoot2Plus2): The algebraic integer in Z[sqrt(sqrt(2)+2)].
        """
        a, b = root2.values
        new_values = [a, b, 0, 0]
        new_integer = RingRootRoot2Plus2(new_values)
        return new_integer

    def __repr__(self) -> str:
        s = f'{self.values[0]} + {self.values[1]}*sqrt(2)'
        s += f' + {self.values[2]}*sqrt(sqrt(2)+2)'
        s += f' + {self.values[3]}*sqrt(2)*sqrt(sqrt(2)+2)'
        return s


class DyadicComplexNumber:
    """A class representing numbers in Z[e^(i*pi/n), 1/2]."""
    def __init__(
        self,
        values: Sequence[int],
        denominator_exponent: int,
    ) -> None:
        """
        Construct a number as (prod_{k}^{m//2-1} a_k * e^(i*k*pi/m)) / 2^l.

        Args:
            values (Sequence[int]): The integer coefficient values of the
                numerator. This sequence must be of length m//2. Support
                is only offered for m == 8 (T) and m == 16 (sqrt T).
            
            denominator_exponent (int): The power of 2 in the denominator.
        
        Raises:
            ValueError: If `values` is not of length 4 (T) or 8 (sqrt T).
        """
        self.values = list(values).copy()
        self.denominator_exponent = denominator_exponent
    
    def match_base_size(self, other: DyadicComplexNumber) -> None:
        self_base, other_base = len(self.values), len(other.values)
        if other_base <= self_base:
            return
        if log2(other_base / self_base) != log2(other_base // self_base):
            m = 'New base must be a power of 2 of the old base.'
            raise ValueError(m)
        gap = other_base // self_base
        new_values = [0] * other_base
        for i, v in enumerate(self.values):
            new_values[i * gap] = v
        self.values = new_values

    def __add__(self, other: DyadicComplexNumber) -> DyadicComplexNumber:
        if len(self.values) < len(other.values):
            self.match_base_size(other)
        elif len(self.values) > len(other.values):
            other.match_base_size(self)
        lhs_values = self.values.copy()
        rhs_values = other.values.copy()
        offset = abs(self.denominator_exponent - other.denominator_exponent)
        if self.denominator_exponent < other.denominator_exponent:
            lhs_values = [v << offset for v in lhs_values]
        elif self.denominator_exponent > other.denominator_exponent:
            rhs_values = [v << offset for v in rhs_values]
        new_power = max(self.denominator_exponent, other.denominator_exponent)
        new_values = [a + b for a, b in zip(lhs_values, rhs_values)]
        new_num = DyadicComplexNumber(new_values, new_power)
        new_num.simplify()
        return new_num

    def __neg__(self) -> DyadicComplexNumber:
        new_values = [-v for v in self.values]
        return DyadicComplexNumber(new_values, self.denominator_exponent)

    def __sub__(self, other: DyadicComplexNumber) -> DyadicComplexNumber:
        return self + (-other)

    def __mul__(self, other: DyadicComplexNumber) -> DyadicComplexNumber:
        # Make power equal
        if len(self.values) < len(other.values):
            self.match_base_size(other)
        elif len(self.values) > len(other.values):
            other.match_base_size(self)
        lhs_values = self.values.copy()
        rhs_values = other.values.copy()
        new_power = self.denominator_exponent + other.denominator_exponent
        # FOIL
        m = len(lhs_values)
        new_values = [0] * m
        for i, coeff_a in enumerate(lhs_values):
            for j, coeff_b in enumerate(rhs_values):
                k = i + j
                new_coeff = coeff_a * coeff_b
                sign = (-1) ** (k >= m)
                new_values[k % m] += sign * new_coeff
        new_num = DyadicComplexNumber(new_values, new_power)
        new_num.simplify()
        return new_num
    
    def simplify(self) -> None:
        max_iterations = self.denominator_exponent
        for i in range(max_iterations):
            if all(v % 2 == 0 for v in self.values):
                self.values = [v >> 1 for v in self.values]
                self.denominator_exponent -= 1
            else:
                break
    
    def conjugate(self) -> DyadicComplexNumber:
        new_values = self.values.copy()
        new_values[1:] = [-v for v in reversed(new_values[1:])]
        return DyadicComplexNumber(new_values, self.denominator_exponent)
    
    def to_complex(self) -> complex:
        total = 0 * 1j
        for i, coeff in enumerate(self.values):
            phase = exp(1j * pi * i / len(self.values))
            total += coeff * phase
        total = total / (2 ** self.denominator_exponent)
        return total
    
    def abs(self) -> float:
        mag = abs(self.to_complex())
        return mag