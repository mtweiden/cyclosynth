from __future__ import annotations

from typing import Any
from typing import Sequence

from cyclosynth.algebra import DyadicComplexNumber
from cyclosynth.algebra import RingRoot2
from cyclosynth.ratio import AlgebraicIntegerOverRoot2
from cyclosynth.ratio import IntegerRatio
from cyclosynth.utils import discrete_cos
from cyclosynth.utils import discrete_sin
from cyclosynth.utils import dyadic_cos
from cyclosynth.utils import dyadic_sin


def unitary_identity(n: int) -> U2Matrix:
    one_values = [1] + [0] * (2 * n - 1)
    zero_values = [0] * (2 * n)
    one = DyadicComplexNumber(one_values, 0)
    zero = DyadicComplexNumber(zero_values, 0)
    mat = U2Matrix([one, zero, zero, one])
    return mat


def unitary_rx(n: int) -> U2Matrix:
    I_values = [0] * (2 * n)
    I_values[n] = 1
    I = DyadicComplexNumber(I_values, 0)
    c = dyadic_cos(1, 2 * n)
    s = -I * dyadic_sin(1, 2 * n)
    mat = U2Matrix([c, s, s, c])
    return mat


def unitary_ry(n: int) -> U2Matrix:
    c = dyadic_cos(1, 2 * n)
    s = dyadic_sin(1, 2 * n)
    mat = U2Matrix([c, -s, s, c])  # type: ignore
    return mat


def unitary_rz(n: int) -> U2Matrix:
    me_values, pe_values = [0] * (2 * n), [0] * (2 * n)
    me_values[-1] = -1
    pe_values[1] = 1
    me = DyadicComplexNumber(me_values, 0)
    pe = DyadicComplexNumber(pe_values, 0)
    zero = DyadicComplexNumber([0] * (2 * n), 0)
    mat = U2Matrix([me, zero, zero, pe])
    return mat


def bloch_identity() -> SO3Matrix:
    one = AlgebraicIntegerOverRoot2(RingRoot2([1, 0]), 0)
    zero = AlgebraicIntegerOverRoot2(RingRoot2([0, 0]), 0)
    mat = SO3Matrix(
        [
            one, zero, zero,
            zero, one, zero,
            zero, zero, one,
        ]
    )
    return mat


def bloch_rx(n: int, dagger: bool = False) -> SO3Matrix:
    c, s  = discrete_cos(n), discrete_sin(n)
    one = AlgebraicIntegerOverRoot2(RingRoot2([1, 0]), 0)
    zero = AlgebraicIntegerOverRoot2(RingRoot2([0, 0]), 0)
    mat = SO3Matrix(
        [
            one, zero, zero,
            zero, c, s,
            zero, -s, c,
        ]
    )
    if dagger:
        rx = mat.copy()
        for _ in range(2 * n - 2):
            mat = rx * mat
    return mat


def bloch_ry(n: int, dagger: bool = False) -> SO3Matrix:
    c, s  = discrete_cos(n), discrete_sin(n)
    one = AlgebraicIntegerOverRoot2(RingRoot2([1, 0]), 0)
    zero = AlgebraicIntegerOverRoot2(RingRoot2([0, 0]), 0)
    mat = SO3Matrix(
        [
            c, zero, -s,
            zero, one, zero,
            s, zero, c,
        ]
    )
    if dagger:
        ry = mat.copy()
        for _ in range(2 * n - 2):
            mat = ry * mat
    return mat


def bloch_rz(n: int, dagger: bool = False) -> SO3Matrix:
    c, s  = discrete_cos(n), discrete_sin(n)
    one = AlgebraicIntegerOverRoot2(RingRoot2([1, 0]), 0)
    zero = AlgebraicIntegerOverRoot2(RingRoot2([0, 0]), 0)
    mat = SO3Matrix(
        [
            c, s, zero,
            -s, c, zero,
            zero, zero, one,
        ]
    )
    if dagger:
        rz = mat.copy()
        for _ in range(2 * n - 2):
            mat = rz * mat
    return mat


class Matrix:
    def __init__(self, n: int, values: Sequence[IntegerRatio]) -> None:
        """
        Construct a Matrix.

        Args:
            n (int): The size of the matrix, either 2 or 3.

            values (Sequence[IntegerRatio]): The values of the matrix in row
                major form.

        Raises:
            ValueError: If len(values) != n * n.

            ValueError: If n != 2 or n != 3.
        
        TODO:
            - Implement in rust or C for performance. This is ~100x slower
              than numpy.
        """
        if len(values) != n * n:
            m = f'Matrix values must have length n**2, got {len(values)}.'
            ValueError(m)
        if n != 2 or n != 3:
            m = f'Matrix must be either 2x2 or 3x3, got {n}x{n}.'
            ValueError(m)
        self.n = n
        self.values = list(values).copy()

    @property
    def shape(self) -> tuple[int, int]:
        return (self.n, self.n)

    @staticmethod
    def type_check(matrix: Matrix) -> Any:
        ring_type = type(matrix.values[0])
        if not all(isinstance(x, ring_type) for x in matrix.values):
            m = 'All matrix elements must be members of the same ring. '
            m += f'Make sure every number is type {ring_type}.'
            ValueError(m)
        return ring_type

    def __getitem__(self, key: int | Sequence[int]) -> IntegerRatio:
        if isinstance(key, int):
            return self.values[key]
        assert len(key) == 2, 'Matrix indices must be 2-tuples.'
        x, y = key
        assert x < self.n and x >= 0, 'Matrix index x out of range.'
        assert y < self.n and y >= 0, 'Matrix index y out of range.'
        index = x * self.n + y
        return self.values[index]

    def __mul__(self, other: Matrix) -> Matrix:
        """
        Matrix multiplication.

        Args:
            other (Matrix): Right hand side matrix.

        Raises:
            ValueError:
        """
        raise NotImplementedError('Matrix multiplication not implemented.')
    
    def to_float(self) -> list[float]:
        return [x.to_float() for x in self.values]


class SO3Matrix(Matrix):
    def __init__(self, values: Sequence[IntegerRatio]) -> None:
        """
        Construct a Matrix in SO3.

        Args:
            values (Sequence[IntegerRatio]): The values of the matrix, in
                row-major order.

        Raises:
            ValueError: If len(values) != 9.

            ValueError: If the elements of values are not all members of the
                same ring.
        """
        if len(values) != 9:
            m = f'Matrix values must have length n**2, got {len(values)}.'
            ValueError(m)
        if not isinstance(values[0], IntegerRatio):
            m = f'Matrix values must be InterRatio, got {type(values[0])}.'
            TypeError(m)
        self.n = 3
        self.values = list(values).copy()
        Matrix.type_check(self)

    def copy(self) -> SO3Matrix:
        return SO3Matrix(self.values)
    
    def __mul__(self, other: SO3Matrix) -> SO3Matrix:
        a, b = self, other
        c11 = a[0,0] * b[0,0] + a[0,1] * b[1,0] + a[0,2] * b[2,0]
        c12 = a[0,0] * b[0,1] + a[0,1] * b[1,1] + a[0,2] * b[2,1]
        c13 = a[0,0] * b[0,2] + a[0,1] * b[1,2] + a[0,2] * b[2,2]
        c21 = a[1,0] * b[0,0] + a[1,1] * b[1,0] + a[1,2] * b[2,0]
        c22 = a[1,0] * b[0,1] + a[1,1] * b[1,1] + a[1,2] * b[2,1]
        c23 = a[1,0] * b[0,2] + a[1,1] * b[1,2] + a[1,2] * b[2,2]
        c31 = a[2,0] * b[0,0] + a[2,1] * b[1,0] + a[2,2] * b[2,0]
        c32 = a[2,0] * b[0,1] + a[2,1] * b[1,1] + a[2,2] * b[2,1]
        c33 = a[2,0] * b[0,2] + a[2,1] * b[1,2] + a[2,2] * b[2,2]
        c = SO3Matrix([c11, c12, c13, c21, c22, c23, c31, c32, c33])
        return c
    
    def exponents(self) -> tuple[int]:
        """Return the max exponent of each row of the matrix."""
        for value in self.values:
            value.simplify()
        exponents = []
        for i in range(3):
            row = max([self[i, j].denominator_power for j in range(3)])
            exponents.append(row)
        return tuple(exponents)
    
    def maximum_denominator_exponent(self) -> int:
        return max(self.exponents())


class U2Matrix(Matrix):
    def __init__(self, values: Sequence[DyadicComplexNumber]) -> None:
        """
        Construct a Matrix in U2.

        Args:
            values (Sequence[DyadicComplexNumber]): The values of the matrix,
                in row-major order.

        Raises:
            ValueError: If len(values) != 4.

            ValueError: If the elements of values are not all members of the
                same ring.
        """
        if len(values) != 4:
            m = f'Matrix values must have length n**2, got {len(values)}.'
            ValueError(m)
        if not isinstance(values[0], DyadicComplexNumber):
            m = f'Matrix values must be IntegerRatio, '
            m += f'got {type(values[0])}.'
            TypeError(m)
        self.n = 2
        self.values = list(values).copy()
        Matrix.type_check(self)

    def copy(self) -> U2Matrix:
        return U2Matrix(self.values)

    def __mul__(self, other: U2Matrix) -> U2Matrix:
        a, b = self, other
        c11 = a[0,0] * b[0,0] + a[0,1] * b[1,0]
        c12 = a[0,0] * b[0,1] + a[0,1] * b[1,1]
        c21 = a[1,0] * b[0,0] + a[1,1] * b[1,0]
        c22 = a[1,0] * b[0,1] + a[1,1] * b[1,1]
        c = U2Matrix([c11, c12, c21, c22])
        return c
    
    def to_float(self) -> list[complex]:
        return [v.to_complex() for v in self.values]
    
    def dagger(self) -> U2Matrix:
        a, b, c, d = self.values
        adg = a.conjugate()
        bdg = b.conjugate()
        cdg = c.conjugate()
        ddg = d.conjugate()
        return U2Matrix([adg, cdg, bdg, ddg])