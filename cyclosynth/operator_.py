from __future__ import annotations

from typing import Sequence

from cyclosynth.algebra import AlgebraicInteger
from cyclosynth.ratio import IntegerRatio
from cyclosynth.matrix import Matrix
from cyclosynth.matrix import Vector


class Operator(Matrix):
    
    def __init__(self, values: Sequence[IntegerRatio | int]) -> None:
        super().__init__(2, values)
    
    def __mul__(
        self,
        other: Operator | IntegerRatio | Vector
    ) -> Operator | Vector:
        if isinstance(other, IntegerRatio):
            return self.ratiomul(other)
        elif isinstance(other, Operator):
            return self.matmul(other)
        elif isinstance(other, Vector):
            return self.vecmul(other)
        else:
            raise TypeError(f'No mult defined for {type(other)}.')
    
    def ratiomul(self, other: IntegerRatio) -> Operator:
        return Operator([x * other for x in self.values])
    
    def matmul(self, other: Operator) -> Operator:
        a, b = self, other
        c11 = a[0,0] * b[0,0] + a[0,1] * b[1,0]
        c12 = a[0,0] * b[0,1] + a[0,1] * b[1,1]
        c21 = a[1,0] * b[0,0] + a[1,1] * b[1,0]
        c22 = a[1,0] * b[0,1] + a[1,1] * b[1,1]
        return Operator([c11, c12, c21, c22])
    
    def vecmul(self, other: Vector) -> Vector:
        x, y = other.values
        c11, c12, c21, c22 = self.values
        x_new = c11 * x + c12 * y
        y_new = c21 * x + c22 * y
        return Vector([x_new, y_new])
    
    def act_on(
        self,
        point: Vector | Sequence[IntegerRatio],
    ) -> Vector:
        if not isinstance(point, Vector):
            point = Vector(point)
        point = self.vecmul(point)
        return point

    def __pow__(self, k: int) -> Operator:
        if k < 0:
            raise ValueError('Matrix power must be non-negative.')
        elif k == 0:
            return Operator([1, 0, 0, 1])
        elif k == 1:
            return self
        else:
            return self * self.__pow__(k - 1)
    
    def __add__(self, other: Operator) -> Operator:
        c11 = self[0, 0] + other[0, 0]
        c12 = self[0, 1] + other[0, 1]
        c21 = self[1, 0] + other[1, 0]
        c22 = self[1, 1] + other[1, 1]
        return Operator([c11, c12, c21, c22])
    
    @property
    def transpose(self) -> Operator:
        c11, c12, c21, c22 = self.values
        return Operator([c11, c21, c12, c22])

    def inv(self) -> Operator:
        a, b, c, d = self.values
        norm = (a * d - b * c).inverse()
        aa = d * norm
        bb = -b * norm
        cc = -c * norm
        dd = a * norm
        return Operator([aa, bb, cc, dd])
    
    def conj(self) -> Operator:
        c11, c12, c21, c22 = self.values
        return Operator([c11.conj(), c12.conj(), c21.conj(), c22.conj()])
    
    def __repr__(self) -> str:
        return f'Operator({self.values})'

