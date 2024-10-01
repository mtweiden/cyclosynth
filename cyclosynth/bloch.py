from __future__ import annotations

from cmath import isclose

from cyclosynth.algebra import DyadicComplexNumber
from cyclosynth.matrix import Matrix
from cyclosynth.matrix import SO3Matrix
from cyclosynth.matrix import U2Matrix
from cyclosynth.matrix import SO3Matrix
from cyclosynth.matrix import bloch_rx
from cyclosynth.matrix import bloch_ry
from cyclosynth.matrix import bloch_rz
from cyclosynth.ratio import AlgebraicIntegerOverRoot2
from cyclosynth.ratio import AlgebraicIntegerOverRootRoot2Plus2


class BlochDecomposer:
    """
    For decomposing exactly implementable unitaries into discrete rotations.
    """
    def __init__(self, target: SO3Matrix | U2Matrix) -> None:
        """
        Construct a BlochDecomposer.
        
        Args:
            target (SO3Matrix | U2Matrix): An exactly implementable unitary
                represented as a matrix in SO3 or U2.
        """
        if isinstance(target, U2Matrix):
            target = BlochDecomposer.from_unitary(target)
        self.matrix = target
        self.base = 4 if Matrix.type_check(self.matrix) is \
                AlgebraicIntegerOverRoot2 else 8
        self.rx = bloch_rx(self.base)
        self.ry = bloch_ry(self.base)
        self.rz = bloch_rz(self.base)

    @staticmethod
    def from_unitary(unitary: U2Matrix) -> SO3Matrix:
        assert unitary.shape == (2, 2)
        a: DyadicComplexNumber = unitary[0, 0]
        b: DyadicComplexNumber = unitary[0, 1]
        c: DyadicComplexNumber = unitary[1, 0]
        d: DyadicComplexNumber = unitary[1, 1]
        adg = a.conjugate()
        bdg = b.conjugate()
        cdg = c.conjugate()
        ddg = d.conjugate()

        n = len(a.values)
        half_values = [0] * n
        half_values[0] = 1
        i_values = [0] * n
        i_values[n // 2] = 1
        half = DyadicComplexNumber(half_values, 1)
        half_i = DyadicComplexNumber(i_values, 1)
        dyadic_i = DyadicComplexNumber(i_values, 0)

        ax = half * ((c * bdg + d * adg) + (a * ddg + b * cdg))
        bx = -dyadic_i * (c * bdg + d * adg - ax)
        cx = a * bdg + b * adg
        assert isclose(-cx.to_complex(), (c * ddg + d * cdg).to_complex())

        ay = half_i * ((-c * bdg + d * adg) + (-a * ddg + b * cdg))
        by = -dyadic_i * (-dyadic_i * c * bdg + dyadic_i * d * adg - ay)
        cy = -dyadic_i * a * bdg + dyadic_i * b * adg
        assert isclose(
            cy.to_complex(),
            (dyadic_i * c * ddg - dyadic_i * d * cdg).to_complex(),
        )

        az = half * ((c * adg - d * bdg) + (a * cdg - b * ddg))
        bz = -dyadic_i * (c * adg - d * bdg - az)
        cz = a * adg - b * bdg
        assert isclose(-cz.to_complex(), (c * cdg - d * ddg).to_complex())

        values = [ax, bx, cx, ay, by, cy, az, bz, cz]
        if n == 16:
            int_type = AlgebraicIntegerOverRootRoot2Plus2
        else:
            int_type = AlgebraicIntegerOverRoot2
        values = [int_type.from_dyadic(v) for v in values]
        bloch = SO3Matrix(values)
        return bloch

    @staticmethod
    def to_unitary(bloch: Matrix) -> Matrix:
        pass

    def try_rx(self) -> tuple[int]:
        """
        Try applying rotations about the x-axis.

        Returns:
            tuple[int]: The maximum denominator exponents resulting from
                applying discrete x-axis rotations of `pi / self.base`.
                The index in the tuple corresponds to the number of con-
                secutive rotations applied. This tuple is always of length
                `self.base // 2 - 1`.
        """
        max_exponents = []
        mat = self.matrix.copy()
        for _ in range(self.base // 2 - 1):
            mat = self.rx * mat
            exponent = mat.maximum_denominator_exponent()
            max_exponents.append(exponent)
        return tuple(max_exponents)

    def try_ry(self) -> tuple[int]:
        """
        Try applying rotations about the y-axis.

        Returns:
            tuple[int]: The maximum denominator exponents resulting from
                applying discrete y-axis rotations of `pi / self.base`.
                The index in the tuple corresponds to the number of con-
                secutive rotations applied. This tuple is always of length
                `self.base // 2 - 1`.
        """
        max_exponents = []
        mat = self.matrix.copy()
        for _ in range(self.base // 2 - 1):
            mat = self.ry * mat
            exponent = mat.maximum_denominator_exponent()
            max_exponents.append(exponent)
        return tuple(max_exponents)

    def try_rz(self) -> tuple[int]:
        """
        Try applying rotations about the z-axis.

        Returns:
            tuple[int]: The maximum denominator exponents resulting from
                applying discrete z-axis rotations of `pi / self.base`.
                The index in the tuple corresponds to the number of con-
                secutive rotations applied. This tuple is always of length
                `self.base // 2 - 1`.
        """
        max_exponents = []
        mat = self.matrix.copy()
        for _ in range(self.base // 2 - 1):
            mat = self.rz * mat
            exponent = mat.maximum_denominator_exponent()
            max_exponents.append(exponent)
        return tuple(max_exponents)