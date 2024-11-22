from __future__ import annotations

from cmath import isclose

from cyclosynth.algebra import DyadicComplexNumber
from cyclosynth.cliffords import match_clifford
from cyclosynth.matrix import Matrix
from cyclosynth.matrix import SO3Matrix
from cyclosynth.matrix import U2Matrix
from cyclosynth.matrix import SO3Matrix
from cyclosynth.matrix import bloch_rx
from cyclosynth.matrix import bloch_ry
from cyclosynth.matrix import bloch_rz
from cyclosynth.matrix import unitary_rx
from cyclosynth.matrix import unitary_ry
from cyclosynth.matrix import unitary_rz
from cyclosynth.ratio import AlgebraicIntegerOverRoot2
from cyclosynth.ratio import AlgebraicIntegerOverRootRoot2Plus2
from cyclosynth.translation import translate_decomposition


class BlochDecomposer:
    """
    For decomposing exactly implementable unitaries into discrete rotations.
    """
    def __init__(self, target: U2Matrix, translate_gates: bool = True) -> None:
        """
        Construct a BlochDecomposer.
        
        Args:
            target (U2Matrix): An exactly implementable unitary represented
                as a matrix in U2.
            
            translate_gates (bool): If True, translate the decomposition into
                Clifford+T or Clifford+Q gates. Otherwise, return the decom-
                position in terms of discrete x, y, z rotations.
        """
        if not isinstance(target, U2Matrix):
            raise ValueError('The `target` must be a U2Matrix.')
        self.target = target.copy()
        self.matrix = BlochDecomposer.from_unitary(target)
        self.base = 4 if Matrix.type_check(self.matrix) is \
                AlgebraicIntegerOverRoot2 else 8
        self.translate_gates = translate_gates
        self.rx_so3 = bloch_rx(self.base, dagger=True)
        self.ry_so3 = bloch_ry(self.base, dagger=True)
        self.rz_so3 = bloch_rz(self.base, dagger=True)
        self.rx_u2 = unitary_rx(self.base, dagger=True)
        self.ry_u2 = unitary_ry(self.base, dagger=True)
        self.rz_u2 = unitary_rz(self.base, dagger=True)

    @staticmethod
    def from_unitary(unitary: U2Matrix) -> SO3Matrix:
        """
        Convert an exactly implementable U2Matrix into an SO3Matrix.

        Args:
            unitary (U2Matrix): A unitary matrix that is exactly implementable
                in the Clifford+RZ(pi/self.base) gate set.
        
        Returns:
            (SO3Matrix): The SO3Matrix representation of the unitary.
        """
        assert unitary.shape == (2, 2)
        a: DyadicComplexNumber = unitary[0, 0]
        b: DyadicComplexNumber = unitary[0, 1]
        c: DyadicComplexNumber = unitary[1, 0]
        d: DyadicComplexNumber = unitary[1, 1]
        adg = a.conj()
        bdg = b.conj()
        cdg = c.conj()
        ddg = d.conj()

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

    def try_rx(self, residual: SO3Matrix) -> tuple[int]:
        """
        Try applying rotations about the x-axis.

        Args:
            residual (SO3Matrix): The residual matrix being decomposed.

        Returns:
            tuple[int]: The maximum denominator exponents resulting from
                applying discrete x-axis rotations of `pi / self.base`.
                The index in the tuple corresponds to the number of con-
                secutive rotations applied. This tuple is always of length
                `self.base // 2 - 1`.
        """
        max_exponents = []
        mat = residual.copy()
        for _ in range(self.base // 2 - 1):
            mat = self.rx_so3 * mat
            exponent = mat.maximum_denominator_exponent()
            max_exponents.append(exponent)
        return tuple(max_exponents)

    def try_ry(self, residual: SO3Matrix) -> tuple[int]:
        """
        Try applying rotations about the y-axis.

        Args:
            residual (SO3Matrix): The residual matrix being decomposed.

        Returns:
            tuple[int]: The maximum denominator exponents resulting from
                applying discrete y-axis rotations of `pi / self.base`.
                The index in the tuple corresponds to the number of con-
                secutive rotations applied. This tuple is always of length
                `self.base // 2 - 1`.
        """
        max_exponents = []
        mat = residual.copy()
        for _ in range(self.base // 2 - 1):
            mat = self.ry_so3 * mat
            exponent = mat.maximum_denominator_exponent()
            max_exponents.append(exponent)
        return tuple(max_exponents)

    def try_rz(self, residual: SO3Matrix) -> tuple[int]:
        """
        Try applying rotations about the z-axis.

        Args:
            residual (SO3Matrix): The residual matrix being decomposed.

        Returns:
            tuple[int]: The maximum denominator exponents resulting from
                applying discrete z-axis rotations of `pi / self.base`.
                The index in the tuple corresponds to the number of con-
                secutive rotations applied. This tuple is always of length
                `self.base // 2 - 1`.
        """
        max_exponents = []
        mat = residual.copy()
        for _ in range(self.base // 2 - 1):
            mat = self.rz_so3 * mat
            exponent = mat.maximum_denominator_exponent()
            max_exponents.append(exponent)
        return tuple(max_exponents)
    
    def decompose(self) -> str:
        """
        Decompose self.matrix into discrete rotations.

        The decomposition is returned as a string consisting of elements in
        {x, y, z}, where each represents a discrete rotation about the corre-
        sponding axis by `pi / self.base`. As a convention, lower case char-
        acters indicate non-Clifford rotations while upper case characters
        indicate Clifford rotations.

        Returns:
            str: The decomposition of self.matrix. The string is returned in
                circuit order, i.e., the first character corresponds to the
                right most unitary in a product of matrices.
        """
        residual_so3 = self.matrix.copy()
        residual_u2 = self.target.copy()
        max_steps = residual_so3.maximum_denominator_exponent()
        decomposition = ''
        for _ in range(max_steps):
            x = self.try_rx(residual_so3)
            y = self.try_ry(residual_so3)
            z = self.try_rz(residual_so3)
            # select axis
            min_x, min_y, min_z = min(x), min(y), min(z)
            if min_x == min(min_x, min_y, min_z):
                axis = 'x'
                applications = x.index(min_x) + 1
                for _ in range(applications):
                    residual_so3 = self.rx_so3 * residual_so3
                    residual_u2 = residual_u2 * self.rx_u2
            elif min_y == min(min_x, min_y, min_z):
                axis = 'y'
                applications = y.index(min_y) + 1
                for _ in range(applications):
                    residual_so3 = self.ry_so3 * residual_so3
                    residual_u2 = residual_u2 * self.ry_u2
            elif min_z == min(min_x, min_y, min_z):
                axis = 'z'
                applications = z.index(min_z) + 1
                for _ in range(applications):
                    residual_so3 = self.rz_so3 * residual_so3
                    residual_u2 = residual_u2 * self.rz_u2
            else:
                raise ValueError('No minimum found.')

            decomposition += axis * applications

            if min(min_x, min_y, min_z) == 0:
                break

        cliffords = match_clifford(residual_u2)
        if cliffords is not None and cliffords is not 'I':
            decomposition += cliffords
        
        if self.translate_gates:
            magic_gate = 'T' if self.base == 4 else 'Q'
            decomposition = translate_decomposition(decomposition, magic_gate)

        return decomposition