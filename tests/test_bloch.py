from random import randint
from random import choice
from random import choices

from cyclosynth.algebra import RingRoot2
from cyclosynth.algebra import RingRootRoot2Plus2
from cyclosynth.bloch import BlochDecomposer
from cyclosynth.cliffords import clifford_gates_to_u2
from cyclosynth.matrix import unitary_identity
from cyclosynth.matrix import unitary_rx
from cyclosynth.matrix import unitary_ry
from cyclosynth.matrix import unitary_rz
from cyclosynth.matrix import U2Matrix
from cyclosynth.ratio import AlgebraicIntegerOverRoot2
from cyclosynth.ratio import AlgebraicIntegerOverRootRoot2Plus2


def rand_integer_values(
    n: int,
    min_val: int = -1_000_000_000_000_000,
    max_val: int = 1_000_000_000_000_000,
) -> list[int]:
    return [randint(min_val, max_val) for _ in range(n)]


def random_ringroot2() -> AlgebraicIntegerOverRoot2:
    integer = RingRoot2(rand_integer_values(2))
    power = randint(0, 100)
    return AlgebraicIntegerOverRoot2(integer, power)


def random_ringrootroot2plus2() -> AlgebraicIntegerOverRootRoot2Plus2:
    integer = RingRootRoot2Plus2(rand_integer_values(4))
    power = randint(0, 100)
    return AlgebraicIntegerOverRootRoot2Plus2(integer, power)


def random_u2(n: int) -> U2Matrix:
    rx, ry, rz = unitary_rx(n), unitary_ry(n), unitary_rz(n)
    mat = unitary_rx(n)
    for _ in range(randint(1, 100)):
        gate = choice([rx, ry, rz])
        mat = mat * gate
    return mat


def construct_unitary(n: int, gates: str) -> U2Matrix:
    mat = unitary_identity(n)
    cliffords = clifford_gates_to_u2.keys()
    for gate in gates:
        if gate == 'x':
            g = unitary_rx(n)
        elif gate == 'y':
            g = unitary_ry(n)
        elif gate == 'z':
            g = unitary_rz(n)
        elif gate in cliffords:
            g = clifford_gates_to_u2[gate]
        mat = g * mat
    return mat

class TestBloch:

    num_trials = 100

    def test_u2_constructor(self) -> None:
        BlochDecomposer(random_u2(4))
        BlochDecomposer(random_u2(8))
    
    def test_try_rz(self) -> None:
        n = 8
        for _ in range(self.num_trials):
            target = unitary_identity(n)
            num_rz_gates = randint(1, 10)
            if num_rz_gates % 2 == 0:
                num_rz_gates += 1
            for _ in range(num_rz_gates):
                target = unitary_rz(n) * target 
            bloch = BlochDecomposer(target)
            min_val = min(bloch.try_rz(bloch.matrix))
            assert all(mde >= min_val for mde in bloch.try_rx(bloch.matrix))
            assert all(mde >= min_val for mde in bloch.try_ry(bloch.matrix))
    
    def test_decompose_sqrtt(self) -> None:
        n = 8
        length = 100
        for _ in range(self.num_trials):
            def constuct_u2(gates: str) -> U2Matrix:
                mat = unitary_identity(n)
                x, y, z = unitary_rx(n), unitary_ry(n), unitary_rz(n)
                for gate in gates:
                    if gate == 'x':
                        mat = x * mat
                    elif gate == 'y':
                        mat = y * mat
                    else:
                        mat = z * mat
                return mat
 
            gates = ''.join(choices('xyz', k=length))
            # gates += 'x' * 8
            u2 = constuct_u2(gates)
            bloch = BlochDecomposer(u2)
            decomposition = bloch.decompose()
 
            # If this check fails, compare unitaries
            if not decomposition == gates:
                decomposition_u = construct_unitary(n, decomposition)
                gates_u = construct_unitary(n, gates)
                dist = decomposition_u.hilbert_schmidt_distance(gates_u)
                assert dist < 1e-8

    def test_decompose_t(self) -> None:
        n = 4
        length = 100
        for _ in range(self.num_trials):
            def constuct_u2(gates: str) -> U2Matrix:
                mat = unitary_identity(n)
                x, y, z = unitary_rx(n), unitary_ry(n), unitary_rz(n)
                for gate in gates:
                    if gate == 'x':
                        mat = x * mat
                    elif gate == 'y':
                        mat = y * mat
                    else:
                        mat = z * mat
                return mat

            gates = ''.join(choices('xyz', k=length))
            u2 = constuct_u2(gates)
            bloch = BlochDecomposer(u2)
            decomposition = bloch.decompose()

            # If this check fails, compare unitaries
            if not decomposition == gates:
                decomposition_u = construct_unitary(n, decomposition)
                gates_u = construct_unitary(n, gates)
                dist = decomposition_u.hilbert_schmidt_distance(gates_u)
                assert dist < 1e-8