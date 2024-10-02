from random import randint
from random import choice

from cyclosynth.algebra import RingRoot2
from cyclosynth.algebra import RingRootRoot2Plus2
from cyclosynth.bloch import BlochDecomposer
from cyclosynth.matrix import bloch_identity
from cyclosynth.matrix import bloch_rx
from cyclosynth.matrix import bloch_ry
from cyclosynth.matrix import bloch_rz
from cyclosynth.matrix import unitary_rx
from cyclosynth.matrix import unitary_ry
from cyclosynth.matrix import unitary_rz
from cyclosynth.matrix import SO3Matrix
from cyclosynth.matrix import U2Matrix
from cyclosynth.ratio import AlgebraicIntegerOverRoot2
from cyclosynth.ratio import AlgebraicIntegerOverRootRoot2Plus2


# from random import seed
# seed(42)


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


class TestBloch:

    num_trials = 100

    def test_u2_constructor(self) -> None:
        BlochDecomposer(random_u2(4))
        BlochDecomposer(random_u2(8))

    def test_so3_constructor(self) -> None:
        values_n4 = [random_ringroot2() for _ in range(9)]
        values_n8 = [random_ringrootroot2plus2() for _ in range(9)]
        bloch_n4 = BlochDecomposer(SO3Matrix(values_n4))
        assert bloch_n4.base == 4
        bloch_n8 = BlochDecomposer(SO3Matrix(values_n8))
        assert bloch_n8.base == 8
    
    def test_rz_simple(self) -> None:
        n = 8
        for _ in range(self.num_trials):
            num_rz_gates = randint(1, 50)
            if num_rz_gates % 2 == 0:
                num_rz_gates += 1
            target = bloch_identity()
            for _ in range(num_rz_gates):
                target = bloch_rz(n) * target 
            bloch = BlochDecomposer(target)
            index = (n // 2 - num_rz_gates) % (n // 2) - 1
            assert bloch.try_rz()[index] == 0
            assert all(mde > 0 for mde in bloch.try_rx())
            assert all(mde > 0 for mde in bloch.try_ry())
    
    def test_rz_complex(self) -> None:
        n = 8
        for _ in range(self.num_trials):
            target = SO3Matrix([random_ringrootroot2plus2() for _ in range(9)])
            num_rz_gates = randint(1, 10)
            if num_rz_gates % 2 == 0:
                num_rz_gates += 1
            for _ in range(num_rz_gates):
                target = bloch_rz(n) * target 
            bloch = BlochDecomposer(target)
            min_val = min(bloch.try_rz())
            assert all(mde >= min_val for mde in bloch.try_rx())
            assert all(mde >= min_val for mde in bloch.try_ry())