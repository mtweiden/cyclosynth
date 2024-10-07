from random import choice
from random import randint

from cyclosynth.cliffords import H
from cyclosynth.cliffords import S
from cyclosynth.cliffords import X
from cyclosynth.cliffords import Y
from cyclosynth.cliffords import Z
from cyclosynth.matrix import U2Matrix
from cyclosynth.matrix import unitary_identity
from cyclosynth.matrix import unitary_rx
from cyclosynth.matrix import unitary_ry
from cyclosynth.matrix import unitary_rz
from cyclosynth.translation import translate_decomposition


cliffords = ['H', 'S', 'X', 'Y', 'Z']
magic_gates = ['T', 'Q']
rotations = ['x', 'y', 'z']


def random_cliffords(n: int) -> str:
    return ''.join(choice(cliffords) for _ in range(n))


def gates_to_unitary(gates: str, n: int) -> U2Matrix:
    mat = unitary_identity(n)
    for gate in gates:
        if gate == 'H':
            mat = H * mat
        elif gate == 'S':
            mat = S * mat
        elif gate == 'X':
            mat = X * mat
        elif gate == 'Y':
            mat = Y * mat
        elif gate == 'Z':
            mat = Z * mat
        elif gate == 'T':
            mat = unitary_rz(4) * mat
        elif gate == 'Q':
            mat = unitary_rz(8) * mat
        else:
            raise ValueError(f'Invalid gate: {gate}')
    return mat


class TestTranslation:

    num_trials = 100
    
    def test_equivalence(self) -> None:
        n = 8
        mg = 'Q'
        preamble_len = 8

        for _ in range(self.num_trials):
            preamble = random_cliffords(preamble_len)

            decomp = preamble
            goal = preamble

            num_phases = randint(1, 20)
            for _ in range(num_phases):
                phase_axis = choice(rotations)
                phase_length = choice(range(1, 4))
                decomp += phase_axis * phase_length
                if phase_axis == 'x':
                    goal += 'H' + mg * phase_length + 'H'
                elif phase_axis == 'y':
                    goal += 'SH' + mg * phase_length + 'HZS'
                elif phase_axis == 'z':
                    goal += mg * phase_length
            suffix = random_cliffords(preamble_len)

            decomp = decomp + suffix
            goal = goal + suffix

            result = translate_decomposition(decomp, mg)

            u_result = gates_to_unitary(result, n)
            u_goal = gates_to_unitary(goal, n)
            assert u_result.hilbert_schmidt_distance(u_goal) <= 1e-8