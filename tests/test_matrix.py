from itertools import product

from random import choice
from random import randint

from numpy import array
from numpy import cos
from numpy import exp
from numpy import isclose
from numpy import ndarray
from numpy import pi
from numpy import sin

from cyclosynth.matrix import Matrix
from cyclosynth.matrix import bloch_rx
from cyclosynth.matrix import bloch_ry
from cyclosynth.matrix import bloch_rz
from cyclosynth.matrix import unitary_rx
from cyclosynth.matrix import unitary_ry
from cyclosynth.matrix import unitary_rz


def bloch_rx_numpy(n: int) -> ndarray:
    c, s  = cos(pi / n), sin(pi / n)
    mat = array([[1, 0, 0], [0, c, s], [0, -s, c]])
    return mat


def bloch_ry_numpy(n: int) -> ndarray:
    c, s  = cos(pi / n), sin(pi / n)
    mat = array([[c, 0, -s], [0, 1, 0], [s, 0, c]])
    return mat


def bloch_rz_numpy(n: int) -> ndarray:
    c, s  = cos(pi / n), sin(pi / n)
    mat = array([[c, s, 0], [-s, c, 0], [0, 0, 1]])
    return mat


def dyadic_rx_numpy(n: int) -> ndarray:
    c = cos(pi / (2 * n))
    s = -1j * sin(pi / (2 * n))
    mat = array([[c, s], [s, c]])
    return mat


def dyadic_ry_numpy(n: int) -> ndarray:
    c = cos(pi / (2 * n))
    s = sin(pi / (2 * n))
    mat = array([[c, -s], [s, c]])
    return mat


def dyadic_rz_numpy(n: int) -> ndarray:
    pe = exp(1j * pi / (2 * n))
    me = exp(-1j * pi / (2 * n))
    mat = array([[me, 0], [0, pe]])
    return mat


class TestMatrix:

    num_trials = 100

    def test_dyadic(self) -> None:
        for n in range(2, 10):
            n = 2 ** n
            rx = unitary_rx(n)
            ry = unitary_ry(n)
            rz = unitary_rz(n)
            rx_n = dyadic_rx_numpy(n)
            ry_n = dyadic_ry_numpy(n)
            rz_n = dyadic_rz_numpy(n)
            for i, j in product(range(2), range(2)):
                assert isclose(rx[i, j].to_complex(), rx_n[i, j], atol=1e-6)
                assert isclose(ry[i, j].to_complex(), ry_n[i, j], atol=1e-6)
                assert isclose(rz[i, j].to_complex(), rz_n[i, j], atol=1e-6)

    def test_bloch(self) -> None:
        for n in [4, 8]:
            rx = bloch_rx(n)
            ry = bloch_ry(n)
            rz = bloch_rz(n)
            rx_n = bloch_rx_numpy(n)
            ry_n = bloch_ry_numpy(n)
            rz_n = bloch_rz_numpy(n)
            for i, j in product(range(3), range(3)):
                assert isclose(rx[i, j].to_float(), rx_n[i, j], atol=1e-6)
                assert isclose(ry[i, j].to_float(), ry_n[i, j], atol=1e-6)
                assert isclose(rz[i, j].to_float(), rz_n[i, j], atol=1e-6)

    def test_numerical_mul(self) -> None:
        rx4, rx8 = bloch_rx(4), bloch_rx(8)
        ry4, ry8 = bloch_ry(4), bloch_ry(8)
        rz4, rz8 = bloch_rz(4), bloch_rz(8)
        rx4_n, rx8_n = bloch_rx_numpy(4), bloch_rx_numpy(8)
        ry4_n, ry8_n = bloch_ry_numpy(4), bloch_ry_numpy(8)
        rz4_n, rz8_n = bloch_rz_numpy(4), bloch_rz_numpy(8)

        symbolic_choices = [rx4, rx8, ry4, ry8, rz4, rz8]
        numeric_choices = [rx4_n, rx8_n, ry4_n, ry8_n, rz4_n, rz8_n]

        def choose_op() -> tuple[Matrix, Matrix]:
            i = choice(range(len(symbolic_choices)))
            return symbolic_choices[i], numeric_choices[i]

        count = 0
        for _ in range(self.num_trials):
            symb, num = choose_op()
            for _ in range(randint(1, 64)):
                next_symb, next_num = choose_op()
                symb = symb * next_symb
                num = num @ next_num
                count += 1

            for i, j in product(range(3), range(3)):
                assert isclose(symb[i, j].to_float(), num[i, j], atol=1e-6)