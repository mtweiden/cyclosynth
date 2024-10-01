from numpy import allclose, cos, pi, sin, array
from cyclosynth.utils import dyadic_cos, dyadic_sin

from cyclosynth.matrix import unitary_rx, unitary_ry, unitary_rz

n = 8
rx = unitary_rx(n)
rx_dy = array(rx.to_float()).reshape(2, 2)
c = cos(pi / (2 * n))
s = -1j * sin(pi / (2 * n))
rx_np = array([[c, s], [s, c]])
assert allclose(rx_np, rx_dy, atol=1e-6)

