from random import random

from numpy import array
from numpy.linalg import eigvals

from cyclosynth.ellipse import Ellipse


from random import seed
seed(42)

def random_ellipse() -> Ellipse:
    a, b, d = 1000 * random(), 1000 * random(), 1000 * random()
    mat = array([[a, b], [b, d]])
    mat = (mat @ mat.T)
    assert all(eigvals(mat) > 0)
    a, b, d = mat[0, 0], mat[0, 1], mat[1, 1]
    return Ellipse([a, b, d])


class TestEllipse:

    num_trials = 1000

    def test_make_upright(self) -> None:
        for _ in range(self.num_trials):
            e = random_ellipse()
            if e.uprightness() >= 0.5:
                continue
            skew = e.skew()
            new_e = e.make_upright()
            assert new_e.skew() <= 0.5 * skew
            assert new_e.uprightness() >= 0.5
    
    def test_upright_operators(self) -> None:
        for _ in range(self.num_trials):
            e = random_ellipse()
            if e.uprightness() >= 0.5:
                continue
            new_e, op = e.make_upright(return_operator=True)
            manual_e = e.copy()
            manual_e = manual_e.apply_operator(op)
            assert new_e.is_close(manual_e)