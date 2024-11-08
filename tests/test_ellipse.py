from random import random

from numpy import array
from numpy.linalg import eigvals

from cyclosynth.ellipse import Ellipse


# from random import seed
# seed(42)

class TestEllipse:

    num_trials = 1000

    def test_make_upright(self) -> None:

        for _ in range(self.num_trials):
            a, b, d = 1000 * random(), 1000 * random(), 1000 * random()
            mat = array([[a, b], [b, d]])
            mat = (mat @ mat.T)
            a, b, d = mat[0, 0], mat[0, 1], mat[1, 1]
            if any(eigvals(mat) <= 0):
                continue
            e = Ellipse([a, b, d], [0, 0])
            if e.uprightness() >= 0.5:
                continue
            skew = e.skew()
            new_e = e.make_upright()
            assert new_e.skew() <= 0.5 * skew
            assert new_e.uprightness() >= 0.5