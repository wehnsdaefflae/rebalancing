import random
from math import sqrt
from typing import Tuple


# TODO: make multivariate (https://de.wikipedia.org/wiki/Multiple_lineare_Regression)
from matplotlib import pyplot


class Regressor:
    def __init__(self, drag: int):
        self.drag = drag
        self.mean_x = 0.
        self.mean_y = 0.
        self.var_x = 0.
        self.var_y = 0.
        self.cov_xy = 0.
        self.initial = True

    def sim(self, x: float, y: float) -> float:
        # https://stackoverflow.com/questions/23762178/normalized-distance-between-3d-2d-points#23763851
        # https://en.wikipedia.org/wiki/Mahalanobis_distance
        # dev = sqrt(var)  (https://en.wikipedia.org/wiki/Standard_deviation)
        fx = self.output(x)
        d = (fx - y) ** 2
        if 0. >= d:
            return 1.

        if self.var_x == 0.:
            return 0.

        return 1. - min(1., d / self.var_y)

    def fit(self, x: float, y: float):
        dx = x - self.mean_x
        dy = y - self.mean_y

        self.var_x = (self.drag * self.var_x + dx ** 2) / (self.drag + 1)
        self.var_y = (self.drag * self.var_y + dy ** 2) / (self.drag + 1)
        self.cov_xy = (self.drag * self.cov_xy + dx * dy) / (self.drag + 1)

        if self.initial:
            self.mean_x = x
            self.mean_y = y
            self.initial = False

        else:
            self.mean_x = (self.drag * self.mean_x + x) / (self.drag + 1)
            self.mean_y = (self.drag * self.mean_y + y) / (self.drag + 1)

    def _get_parameters(self) -> Tuple[float, float]:
        a = 0. if self.var_x == 0. else self.cov_xy / self.var_x
        t = self.mean_y - a * self.mean_x
        return a, t

    def output(self, x: float) -> float:
        a, t = self._get_parameters()
        return x * a + t


if __name__ == "__main__":
    f = lambda _x: .3 * _x - 7.
    x = range(20)
    y = [f(_x) + (random.random() - .5) * 2. for _x in x]
    pyplot.plot(x, y, label="original")

    regressor = Regressor(10)
    for _x, _y in zip(x, y):
        regressor.fit(_x, _y)

    in_value = 1.
    o = regressor.output(in_value)
    s = regressor.sim(in_value, o*.9)

    pyplot.plot(x, [regressor.output(_x) for _x in x], label="fitted")

    pyplot.legend()
    pyplot.show()
