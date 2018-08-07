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

        if self.var_y == 0.:
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
        return t, a

    def output(self, x: float) -> float:
        x0, x1 = self._get_parameters()
        return x0 + x1 * x


class MultiRegressor:
    def __init__(self, dim: int, drag: int):
        # https://mubaris.com/2017/09/28/linear-regression-from-scratch/
        self.drag = drag
        self.dim = dim
        self.mean_x = [0. for _ in range(dim)]
        self.mean_y = 0.
        self.var_x = [0. for _ in range(dim)]
        self.var_y = 0.
        self.cov_xy = [0. for _ in range(dim)]
        self.initial = True

    def sim(self, x: Tuple[float, ...], y: float) -> float:
        assert len(x) == self.dim
        fx = self.output(x)
        d = (fx - y) ** 2
        if 0. >= d:
            return 1.

        if self.var_y == 0.:
            return 0.

        return 1. - min(1., d / self.var_y)

    def fit(self, x: Tuple[float, ...], y: float):
        assert len(x) == self.dim

        dx = [_x - _mean_x for (_x, _mean_x) in zip(x, self.mean_x)]
        dy = y - self.mean_y

        self.var_x = [(self.drag * _var_x + _dx ** 2) / (self.drag + 1) for (_var_x, _dx) in zip(self.var_x, dx)]
        self.var_y = (self.drag * self.var_y + dy ** 2) / (self.drag + 1)
        self.cov_xy = [(self.drag * _cov_xy + _dx * dy) / (self.drag + 1) for (_cov_xy, _dx) in zip(self.cov_xy, dx)]

        if self.initial:
            self.mean_x = x
            self.mean_y = y
            self.initial = False

        else:
            self.mean_x = [(self.drag * _mean_x + _x) / (self.drag + 1) for (_mean_x, _x) in zip(self.mean_x, x)]
            self.mean_y = (self.drag * self.mean_y + y) / (self.drag + 1)

    def output(self, x: Tuple[float, ...]) -> float:
        assert len(x) == self.dim
        xn, x0 = self._get_parameters()
        return x0 + sum(_x * _xn for (_x, _xn) in zip(x, xn))

    def _get_parameters(self) -> Tuple[Tuple[float, ...], float]:
        xn = tuple(0. if _var_x == 0. else _cov_xy / _var_x for (_cov_xy, _var_x) in zip(self.cov_xy, self.var_x))
        x0 = self.mean_y - sum(_xn * _mean_x for (_xn, _mean_x) in zip(xn, self.mean_x))
        return xn, x0


if __name__ == "__main__":
    from mpl_toolkits.mplot3d import Axes3D
    # https://stackoverflow.com/questions/48335279/given-general-3d-plane-equation-how-can-i-plot-this-in-python-matplotlib
    # https://stackoverflow.com/questions/36060933/matplotlib-plot-a-plane-and-points-in-3d-simultaneously
    f = lambda _x, _y: .3 * _x + -.4 * _y - 7.
    x = []
    y = []
    z = []
    for _x in range(20):
        for _y in range(20):
            v = f(_x, _y) + (4. * random.random() - 2.)
            x.append(_x)
            y.append(_y)
            z.append(v)

    fig = pyplot.figure()
    ax = fig.add_subplot(111, projection='3d')
    for _x, _y, _z in zip(x, y, z):
        ax.scatter(_x, _y, _z)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    pyplot.show()

    exit()

    pyplot.plot(x, y, label="original")

    regressor = MultiRegressor(1, 10)
    for _x, _y in zip(x, y):
        regressor.fit((_x, ), _y)

    in_value = 1.
    o = regressor.output((in_value, ))
    s = regressor.sim((in_value, ), o * .9)

    pyplot.plot(x, [regressor.output((_x, )) for _x in x], label="fitted")

    pyplot.legend()
    pyplot.show()

