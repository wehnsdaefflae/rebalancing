import random
from math import sqrt
from typing import Tuple

# TODO: make multivariate (https://de.wikipedia.org/wiki/Multiple_lineare_Regression)
import numpy
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
            self.mean_x = list(x)
            self.mean_y = y
            self.initial = False

        else:
            self.mean_x = [(self.drag * _mean_x + _x) / (self.drag + 1) for (_mean_x, _x) in zip(self.mean_x, x)]
            self.mean_y = (self.drag * self.mean_y + y) / (self.drag + 1)

    def output(self, x: Tuple[float, ...]) -> float:
        assert len(x) == self.dim
        xn = self._get_parameters()
        return sum(_x * _xn for (_x, _xn) in zip(x + (1.,), xn))

    def _get_parameters(self) -> Tuple[float, ...]:
        xn = tuple(0. if _var_x == 0. else _cov_xy / _var_x for (_cov_xy, _var_x) in zip(self.cov_xy, self.var_x))
        x0 = self.mean_y - sum(_xn * _mean_x for (_xn, _mean_x) in zip(xn, self.mean_x))
        parameters = *xn, x0
        return parameters


def plot_surface(ax: pyplot.Axes.axes, a: float, b: float, c: float, size: int):
    x = numpy.linspace(0, size, endpoint=False, num=size)
    y = numpy.linspace(0, size, endpoint=False, num=size)

    _X, _Y = numpy.meshgrid(x, y)
    _Z = a + b * _Y + c * _X

    ax.plot_surface(_X, _Y, _Z, alpha=.2, antialiased=False)


def test3d(x0: float, x1: float, x2: float, size: int = 15):
    from mpl_toolkits.mplot3d import Axes3D
    # https://stackoverflow.com/questions/48335279/given-general-3d-plane-equation-how-can-i-plot-this-in-python-matplotlib
    # https://stackoverflow.com/questions/36060933/matplotlib-plot-a-plane-and-points-in-3d-simultaneously
    f = lambda _x, _y: x2 * _x + x1 * _y + x0

    regressor = MultiRegressor(2, 10)
    fig = pyplot.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    X = []
    Y = []
    Z = []

    shuffled_a = list(range(size))
    shuffled_b = list(range(size))
    random.shuffle(shuffled_a)
    random.shuffle(shuffled_b)

    for each_x in shuffled_a:
        for each_y in shuffled_b:
            each_z = f(each_x, each_y)
            X.append(each_x)
            Y.append(each_y)
            Z.append(each_z)

            p = each_x, each_y
            ax.scatter(each_x, each_y, each_z, antialiased=False, alpha=.2)
            regressor.fit(p, each_z)

    p2, p1, p0 = regressor._get_parameters()
    plot_surface(ax, p0, p1, p2, size)

    dev = 0.
    for each_x, each_y, each_z in zip(X, Y, Z):
        p = each_x, each_y
        each_o = regressor.output(p)
        ax.scatter(each_x, each_y, each_o, antialiased=False, alpha=.2, color="black", marker="^")
        dev += (each_z - each_o) ** 2.

    print(dev)
    pyplot.show()


def test2d(s: float, o: float):
        f = lambda _x: s * _x + o
        X = range(20)
        Y = [f(_x) + 4. * (random.random() - .5) for _x in X]

        fig, ax = pyplot.subplots(1, sharex="all")
        ax.scatter(X, Y, label="original")

        r = MultiRegressor(1, 10)
        for _x, _y in zip(X, Y):
            r.fit((_x,),  _y)

        (a, ), t = r._get_parameters()
        Yd = [a * _x + t for _x in X]
        ax.plot(X, Yd, label="fit")

        var = sum((r.output((_x,)) - _t) ** 2. for (_x, _t) in zip(X, Y))
        print("{:5.2f}".format(var))

        pyplot.legend()
        pyplot.tight_layout()
        pyplot.show()


if __name__ == "__main__":
    random.seed(8746587)

    for _ in range(100):
        a = random.random() * 20. - 10
        b = random.random() * 100. - 50
        c = random.random() * 40. - 20
        test3d(a, b, c, size=10)
        # test2d(a, b)
