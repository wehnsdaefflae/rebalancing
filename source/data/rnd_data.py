import matplotlib.pyplot as plt
import random
import numpy

from matplotlib import pyplot


def histo():
    mean, stddev, size = .0, .05, 100000
    data = [random.gauss(mean, stddev) for c in range(size)]

    mn = sum(data) / size
    sd = (sum(x * x for x in data) / size - (sum(data) / size) ** 2) ** 0.5

    print("Sample mean = %g; Stddev = %g; max = %g; min = %g for %i values"
          % (mn, sd, max(data), min(data), size))

    plt.hist(data, bins=50)
    plt.show()


def relative_brownian(initial=1., drift=.0, volatility=.01):  # relative normally distributed change
    a = initial
    while True:
        yield a
        # r = random.random()
        r = numpy.random.random()
        a = a * (1. + drift + volatility * (r - .5))


def absolute_brownian(initial=1., factor=1., relative_bias=0.):  # constant equiprobable change
    a = initial
    while True:
        yield a
        if 0. < a:
            rnd_value = random.gauss(0., .2 / 4.)
            rnd_value = 2. * factor * random.random() - factor + relative_bias * factor
            a = max(a + rnd_value / 100., .0)


if __name__ == "__main__":
    g = relative_brownian(initial=100., drift=.0)
    X = range(10000)
    Y = [next(g) for _ in X]
    pyplot.plot(X, Y)
    pyplot.show()
