import math


def g(x):
    if x == 0:
        return 0
    return 1. / (2 ** math.ceil(math.log(x + 1, 2)))


def h(x):
    if x == 0:
        return 0
    return (2 ** math.ceil(math.log(x + 1, 2))) - x - 1


def distribute_circular(x):
    assert x >= 0
    if x == 0:
        return 0.
    rec_x = h(x - 1)
    return distribute_circular(rec_x) + g(x)
