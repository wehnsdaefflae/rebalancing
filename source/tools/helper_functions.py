import math
from typing import List, Iterable, Generator


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


def normalize(l: List[float]) -> List[float]:
    min_val, max_val = float("inf"), float("-inf")
    for v in l:
        if v < min_val:
            min_val = v
        elif max_val < v:
            max_val = v

    d = max_val - min_val
    return [(v - min_val) / d for v in l]


def smoothing_generator(values: Iterable[float], drag: int) -> Generator[float, None, None]:
    smooth = 0.
    first = True
    for each_value in values:
        if first:
            smooth = each_value
            first = False
        else:
            smooth = (smooth * drag + each_value) / (drag + 1.)
        yield smooth