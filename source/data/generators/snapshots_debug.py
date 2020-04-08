import random
from typing import Iterator, Sequence


def get_random_sequence(start_value: float, length: int, gaps: float = 0.) -> Iterator[float]:
    value = start_value
    yield value

    for i in range(1, length):
        r = random.uniform(-.1, .1) * value
        value = value + r
        yield -1. if random.random() < gaps else value


def get_random_rates(size: int, no_assets: int, gaps: float = 0.) -> Iterator[Sequence[float]]:
    rg = tuple(get_random_sequence(random.uniform(10., 60.), size, gaps=gaps) for _ in range(no_assets))
    yield from zip(*rg)


def get_debug_rates() -> Iterator[Sequence[float]]:
    # rate_a = 59.69, 65.07, 61.44, 65.73, 59.37
    # rate_b =   -1.,   -1.,   -1.,   -1.,   -1.

    rate_a = 59.69, 65.07, 61.44, 65.73, 59.37
    rate_b = 30.06, 29.96, 27.06, 27.09, 29.13

    #rate_a = 60.98, -1.,   -1., 65.73, 59.37
    #rate_b = -1., -1., 27.06, 29.2, 29.13

    for a, b in zip(rate_a, rate_b):
        yield a, b


def get_decreasing_rates(size: int = 20, no_assets: int = 10) -> Iterator[Sequence[float]]:
    raise NotImplementedError()
