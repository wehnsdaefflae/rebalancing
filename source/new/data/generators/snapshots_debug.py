import random
from typing import Iterator, Tuple, Sequence


def get_random_sequence(start_value: float, length: int) -> Iterator[float]:
    s = [-1. for _ in range(length)]
    s[0] = start_value

    for i in range(1, length):
        value_last = s[i-1]
        r = random.uniform(-.1, .1) * value_last
        s[i] = value_last + r

    return (v for v in s)


def get_random_rates(size: int = 20, no_assets: int = 10, gaps: float = 0.) -> Iterator[Tuple[int, Sequence[float]]]:
    random.seed(25235)

    sequences = list(list(get_random_sequence(random.uniform(10., 60.), size)) for _ in range(no_assets))
    for each_list in sequences:
        for i in range(len(each_list)):
            if random.random() < gaps:
                each_list[i] = -1.

    return ((i, x) for i, x in enumerate(zip(*sequences)))


def get_debug_rates() -> Iterator[Tuple[int, Sequence[float]]]:
    rate_a = 59.69, 65.07, 61.44, 65.73, 59.37
    rate_b =   -1.,   -1.,   -1.,   -1.,   -1.

    #rate_a = 59.69, 65.07, 61.44, 65.73, 59.37
    #rate_b = 30.06, 29.96, 27.06, 27.09, 29.13

    #rate_a = 60.98, -1.,   -1., 65.73, 59.37
    #rate_b = -1., -1., 27.06, 29.2, 29.13

    return ((i, r) for i, r in enumerate(zip(rate_a, rate_b)))


def get_decreasing_rates(size: int = 20, no_assets: int = 10) -> Iterator[Tuple[int, Sequence[float]]]:
    random.seed(235235)

    rates = tuple([] for _ in range(no_assets))

    for each_rate in rates:
        each_rate.append(random.uniform(10., 60.))

    for each_rate in rates:
        for _ in range(size):
            each_rate.append(each_rate[-1] * .9)

    return ((i, x) for i, x in enumerate(rates))