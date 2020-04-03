from __future__ import annotations

import itertools
from typing import TypeVar, Sequence, Iterable, Tuple, Generator, Optional




def z_score_generator(drag: int = -1, offset: float = 0., scale: float = 1., clamp: Optional[Tuple[float, float]] = None) -> Generator[float, float, None]:
    # use to normalize input, enables more precise error value calculation for recurrent and failure approximations
    iteration = 0

    value = yield
    average = value
    deviation = 0.

    if clamp is not None:
        assert clamp[0] < clamp[1]

    while True:
        if deviation == 0.:
            value = yield 0.

        elif clamp is None:
            value = yield ((value - average) / deviation) * scale + offset

        else:
            r = ((value - average) / deviation) * scale + offset
            value = yield max(clamp[0], min(clamp[1], r))

        d = drag if drag >= 0 else iteration
        average = smear(average, value, d)
        deviation = smear(deviation, abs(value - average), d)

        iteration += 1


def z_score_normalized_generator() -> Generator[float, float, None]:
    yield from z_score_generator(drag=-1, scale=.25, offset=.5, clamp=(0., 1.))


def z_score_multiple_normalized_generator(no_values: int) -> Generator[Sequence[float], Sequence[float], None]:
    gs = tuple(z_score_normalized_generator() for _ in range(no_values))
    values = yield tuple(next(each_g) for each_g in gs)

    while True:
        values = yield tuple(each_g.send(x) for each_g, x in zip(gs, values))


def ratio_generator() -> Generator[float, Optional[float], None]:
    value_last = yield  # dont do an initial next?
    value = yield
    while True:
        ratio = 0. if value_last == 0. else value / value_last
        value_last = value
        value = yield ratio


def ratio_generator_multiple(no_values: int) -> Generator[Sequence[float], Optional[Sequence[float]], None]:
    gs = tuple(ratio_generator() for _ in range(no_values))
    for each_g in gs:
        next(each_g)

    values = yield
    ratios = tuple(g.send(v) for g, v in zip(gs, values))

    while True:
        values = yield None if None in ratios else ratios
        ratios = tuple(g.send(v) for g, v in zip(gs, values))


def smear(average: float, value: float, inertia: int) -> float:
    return (inertia * average + value) / (inertia + 1.)


T = TypeVar("T")


def accumulating_combinations_with_replacement(elements: Iterable[T], repetitions: int) -> Generator[Tuple[T, ...], None, None]:
    yield from (c for _r in range(repetitions) for c in itertools.combinations_with_replacement(elements, _r + 1))


def product(values: Sequence[float]) -> float:
    output = 1.
    for _v in values:
        output *= _v
    return output
