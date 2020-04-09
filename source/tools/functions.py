from __future__ import annotations

import glob
import itertools
import os
from typing import TypeVar, Sequence, Iterable, Tuple, Generator, Optional, Union

from source.config import RAW_BINANCE_DIR


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

    while True:
        ratios = tuple(g.send(v) for g, v in zip(gs, values))
        values = yield None if None in ratios else ratios


def smear(average: float, value: float, inertia: int) -> float:
    return (inertia * average + value) / (inertia + 1.)


def index_max(values: Sequence[float], key=lambda x: x) -> Tuple[int, float]:
    i_max, v_max = max(enumerate(values), key=lambda x: key(x[1]))
    return i_max, v_max


T = TypeVar("T")


def accumulating_combinations_with_replacement(elements: Iterable[T], repetitions: int) -> Generator[Tuple[T, ...], None, None]:
    yield from (c for _r in range(repetitions) for c in itertools.combinations_with_replacement(elements, _r + 1))


def product(values: Sequence[float]) -> float:
    output = 1.
    for _v in values:
        output *= _v
    return output


def generate_ratios(generator_values: Iterable[Sequence[float]]) -> Iterable[Sequence[float]]:
    values_last = None
    for values_this in generator_values:
        if values_last is not None and None not in values_last:
            yield tuple(
                -1. if 0. >= v_l or v_t < 0. else v_t / v_l
                for v_t, v_l in zip(values_this, values_last)
            )
        values_last = values_this


def get_pairs_from_filesystem() -> Sequence[Tuple[str, str]]:
    return tuple(
        (x[:-3], x[-3:])
        for x in (
            os.path.splitext(y)[0]
            for y in (
                os.path.basename(z)
                for z in sorted(glob.glob(RAW_BINANCE_DIR + "*.csv"))
            )
        )
    )


def combine_assets_header(pairs: Sequence[Tuple[str, str]], header: Sequence[str]) -> Sequence[str]:
    return ("close_time", ) + tuple(f"{each_pair[0]:s}-{each_pair[1]:s}_{column:s}" for each_pair in pairs for column in header if "close_time" not in column)


def convert_to_string(value: Union[int, float]) -> str:
    if isinstance(value, int):
        return f"{value:d}"

    elif isinstance(value, float):
        return f"{value:.8f}"

    raise ValueError("inconvertible")