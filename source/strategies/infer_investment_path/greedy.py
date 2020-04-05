from typing import Sequence, Iterable

from source.tools.functions import ratio_generator_multiple, index_max


def make_path(rates: Iterable[Sequence[float]]) -> Iterable[int]:
    rg = None
    for each_rate in rates:
        if rg is None:
            rg = ratio_generator_multiple(len(each_rate))
            next(rg)
        ratio = rg.send(each_rate)
        if ratio is None:
            continue
        i_max, v_max = index_max(ratio)
        yield i_max
