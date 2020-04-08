from typing import Sequence, Iterable

from source.tools.functions import ratio_generator_multiple, index_max


def make_path(rates: Iterable[Sequence[float]], fee: float) -> Iterable[int]:
    rg = None
    asset_last = -1
    for each_rate in rates:
        if rg is None:
            rg = ratio_generator_multiple(len(each_rate))
            next(rg)
        ratio = rg.send(each_rate)
        if ratio is None:
            continue
        i_max, v_max = index_max(ratio)
        if asset_last < 0 or (1. / (1. - fee) < v_max and asset_last != i_max):
            asset_last = i_max
        yield asset_last

