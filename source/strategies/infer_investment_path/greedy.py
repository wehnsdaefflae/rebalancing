from typing import Sequence, Iterable, Tuple

from source.tools.functions import ratio_generator_multiple, index_max


def make_path(rates: Iterable[Sequence[float]], fee: float) -> Iterable[int]:
    def get_best_asset(ratios: Sequence[float], asset_current: int) -> int:
        i, v = max(enumerate(ratios), key=lambda x: x[1] * (1. - fee) ** int(x[0] != asset_current))
        return i

    rg = None
    asset = -1
    for each_rate in rates:
        if rg is None:
            rg = ratio_generator_multiple(len(each_rate))
            next(rg)
        ratio = rg.send(each_rate)
        if ratio is None:
            continue
        asset = get_best_asset(ratio, asset)
        yield asset

