from typing import Sequence, Iterable

from source.tools.functions import generate_ratios


def make_path(rates: Iterable[Sequence[float]], fee: float) -> Iterable[int]:
    def get_best_asset(_ratios: Sequence[float], _asset_current: int) -> int:
        i, v = max(enumerate(_ratios), key=lambda x: x[1] * (1. - fee) ** int(x[0] != _asset_current))
        return i

    ratios = generate_ratios(rates)
    asset = -1
    for each_ratio in ratios:
        asset = get_best_asset(each_ratio, asset)
        yield asset

