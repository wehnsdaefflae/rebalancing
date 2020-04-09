from typing import Iterable, Sequence

from source.tools.functions import generate_ratios


def simulate(rates: Iterable[Sequence[float]], path: Iterable[int], fee: float) -> Iterable[float]:
    ratios = generate_ratios(rates)
    amount = 1.
    asset_last = -1
    for each_ratios, each_asset in zip(ratios, path):
        yield amount
        amount *= each_ratios[each_asset] * (1. - fee) ** int(asset_last != each_asset)
        asset_last = each_asset

    yield amount
