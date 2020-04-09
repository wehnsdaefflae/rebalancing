# https://www.dropbox.com/s/ed5hm4rd0b8bz18/optimal.pdf?dl=0
from typing import Sequence, Iterator, Iterable

# from matplotlib import pyplot
from source.tools.functions import generate_ratios, index_max
from source.tools.timer import Timer


def generate_matrix(no_assets: int, ratios: Iterable[Sequence[float]], fee: float, bound: float = 100.) -> Iterable[Sequence[float]]:
    assert 1. >= fee >= 0.
    assert bound >= 0.

    values_objective = [1. for _ in range(no_assets)]

    for t, changes_asset in enumerate(ratios):
        assert len(changes_asset) == no_assets

        values_tmp = values_objective[:]
        for asset_to, each_ratio in enumerate(changes_asset):
            each_ratio = max(each_ratio, 0.)

            values_tmp[asset_to] = max(
                (
                    each_ratio * each_amount * (1. - fee) ** int(asset_from != asset_to or t == 0)
                    for asset_from, each_amount in enumerate(values_objective)
                )
            )

        if 0. < bound < max(values_tmp):
            for i, v in enumerate(values_tmp):
                values_objective[i] = v / bound

        elif 0. >= max(values_tmp):
            print("all negative")
            for i in range(len(values_objective)):
                values_objective[i] = 1.

        else:
            for i, v in enumerate(values_tmp):
                values_objective[i] = v

        yield tuple(values_objective)


def make_path(matrix: Sequence[Sequence[float]]) -> Sequence[int]:
    len_path = len(matrix)

    path = []
    i = len_path - 1
    while i >= 0:
        amounts = matrix[i]
        asset_last, _ = index_max(amounts)
        path.insert(0, asset_last)
        i -= 1
        if Timer.time_passed(2000):
            print(f"finished {(len_path - i) * 100. / len_path:5.2f}% of making path...")

    return path


def make_path_memory(rates: Iterator[Sequence], no_assets: int, fees: float, bound: int = 100) -> Iterable[int]:
    ratios = generate_ratios(rates)
    matrix = generate_matrix(no_assets, ratios, fees, bound=bound)
    matrix_list = list(matrix)
    return make_path(matrix_list)
