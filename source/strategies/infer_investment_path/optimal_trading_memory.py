# https://www.dropbox.com/s/ed5hm4rd0b8bz18/optimal.pdf?dl=0
from typing import Sequence, Tuple, Iterator, Iterable

# from matplotlib import pyplot
from source.tools.functions import generate_ratios
from source.tools.timer import Timer


def generate_matrix(
        no_assets: int,
        changes: Iterable[Sequence[float]], fees: float,
        bound: float = 0.) -> Iterable[Tuple[Sequence[int], Tuple[int, float]]]:

    assert 1. >= fees >= 0.
    values_objective = [1. for _ in range(no_assets)]

    for t, changes_asset in enumerate(changes):
        assert len(changes_asset) == no_assets

        asset_sources = list(range(no_assets))
        values_tmp = values_objective[:]
        for asset_to, each_change in enumerate(changes_asset):
            each_change = max(each_change, 0.)

            best_predecessor, value_max = max(
                (
                    # (asset_from, each_interest * (1. - float(asset_from != asset_to and 0 < t) * fees) * each_change)
                    (asset_from, each_interest * (1. - fees) ** int(asset_from != asset_to and 0 < t) * each_change)
                    for asset_from, each_interest in enumerate(values_objective)
                ), key=lambda x: x[1]
            )

            if value_max == values_objective[asset_to] * each_change:
                best_predecessor = asset_to

            asset_sources[asset_to], values_tmp[asset_to] = best_predecessor, value_max

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

        yield tuple(asset_sources), max(enumerate(values_objective), key=lambda x: x[1])
        if Timer.time_passed(2000):
            print(f"finished {t:d} time steps in matrix...")


def make_path(roi_matrix: Sequence[Sequence[float]]) -> Sequence[int]:
    print("determining optimal investment path...")
    len_path = len(roi_matrix)
    path = []
    for i in range(len_path - 1, -1, -1):
        v_max = -1.
        i_max = -1
        for j, v in enumerate(roi_matrix[i]):
            if v_max < v or (v_max == v and i < len_path - 1 and j == path[0]):
                v_max = v
                i_max = j
        path.insert(0, i_max)
        if Timer.time_passed(2000):
            print(f"finished {(len_path - i) * 100. / len_path:5.2f}% of generating path...")

    return path


def make_path_from_sourcematrix(matrix: Sequence[Tuple[Sequence[int], Tuple[int, float]]]) -> Sequence[int]:
    print(f"finding path in matrix...")
    len_path = len(matrix)

    snapshot, (asset_last, _) = matrix[-1]
    path = [asset_last]
    i = len_path - 2
    while i >= 0:
        asset_last = snapshot[asset_last]
        path.insert(0, asset_last)
        snapshot, (_, _) = matrix[i]
        i -= 1
        if Timer.time_passed(2000):
            print(f"finished {(len_path - i) * 100. / len_path:5.2f}% of making path...")

    return path


def make_path_memory(rates: Iterator[Sequence], no_assets: int, fees: float, bound: int = 100) -> Iterator[int]:
    ratios = generate_ratios(rates)
    matrix = generate_matrix(no_assets, ratios, fees, bound=bound)
    return make_path_from_sourcematrix(list(matrix))
