from typing import Tuple, Sequence, Optional, List

from source.data.generators.snapshots_debug import get_random_rates
from source.strategies.infer_investment_path.optimal_trading import generate_multiple_changes, generate_matrix, make_path_from_sourcematrix
from source.strategies.simulations.simulation import simulate


def split_time_and_data(
        input_data: Tuple[int, Sequence[float]],
        timestamp_storage: Optional[List[int]] = None,
        rate_storage: Optional[List[Sequence[float]]] = None) -> Sequence[float]:

    timestamp, data = input_data

    if timestamp_storage is not None:
        timestamp_storage.append(timestamp)

    if rate_storage is not None:
        rate_storage.append(data)

    return data


def compare():
    no_assets = 3
    get_rates = lambda: get_random_rates(size=10, no_assets=no_assets, gaps=.4)
    fees = .01

    # source
    timestamps = []
    rates = []
    generate_rates = (split_time_and_data(x, timestamps, rates) for x in get_rates())
    generate_rates_fix = list(generate_rates)
    generate_rates = (x for x in generate_rates_fix)

    print("tick    " + "".join(f"{i: 9d}" for i in range(len(rates))))
    print()
    for i, each_rate in enumerate(zip(*rates)):
        print(f"ass_{i:03d} " + "".join(f"{x:9.2f}" for x in each_rate))
    print()

    matrix_change = generate_multiple_changes(generate_rates)
    matrix_change_fix = list(matrix_change)
    print("change matrix source")
    print("\n".join(["    ".join(["        "] + [f"{max(0., v):5.2f}" for v in x]) for x in zip(*[y for y in matrix_change_fix])]))
    print()

    matrix = generate_matrix(no_assets, matrix_change_fix, .01, bound=100)
    matrix_fix = tuple(matrix)

    print("asset deprecated matrix source")
    print("\n".join(["".join(["     "] + [f"{v: 9d}" for v in x]) for x in zip(*[y[0] for y in matrix_fix])]))
    print()
    print(f"roi: {matrix_fix[-1][1][1]:5.5f}")
    print()

    path_new = make_path_from_sourcematrix(matrix_fix)

    roi_path = simulate(rates, path_new, fees)
    next(roi_path)
    print("path    " + "".join(f"  ass_{x:03d}" for x in path_new))
    print("reward  " + "".join(f"{x:9.2f}" for x in roi_path))
    print()


def main():
    pass


if __name__ == "__main__":
    compare()
