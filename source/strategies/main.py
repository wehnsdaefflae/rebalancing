import random
from typing import Sequence, Union

from source.data.generators.snapshots_debug import get_random_rates, get_debug_rates
from source.strategies.infer_investment_path.greedy import make_path
from source.strategies.infer_investment_path.optimal_trading_memory import make_path_memory


def print_sequence(sequence: Union[Sequence[float], Sequence[int]]) -> str:
    return "\t".join(f"{v:5.2f}" if isinstance(v, float) else f"{v: 5d}" for v in sequence)


def print_rates(rates: Sequence[Sequence[float]]) -> str:
    rates_i = zip(*rates)
    rows_str = tuple(print_sequence(asset) for asset in rates_i)
    indices_str = print_sequence(list(range(len(rates))))
    return "\n".join((indices_str, ) + rows_str)


def main():
    random.seed(23235232)

    no_assets = 5
    fee = .01

    rates = list(get_random_rates(20, no_assets))

    path_dp = make_path_memory(iter(rates), no_assets, fee)
    path_greedy = make_path(rates, fee)

    print()
    print(print_rates(list(rates)))
    print()
    print(print_sequence(list(path_dp)) + "\tdp")
    print(print_sequence(list(path_greedy)) + "\tgreedy")


if __name__ == "__main__":
    main()
    # stack()
