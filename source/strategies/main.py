import random
from typing import Sequence, Union

from source.data.generators.snapshots_debug import get_random_rates
from source.strategies.infer_investment_path.greedy import make_path
from source.strategies.infer_investment_path.optimal_trading_memory import make_path_memory
from source.strategies.simulations.simulation import simulate


def print_sequence(sequence: Sequence[Union[float, int, str]]) -> str:
    string = "\t".join(f"{v:6.3f}" if isinstance(v, float) else f"{v: 6d}" if isinstance(v, int) else f"{v:>6s}" for v in sequence)
    return string


def print_rates(rates: Sequence[Sequence[float]]) -> str:
    rates_i = zip(*rates)
    rows_str = tuple(print_sequence((i,) + asset) for i, asset in enumerate(rates_i))
    indices_str = print_sequence(list(range(-1, len(rates))))
    return "\n".join((indices_str, ) + rows_str)


def main():
    random.seed(2323424)

    no_assets = 5
    fee = .01

    while True:
        rates = list(get_random_rates(20, no_assets))

        path_dp = list(make_path_memory(iter(rates), no_assets, fee))
        path_greedy = list(make_path(rates, fee))

        if path_dp == path_greedy:
            continue

        print()
        print(print_rates(list(rates)))
        print()
        print(print_sequence(["dp"] + path_dp))
        print(print_sequence(["grd"] + path_greedy))
        print()
        print(print_sequence(["dp"] + list(simulate(rates, path_dp, fee))))
        print(print_sequence(["grd"] + list(simulate(rates, path_greedy, fee))))

        break


if __name__ == "__main__":
    main()
    # stack()
