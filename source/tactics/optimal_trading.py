# https://www.dropbox.com/s/ed5hm4rd0b8bz18/optimal.pdf?dl=0
import math
import random
from typing import Sequence, Tuple

# from matplotlib import pyplot
from matplotlib import pyplot


def get_sequence(start_value: float, length: int) -> Sequence[float]:
    s = [-1. for _ in range(length)]
    s[0] = start_value

    for i in range(1, length):
        value_last = s[i-1]
        r = random.uniform(-.1, .1) * value_last
        s[i] = value_last + r

    return s


def forward(rates: Sequence[Sequence[float]]) -> Tuple[Sequence[int], float]:
    no_assets = len(rates)
    len_sequence, = set(len(x) for x in rates)
    len_path = len_sequence - 1

    values_objective = [1. for _ in rates]
    paths = tuple([] for _ in rates)

    for t in range(len_path):
        rates_now = tuple(x[t] for x in rates)
        rates_next = tuple(x[t + 1] for x in rates)
        values_tmp = values_objective[:]
        for asset_to, (rate_to_now, rate_to_next) in enumerate(zip(rates_now, rates_next)):
            change = rate_to_next / rate_to_now

            asset_from = -1
            value_max = -1.
            for asset_tmp in range(no_assets):
                value_tmp = values_objective[asset_tmp] * change
                if value_max < value_tmp or (value_max == value_tmp and asset_tmp == asset_to):  # additional condition can be removed for fees
                    asset_from = asset_tmp
                    value_max = value_tmp

            asset_path = paths[asset_to]
            asset_path.append(asset_from)
            values_tmp[asset_to] = value_max

        values_objective = values_tmp

        continue

    asset_last, roi = max(enumerate(values_objective), key=lambda x: x[1])
    path = [asset_last]
    for i in range(len_path - 1, 0, -1):
        asset_path = paths[asset_last]
        asset_last = asset_path[i]
        path.insert(0, asset_last)

    return path, roi


def plot_trading(path_trade: Sequence[int], seq_ass: Sequence[float], seq_sec: Sequence[float]):
    size = len(seq_ass)
    assert len(seq_sec) == size

    fig, ax = pyplot.subplots()
    ax.plot(range(size), seq_ass, label="asset")
    label_buy = True
    ax.plot(range(size), seq_sec, label="security")
    label_sell = True
    for i in range(len(path_trade) - 1):
        if path_trade[i] == 0 and path_trade[i + 1] == 1:
            if label_buy:
                ax.axvline(i, label="buy", color="red")
                label_buy = False
            else:
                ax.axvline(i, color="red")

        elif path_trade[i] == 1 and path_trade[i + 1] == 0:
            if label_sell:
                ax.axvline(i, label="sell", color="green")
                label_sell = False
            else:
                ax.axvline(i, color="green")
    pyplot.legend()
    pyplot.show()


def simulate(path_trading: Sequence[int], rates: Sequence[Sequence[float]], objective_value: float = 1.) -> float:
    no_rates, =  set(len(x) for x in rates)
    assert no_rates - 1 == len(path_trading)

    asset_current = path_trading[0]
    rates_asset = rates[asset_current]
    amount_asset = objective_value / rates_asset[0]

    for i, asset_target in enumerate(path_trading):
        each_rates = tuple(each_sequence[i] for each_sequence in rates)

        # objective value does not change
        rate_current = each_rates[asset_current]
        rate_target = each_rates[asset_target]

        ratio_conversion = rate_current / rate_target
        amount_asset *= ratio_conversion
        asset_current = asset_target

    rates_asset = rates[asset_current]
    objective_result = amount_asset * rates_asset[-1]
    return objective_result


def simulate_alternative(path_trading: Sequence[int], rates: Sequence[Sequence[float]], objective_value: float = 1.) -> float:
    no_rates, =  set(len(x) for x in rates)
    assert no_rates - 1 == len(path_trading)

    for i, asset_target in enumerate(path_trading):
        rates_asset = rates[asset_target]

        # amount of asset changes
        ratio_rates = rates_asset[i + 1] / rates_asset[i]
        objective_value *= ratio_rates

    return objective_value


def main():
    random.seed(235235)

    size = 10
    no_assets = 2

    # rates = [[1., 1., 1., 1., 1.], [1., 2., 2., 3., 2.]]
    rates = tuple(get_sequence(random.uniform(10., 60.), size) for _ in range(no_assets))

    print("tick    " + "".join(f"{i: 9d}" for i in range(size)))
    print()
    for i, each_rate in enumerate(rates):
        print(f"ass_{i:03d} " + "".join(f"{x:9.2f}" for x in each_rate))

    print()

    path_trade, roi = forward(rates)

    print("get     " + "".join([f"  ass_{x:03d}" for x in path_trade] + [f"{roi:9.2f}"]))
    print()

    # plot_trading(path_trade, seq_ass, seq_sec)

    amount = simulate(path_trade, rates, objective_value=1.)
    print(f"roi simulation 0: {amount:5.5f}.")

    amount = simulate_alternative(path_trade, rates, objective_value=1.)
    print(f"roi simulation 1: {amount:5.5f}.")


if __name__ == "__main__":
    main()
