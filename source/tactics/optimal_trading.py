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


def get_trading_sequence(rates: Sequence[Sequence[float]]) -> Tuple[Sequence[int], float]:
    assert len(rates) == 2
    seq_sec, seq_ass = rates

    size, = set(len(x) for x in rates)

    origin_ass = [0 for _ in range(size-1)]  # [index of source asset]
    origin_sec = [0 for _ in range(size-1)]  # [index of source asset]

    val_ass = [-1. for _ in range(size)]
    val_sec = [-1. for _ in range(size)]

    val_ass[0] = 0.
    val_sec[0] = 1.

    # keep two alternative accumulative value sequences
    # split on each switch, keep alternative, discard worse sequence
    # remember actions, not states

    # todo: generalize to n assets, add fees, is log required?

    for i in range(size - 1):
        # next sec
        change_sec = seq_sec[i+1] / seq_sec[i]

        hold_sec = val_sec[i] * change_sec
        sell_ass = val_ass[i] * change_sec
        if hold_sec < sell_ass or i >= size - 2:
            origin_sec[i] = 1           # set origin of sec i + 1 to from ass
            val_sec[i+1] = sell_ass

        else:
            origin_sec[i] = 0          # set origin of sec i + 1 to from sec
            val_sec[i+1] = hold_sec

        # next ass
        change_ass = seq_ass[i+1] / seq_ass[i]

        hold_ass = val_ass[i] * change_ass
        sell_sec = val_sec[i] * change_ass
        if hold_ass >= sell_sec or i >= size - 2:
            origin_ass[i] = 1          # set origin of ass i + 1 to from ass
            val_ass[i+1] = hold_ass

        else:
            origin_ass[i] = 0           # set origin of ass i + 1 to from sec
            val_ass[i+1] = sell_sec

    origin = origin_sec
    storage_inv = []
    for i in range(size - 2, -1, -1):
        storage_inv.append(origin[i])
        if origin[i] == 0:
            origin = origin_sec
        else:
            origin = origin_ass

    print("val_ass " + "".join(f"{x: 9.2f}" for x in val_ass))
    print("origin  " + "".join(["     init"] + [f"{x: 9d}" for x in origin_ass]))
    print()
    print("val_sec " + "".join(f"{x:9.2f}" for x in val_sec))
    print("origin  " + "".join(["     init"] + [f"{x: 9d}" for x in origin_sec]))
    print()

    assert origin_ass == origin_sec
    return storage_inv[::-1], val_sec[-1]


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


def simulate(path_trading: Sequence[int], rates: Sequence[Sequence[float]], objective_value: float = 1., asset_start: int = 0) -> Tuple[float, int]:
    no_rates, =  set(len(x) for x in rates)
    assert no_rates - 1 == len(path_trading)

    asset_current = asset_start
    rates_asset = rates[asset_current]
    amount_asset = objective_value / rates_asset[0]

    for i, asset_target in enumerate(path_trading):
        each_rates = tuple(each_sequence[i] for each_sequence in rates)

        # objective value does not change
        ratio_conversion = each_rates[asset_current] / each_rates[asset_target]
        amount_asset *= ratio_conversion
        asset_current = asset_target

    rates_asset = rates[asset_current]
    return amount_asset * rates_asset[-1], asset_current


def main():
    random.seed(235235)

    size = 3
    seq_ass = get_sequence(53.5, size)
    seq_sec = get_sequence(12.2, size)
    rates = seq_sec, seq_ass

    print("tick    " + "".join(f"{i: 9d}" for i in range(size)))
    print()
    print("seq_ass " + "".join(f"{x:9.2f}" for x in seq_ass))
    print("seq_sec " + "".join(f"{x:9.2f}" for x in seq_sec))
    print()

    path_trade, final_value = get_trading_sequence(rates)

    print("path    " + "".join(f"{x: 9d}" for x in path_trade))
    print()

    print(f"final value: {final_value:5.5f}")

    # plot_trading(path_trade, seq_ass, seq_sec)

    amount, asset = simulate(path_trade, rates, objective_value=seq_sec[0], asset_start=0)
    print(f"result end of simulation 0 {amount:5.5f} of asset {asset:d}.")


if __name__ == "__main__":
    main()
