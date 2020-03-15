# https://www.dropbox.com/s/ed5hm4rd0b8bz18/optimal.pdf?dl=0
import math
import random
from pprint import pprint
from typing import Sequence, Tuple, Callable, Generator, Iterable, Iterator

# from matplotlib import pyplot
from matplotlib import pyplot

from source.data.merge_csv import merge_generator
from source.tools.timer import Timer


def get_random_sequence(start_value: float, length: int) -> Iterator[float]:
    s = [-1. for _ in range(length)]
    s[0] = start_value

    for i in range(1, length):
        value_last = s[i-1]
        r = random.uniform(-.1, .1) * value_last
        s[i] = value_last + r

    return (v for v in s)


def forward(
        rates: Sequence[Sequence[float]],
        fees: Callable[[float, int, int], float] = lambda _amount_from, _asset_from, _asset_to: 0.) -> Tuple[Sequence[int], float]:
    print(f"forward pass...")
    no_assets = len(rates)
    len_sequence, = set(len(x) for x in rates)
    len_path = len_sequence - 1

    values_objective = [1. for _ in rates]
    matrix = tuple([] for _ in rates)

    iterations_total = len_path * no_assets * no_assets
    iterations_done = 0

    for t in range(len_path):
        rates_now = tuple(x[t] for x in rates)
        rates_next = tuple(x[t + 1] for x in rates)
        values_tmp = values_objective[:]
        for asset_to, (rate_to_now, rate_to_next) in enumerate(zip(rates_now, rates_next)):
            change = rate_to_next / rate_to_now

            asset_from = -1
            value_max = -1.
            for asset_tmp in range(no_assets):
                if Timer.time_passed(2000):
                    print(f"matrix built by {iterations_done * 100. / iterations_total:5.2f}%...")
                iterations_done += 1

                value_tmp = values_objective[asset_tmp] * change
                value_tmp -= fees(value_tmp, asset_tmp, asset_to)
                if value_tmp < 0.:
                    continue

                if value_max < value_tmp or (value_max == value_tmp and asset_tmp == asset_to):  # additional condition can be removed for fees
                    asset_from = asset_tmp
                    value_max = value_tmp

            asset_row = matrix[asset_to]
            asset_row.append(asset_from)
            values_tmp[asset_to] = value_max

        values_objective = values_tmp

        continue

    print(f"backwards pass...")
    asset_last, roi = max(enumerate(values_objective), key=lambda x: x[1])
    path = [asset_last]
    for i in range(len_path - 1, 0, -1):
        asset_row = matrix[asset_last]
        asset_last = asset_row[i]
        path.insert(0, asset_last)
        if Timer.time_passed(2000):
            print(f"finished {(len_path - 1 - i) * 100. / (len_path - 1):5.2f}% of backwards pass...")

    return path, roi


def generate_positive_change() -> Generator[float, float, None]:
    rate_last = yield
    each_rate = yield
    while True:

        if each_rate < 0.:
            change = -1.

        elif 0. >= rate_last:
            change = -1.
            rate_last = each_rate

        else:
            change = each_rate / rate_last
            rate_last = each_rate

        each_rate = yield change


def generate_changes(generator_rates: Iterator[Sequence[float]]) -> Generator[Sequence[float], None, None]:
    rates_now = next(generator_rates)
    no_rates = len(rates_now)

    generators_change = tuple(generate_positive_change() for _ in rates_now)
    for each_change_gen, first_rate in zip(generators_change, rates_now):
        next(each_change_gen)               # initialize
        each_change_gen.send(first_rate)    # send first rate

    for rates_now in generator_rates:
        assert len(rates_now) == no_rates
        yield tuple(x.send(r) for x, r in zip(generators_change, rates_now))


def forward_negative(
        rates: Iterator[Sequence[float]],
        fees: Callable[[float, int, int], float] = lambda _amount_from, _asset_from, _asset_to: 0.) -> Tuple[Sequence[int], float]:

    matrix_change = generate_changes(rates)

    return [], -1.


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


def simulate(
        path_trading: Sequence[int],
        rates: Sequence[Sequence[float]],
        objective_value: float = 1.,
        fees: Callable[[float, int, int], float] = lambda _amount_from, _asset_from, _asset_to: 0.) -> Tuple[Sequence[float], float]:

    # TODO: return by step return of interest
    print(f"simulating trading strategy...")
    no_rates, =  set(len(x) for x in rates)
    assert no_rates - 1 == len(path_trading)

    asset_current = path_trading[0]
    rates_asset = rates[asset_current]
    amount_asset = objective_value / rates_asset[0]

    ratio_history = []
    for i, asset_target in enumerate(path_trading):
        # subtract fees prior to conversion
        amount_asset -= fees(amount_asset, asset_current, asset_target)
        if amount_asset < 0.:
            raise ValueError(f"Trade at time index {i:d} not possible due to fees!")

        each_rates = tuple(each_sequence[i] for each_sequence in rates)

        # objective value does not change
        rate_current = each_rates[asset_current]
        rate_target = each_rates[asset_target]

        ratio_conversion = rate_current / rate_target
        amount_asset *= ratio_conversion

        ratio_history.append(rates[asset_target][i+1] / rate_target)

        asset_current = asset_target

    rates_asset = rates[asset_current]
    objective_result = amount_asset * rates_asset[-1]
    return ratio_history, objective_result


def simulate_alternative(
        path_trading: Sequence[int],
        rates: Sequence[Sequence[float]],
        objective_value: float = 1.,
        fees: Callable[[float, int, int], float] = lambda _amount_from, _asset_from, _asset_to: 0.) -> Tuple[Sequence[float], float]:

    # TODO: return by step return of interest
    print(f"simulating trading strategy...")
    no_rates, =  set(len(x) for x in rates)
    assert no_rates - 1 == len(path_trading)

    ratio_history = []
    for i, asset_target in enumerate(path_trading):
        # subtract fees before conversion
        if i >= 1:
            asset_last = path_trading[i - 1]
            objective_value -= fees(objective_value, asset_last, asset_target)
            if objective_value < 0.:
                raise ValueError(f"Trade at time index {i:d} not possible due to fees!")

        rates_asset = rates[asset_target]

        # amount of asset changes
        ratio_rates = rates_asset[i + 1] / rates_asset[i]
        objective_value *= ratio_rates
        ratio_history.append(ratio_rates)

    return ratio_history, objective_value


def fees_debug(amount_from: float, asset_from: int, asset_to: int) -> float:
    if asset_from == asset_to:
        return 0.
    return amount_from * .0


def get_random_rates(size: int = 20, no_assets: int = 10) -> Iterator[Tuple[int, Sequence[float]]]:
    random.seed(235235)

    sequences = tuple(get_random_sequence(random.uniform(10., 60.), size) for _ in range(no_assets))
    return ((i, x) for i, x in enumerate(zip(*sequences)))


def get_debug_rates() -> Iterator[Tuple[int, Sequence[float]]]:
    rate_a = 59.69, -1.,   -1., 65.73, 59.37
    rate_b = 30.06, -1., 27.06, 27.09, 29.13

    return ((i, r) for i, r in enumerate(zip(rate_a, rate_b)))


def get_decreasing_rates(size: int = 20, no_assets: int = 10) -> Iterator[Tuple[int, Sequence[float]]]:
    random.seed(235235)

    rates = tuple([] for _ in range(no_assets))

    for each_rate in rates:
        each_rate.append(random.uniform(10., 60.))

    for each_rate in rates:
        for _ in range(size):
            each_rate.append(each_rate[-1] * .9)

    return ((i, x) for i, x in enumerate(rates))


def get_crypto_rates(interval_minutes: int = 1) -> Iterator[Tuple[int, Sequence[float]]]:
    pairs = (
        ("bcc", "eth"), ("bnb", "eth"), ("dash", "eth"), ("icx", "eth"),
        ("iota", "eth"), ("ltc", "eth"), ("nano", "eth"), ("poa", "eth"),
        ("qtum", "eth"), ("theta", "eth"), ("tusd", "eth"), ("xmr", "eth")
    )
    generator = merge_generator(pairs, interval_minutes=interval_minutes, header=("close_time", "close", ))
    return ((snapshot[0][0], tuple(each_data[1] for each_data in snapshot)) for snapshot in generator)


def main():
    # timestamps, rates = zip(*get_decreasing_rates(no_assets=3, size=5))
    # timestamps, rates = zip(*get_random_rates(no_assets=3, size=5))
    timestamps, rates = zip(*get_debug_rates())
    # timestamps, rates = zip(*get_crypto_rates(interval_minutes=1))

    rates = list(zip(*rates))
    rates = [list(x) for x in rates]
    #rates[0][1] = -1.
    #rates[1][1] = -1.
    #rates[1][2] = -1.
    #rates[2][1] = -1.
    # rates.append([1. for _ in range(5)])

    size, = set(len(x) for x in rates)

    path_trade, roi = forward(rates, fees=fees_debug)

    if size < 30:
        print("tick    " + "".join(f"{i: 9d}" for i in range(size)))
        print()
        for i, each_rate in enumerate(rates):
            print(f"ass_{i:03d} " + "".join(f"{x:9.2f}" for x in each_rate))
        print()
        print("get     " + "".join([f"  ass_{x:03d}" for x in path_trade] + [f" {roi:8.2f} times investment returned"]))

    exit()
    history_a, amount_a = simulate(path_trade, rates, objective_value=1., fees=fees_debug)
    # history_b, amount_b = simulate_alternative(path_trade, rates, objective_value=1., fees=fees_debug)

    if size < 30:
        print("ratio   " + "".join([f"{1.:9.2f}"] + [f"{x:9.2f}" for x in history_a]))
        print()
        print()
        print(f"history simulation 0: {str(history_a):s}")
        # print(f"history simulation 0: {str(history_b):s}")

    print(f"roi simulation 0: {amount_a:5.5f}.")
    # print(f"roi simulation 1: {amount_b:5.5f}.")
    print()

    with open("../../data/examples/test.csv", mode="a") as file:
        header = "timestamp", "rates", "action", "intensity"
        file.write("\t".join(header) + "\n")
        for i in range(len(path_trade)):
            ts = timestamps[i]
            r = tuple(each_rates[i] for each_rates in rates)
            a = path_trade[i]
            i = history_a[i]
            values = f"{ts:d}", f"{', '.join(f'{x:.8f}' for x in r):s}", f"ass_{a:03d}", f"{i:.8f}"
            file.write("\t".join(values) + "\n")
            if Timer.time_passed(2000):
                print(f"finished {i * 100. / len(path_trade):5.2f}% of example generation...")


if __name__ == "__main__":
    main()
