# https://www.dropbox.com/s/ed5hm4rd0b8bz18/optimal.pdf?dl=0
import random
from typing import Sequence, Tuple, Callable, Generator, Iterator, List, Iterable

# from matplotlib import pyplot
from matplotlib import pyplot

from source.data.merge_csv import merge_generator
from source.tools.timer import Timer


def fees_debug(amount_from: float, asset_from: int, asset_to: int) -> float:
    if asset_from == asset_to:
        return 0.
    return amount_from * .02


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
    value_matrix = [[1. for _ in rates]]

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

        value_matrix.append(values_tmp[:])

        values_objective = values_tmp

        continue

    print("roi matrix old")
    print("\n".join(["  ".join(f"{v:7.4f}" for v in x) for x in zip(*value_matrix)]))
    print()

    print("source matrix old")
    print("\n".join(["  ".join(f"{v: 7d}" for v in x) for x in matrix]))
    print()

    print(f"backwards pass...")
    asset_last, roi_final = max(enumerate(values_objective), key=lambda x: x[1])
    path = [asset_last]
    for i in range(len_path - 1, 0, -1):
        asset_row = matrix[asset_last]
        asset_last = asset_row[i]
        path.insert(0, asset_last)
        if Timer.time_passed(2000):
            print(f"finished {(len_path - 1 - i) * 100. / (len_path - 1):5.2f}% of backwards pass...")

    return path, roi_final


def generate_positive_change() -> Generator[float, float, None]:
    rate_last = yield
    each_rate = yield

    # only returns -1. when
    #   all last values have been negative
    #   the new value is negative
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
    print(f"generating changes...")
    rates_now = next(generator_rates)
    no_rates = len(rates_now)

    generators_change = tuple(generate_positive_change() for _ in rates_now)
    for each_change_gen, first_rate in zip(generators_change, rates_now):
        next(each_change_gen)               # initialize
        each_change_gen.send(first_rate)    # send first rate

    for i, rates_now in enumerate(generator_rates):
        assert len(rates_now) == no_rates
        yield tuple(x.send(r) for x, r in zip(generators_change, rates_now))
        if Timer.time_passed(2000):
            print(f"finished determining {i:d} rate changes...")


def generate_matrix(
        no_assets: int,
        changes: Iterator[Sequence[float]], fees: Callable[[float, int, int], float],
        bound: float = 0.) -> Generator[Tuple[Sequence[int], Tuple[int, float]], None, None]:

    values_objective = [1. for _ in range(no_assets)]

    for t, changes_asset in enumerate(changes):
        assert len(changes_asset) == no_assets

        asset_sources = list(range(no_assets))
        values_tmp = values_objective[:]
        for asset_to, each_change in enumerate(changes_asset):
            tmp = tuple(
                (asset_from, (each_interest - fees(each_interest, asset_from, asset_to)) * (1. if each_change < 0. else each_change))
                for asset_from, each_interest in enumerate(values_objective)
            )
            a_s, v_t = max(
                tmp, key=lambda x: x[1]
            )
            asset_sources[asset_to], values_tmp[asset_to] = a_s, v_t

        for i, v in enumerate(values_tmp):
            values_objective[i] = v

        if 0. < bound and any(x >= bound for x in values_objective):
            for i, v in enumerate(values_objective):
                values_objective[i] = v / bound

        yield tuple(asset_sources), max(enumerate(values_tmp), key=lambda x: x[1])
        if Timer.time_passed(2000):
            print(f"finished {t:d} time steps in roi matrix...")


def make_source_matrix(
        no_assets: int,
        rates: Iterator[Sequence[float]],
        fees: Callable[[float, int, int], float] = fees_debug,
        ) -> Generator[Tuple[Sequence[int], Tuple[int, float]], None, None]:

    """
    rates_full = [x for x in rates]
    rts = list(zip(*rates_full))
    print("rates")
    print("\n".join(["  ".join(f"{v:7.4f}" for v in x) for x in rts]))
    print()
    rates = (x for x in rates_full)
    """

    matrix_change = generate_changes(rates)
    """
    matrix_full = [x for x in matrix_change]
    cng = list(zip(*matrix_full))
    print("changes")
    print("\n".join(["  ".join(f"{v:7.4f}" for v in x) for x in ((-1.,) + x for x in cng)]))
    print()
    matrix_change = (x for x in matrix_full)
    """

    matrix_source = generate_matrix(no_assets, matrix_change, fees, bound=100)

    """
    print("converting matrix to list...")
    matrix_full = list(x for x in matrix_source)
    acc = list(zip(*[x[0] for x in matrix_full]))
    print("source matrix new")
    print("\n".join(["  ".join(f"ass_{v:8d}" for v in x) for x in acc]))
    print()
    #"""

    return matrix_source


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
    print(f"backwards pass...")
    len_path = len(matrix)

    snapshot, (asset_last, _) = matrix[-1]
    path = [asset_last]
    i = len_path - 2
    while i >= 0:
        asset_last = snapshot[asset_last]
        path.insert(0, asset_last)
        snapshot, (_, _) = matrix[i]
        i -= 1

    return path


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
    no_rates = len(rates)
    assert no_rates - 1 == len(path_trading)

    rates_current = rates[0]
    asset_current = path_trading[0]
    amount_asset = objective_value / rates_current[asset_current]

    ratio_history = []
    for i, asset_target in enumerate(path_trading):
        # subtract fees prior to conversion
        amount_asset -= fees(amount_asset, asset_current, asset_target)
        if amount_asset < 0.:
            raise ValueError(f"Trade at time index {i:d} not possible due to fees!")

        each_rates = rates[i]

        # objective value does not change
        rate_current = each_rates[asset_current]
        rate_target = each_rates[asset_target]

        ratio_conversion = rate_current / rate_target
        amount_asset *= ratio_conversion

        ratio_history.append(rates[i+1][asset_target] / rate_target)

        asset_current = asset_target

    rates_last = rates[-1]
    objective_result = amount_asset * rates_last[asset_current]
    return ratio_history, objective_result


def simulate_alternative(
        path_trading: Sequence[int],
        rates: Sequence[Sequence[float]],
        objective_value: float = 1.,
        fees: Callable[[float, int, int], float] = lambda _amount_from, _asset_from, _asset_to: 0.) -> Tuple[Sequence[float], float]:

    # TODO: return by step return of interest
    print(f"simulating trading strategy...")
    no_rates = len(rates)
    assert no_rates - 1 == len(path_trading)

    ratio_history = []
    for i, asset_target in enumerate(path_trading):
        # subtract fees before conversion
        if i >= 1:
            asset_last = path_trading[i - 1]
            objective_value -= fees(objective_value, asset_last, asset_target)
            if objective_value < 0.:
                raise ValueError(f"Trade at time index {i:d} not possible due to fees!")

        current_rates = rates[i]
        next_rates = rates[i+1]

        # amount of asset changes
        ratio_rates = next_rates[asset_target] / current_rates[asset_target]
        objective_value *= ratio_rates
        ratio_history.append(ratio_rates)

    return ratio_history, objective_value


def get_random_rates(size: int = 20, no_assets: int = 10) -> Iterator[Tuple[int, Sequence[float]]]:
    random.seed(235235)

    sequences = tuple(get_random_sequence(random.uniform(10., 60.), size) for _ in range(no_assets))
    return ((i, x) for i, x in enumerate(zip(*sequences)))


def get_debug_rates() -> Iterator[Tuple[int, Sequence[float]]]:
    rate_a = 59.69, 65.07, 61.44, 65.73, 59.37
    rate_b = 30.06, 29.96, 27.06, 27.09, 29.13
    #rate_a = -1., -1.,   -1., 65.73, 59.37
    #rate_b = -1., -1., 27.06, 27.09, 29.13

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
    return (
        (
            snapshot[0][0],
            tuple(each_data[1] for each_data in snapshot)
        )
        for snapshot in generator)


def split_time_and_data(input_data: Tuple[int, Sequence[float]], timestamp_storage: List[int], rate_storage: List[Sequence[float]]) -> Sequence[float]:
    timestamp, data = input_data
    timestamp_storage.append(timestamp)
    rate_storage.append(data)

    return data


def compare():
    no_assets = 5
    get_rates = lambda: get_random_rates(size=10, no_assets=no_assets)
    # get_rates = get_debug_rates

    # new
    timestamps = []
    rates = []
    generate_rates = (split_time_and_data(x, timestamps, rates) for x in get_rates())

    matrix = make_source_matrix(no_assets, generate_rates, fees=fees_debug)
    print("fixing matrix...")
    matrix_fix = tuple(matrix)

    print("source matrix new")
    print("\n".join(["  ".join(f"{v: 7d}" for v in x) for x in zip(*[y[0] for y in matrix_fix])]))
    print()

    path_new = make_path_from_sourcematrix(matrix_fix)

    sim_a_new, amount_a_new = simulate(path_new, rates, objective_value=1., fees=fees_debug)
    sim_b_new, amount_b_new = simulate_alternative(path_new, rates, objective_value=1., fees=fees_debug)

    print("tick    " + "".join(f"{i: 9d}" for i in range(len(rates))))
    print()
    for i, each_rate in enumerate(zip(*rates)):
        print(f"ass_{i:03d} " + "".join(f"{x:9.2f}" for x in each_rate))
    print()
    print("path new" + "".join(f"  ass_{x:03d}" for x in path_new))
    print("sim. a n" + "".join([f"{1.:9.2f}"] + [f"{x:9.2f}" for x in sim_a_new]) + f"  roi: {amount_a_new:5.5f}.")
    print("sim. b n" + "".join([f"{1.:9.2f}"] + [f"{x:9.2f}" for x in sim_b_new]) + f"  roi: {amount_b_new:5.5f}.")
    print()


if __name__ == "__main__":
    compare()
