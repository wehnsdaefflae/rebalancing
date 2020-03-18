# https://www.dropbox.com/s/ed5hm4rd0b8bz18/optimal.pdf?dl=0
import random
from typing import Sequence, Tuple, Callable, Generator, Iterator, List, Iterable, Optional, Union

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


def forward_deprecated(
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


def generate_change() -> Generator[float, float, None]:
    rate_last = yield
    each_rate = yield

    # only returns -1. when
    #   all last values have been negative
    #   the new value is negative
    while True:

        if each_rate < 0. or rate_last < 0.:
            change = -1.

        else:
            change = each_rate / rate_last

        rate_last = each_rate
        each_rate = yield change


def generate_multiple_changes(generator_rates: Iterator[Sequence[float]]) -> Generator[Sequence[float], None, None]:
    print(f"generating changes...")
    rates_now = next(generator_rates)
    no_rates = len(rates_now)

    generators_change = tuple(generate_change() for _ in rates_now)
    # generators_change = tuple(generate_positive_change() for _ in rates_now)
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
        changes: Iterator[Sequence[float]], fees: float,
        bound: float = 0.) -> Generator[Tuple[Sequence[int], Tuple[int, float]], None, None]:

    assert 1. >= fees >= 0.
    values_objective = [1. for _ in range(no_assets)]

    changes_asset_prev = [-1. for _ in range(no_assets)]
    for t, changes_asset in enumerate(changes):
        assert len(changes_asset) == no_assets

        asset_sources = list(range(no_assets))
        values_tmp = values_objective[:]
        for asset_to, each_change in enumerate(changes_asset):
            if each_change < 0. and changes_asset_prev[asset_to] >= 0. and False:
                # reduce changes to be source for next iteration
                best_predecessor = asset_to
                value_max = .1

            else:
                best_predecessor, value_max = max(
                    (
                        (asset_from, each_interest * (1. - float(asset_from != asset_to and 0 < t) * fees) * each_change)
                        for asset_from, each_interest in enumerate(values_objective)
                    ), key=lambda x: x[1]
                )

            asset_sources[asset_to], values_tmp[asset_to] = best_predecessor, value_max

            if each_change < 0.:
                values_tmp[asset_to] = 0.

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

        changes_asset_prev = changes_asset


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


def simulate_deprecated(
        path_trading: Sequence[int],
        rates: Sequence[Sequence[float]],
        objective_value: float = 1.,
        fees: Callable[[float, int, int], float] = lambda _amount_from, _asset_from, _asset_to: 0.) -> Tuple[Sequence[float], float]:

    # TODO: convert to generator that takes iterator
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


def simulate_alternative_deprecated(
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

        rate_current = current_rates[asset_target]
        rate_next = next_rates[asset_target]

        # amount of asset changes
        if rate_current >= 0. and rate_next >= 0.:
            ratio_rates = rate_next / rate_current
            objective_value *= ratio_rates
        else:
            ratio_rates = -1.

        ratio_history.append(ratio_rates)

    return ratio_history, objective_value


def get_random_rates(size: int = 20, no_assets: int = 10, gaps: float = 0.) -> Iterator[Tuple[int, Sequence[float]]]:
    random.seed(25235)

    sequences = list(list(get_random_sequence(random.uniform(10., 60.), size)) for _ in range(no_assets))
    for each_list in sequences:
        for i in range(len(each_list)):
            if random.random() < gaps:
                each_list[i] = -1.

    return ((i, x) for i, x in enumerate(zip(*sequences)))


def get_debug_rates() -> Iterator[Tuple[int, Sequence[float]]]:
    rate_a = 59.69, 65.07, 61.44, 65.73, 59.37
    rate_b =   -1.,   -1.,   -1.,   -1.,   -1.

    #rate_a = 59.69, 65.07, 61.44, 65.73, 59.37
    #rate_b = 30.06, 29.96, 27.06, 27.09, 29.13

    #rate_a = 60.98, -1.,   -1., 65.73, 59.37
    #rate_b = -1., -1., 27.06, 29.2, 29.13

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


def get_crypto_debug_rates() -> Iterator[Tuple[int, Sequence[float]]]:
    generator = merge_generator(
        (
            ("bcc", "eth"), ("bnb", "eth"), ("tusd", "eth"),
        ),
        interval_minutes=1,
        header=("close_time", "close",),
        timestamp_range=(1527000000000, 1527000600000),
    )
    return (
        (
            snapshot[0][0],
            tuple(each_data[1] for each_data in snapshot)
        )
        for snapshot in generator)


def get_crypto_rates(
        pairs: Sequence[Tuple[str, str]],
        stats: Sequence[str],
        timestamp_range: Optional[Tuple[int, int]] = None,
        interval_minutes: int = 1,
        directory_data: str = "../../data/") -> Iterator[Tuple[int, Tuple[Sequence[Union[int, float]], ...]]]:

    # returns (timestamp, tuple of (tuple of int and float values))
    generator = merge_generator(
        pairs=pairs,
        timestamp_range=timestamp_range,
        interval_minutes=interval_minutes,
        directory_data=directory_data,
        header=("close_time",) + tuple(stats))

    generator_modified = (
        (
            int(snapshot[0][0]),                                # timestamp
            tuple(each_asset[1:] for each_asset in snapshot)      # rest of data without timestamp
        )
        for snapshot in generator)

    return generator_modified


def simulate(rates: Iterable[Sequence[float]], path: Sequence[int], fees: float) -> Generator[float, None, None]:
    len_path = len(path)

    amount_asset = -1.
    asset_current = -1
    last_rate = -1.

    amount_last = -1.

    for i, rates_current in enumerate(rates):
        if i < len_path:
            asset_next = path[i]

            # first iteration, initialize stuff
            if i == 0:
                asset_current = asset_next
                amount_asset = 1. / rates_current[asset_current]

            # if hold
            rate_this = rates_current[asset_current]
            if asset_next == asset_current:
                if rate_this < 0.:
                    amount = -1. if last_rate < 0. else amount_asset * last_rate
                    yield 0. if amount < 0. or amount_last < 0. else amount - amount_last
                    amount_last = amount
                else:
                    amount = amount_asset * rate_this
                    yield 0. if amount < 0. or amount_last < 0. else amount - amount_last
                    amount_last = amount

            # if switch
            else:
                amount = amount_asset * rate_this
                amount = amount * (1. - fees)
                yield 0. if amount < 0. or amount_last < 0. else amount - amount_last
                amount_last = amount

                rate_other = rates_current[asset_next]
                if rate_other >= 0.:
                    amount_asset = amount / rate_other
                    rate_this = rate_other
                    asset_current = asset_next

                else:
                    # should actually never switch into unknown asset
                    print(f"switching into unknown asset at rate {i:d}! why?!")

            last_rate = rate_this if rate_this >= 0. else last_rate

        elif i == len_path:
            asset_current = path[-1]
            rate_this = rates_current[asset_current]
            if rate_this < 0.:
                amount = amount_asset * last_rate
                yield 0. if amount < 0. or amount_last < 0. else amount - amount_last
                amount_last = amount

            else:
                amount = amount_asset * rate_this
                yield 0. if amount < 0. or amount_last < 0. else amount - amount_last
                amount_last = amount

            break

        elif len_path < i:
            break


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

    # new
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
    print("change matrix new")
    print("\n".join(["    ".join(["        "] + [f"{v:5.2f}" for v in x]) for x in zip(*[y for y in matrix_change_fix])]))
    print()

    matrix = generate_matrix(no_assets, matrix_change_fix, .01, bound=100)
    matrix_fix = tuple(matrix)

    print("asset source matrix new")
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


if __name__ == "__main__":
    compare()
