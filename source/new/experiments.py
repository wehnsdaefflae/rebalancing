import datetime
import random
from typing import Iterable, Sequence, Tuple, Generator, Union, Collection, Callable, Type, Any, Optional

from matplotlib import pyplot, dates
from matplotlib.ticker import MaxNLocator

from source.new.binance_examples import STATS, get_pairs_from_filesystem, binance_matrix
from source.new.learning import Classification, MultivariateRegression, PolynomialClassification, RecurrentPolynomialClassification, Approximation, smear, \
    MultivariatePolynomialRegression, MultivariatePolynomialRecurrentRegression
from source.new.snapshot_generation import merge_generator
from source.tools.timer import Timer


def types_binance(column: str) -> Type:
    if column == "timestamp":
        return int
    if column == "target":
        return str
    if "open_time" in column:
        return int
    if "close_time" in column:
        return int
    if "open" in column:
        return float
    if "high" in column:
        return float
    if "low" in column:
        return float
    if "close" in column:
        return float
    if "volume" in column:
        return float
    if "quote_asset_volume" in column:
        return float
    if "number_of_trades" in column:
        return int
    if "taker_buy_base_asset_volume" in column:
        return float
    if "taker_buy_quote_asset_volume" in column:
        return float
    if "ignore" in column:
        return float


def binance_columns(assets: Collection[str], columns_available: Collection[str]) -> Collection[str]:
    return ("timestamp",) + tuple(
        f"{each_asset.upper():s}_{each_stat.lower():s}"
        for each_asset in assets
        for each_stat in columns_available
    )


SNAPSHOT_BINANCE = Tuple[Union[float, int, str], ...]  # timestamp, rates, action


def iterate_snapshots(path_file: str, columns: Collection[str], get_column_type: Callable[[str], Type]) -> Generator[SNAPSHOT_BINANCE, None, None]:
    with open(path_file, mode="r") as file:
        line = next(file)
        stripped = line.strip()
        header = stripped.split("\t")
        indices = tuple(header.index(c) for c in columns)
        types = tuple(get_column_type(c) for c in columns)
        for line in file:
            stripped = line.strip()
            split = stripped.split("\t")
            y = tuple(t(split[i]) for t, i in zip(types, indices))
            yield y


def extract_changes(generator: Iterable[Sequence[Any]], indices_float: Collection[int]) -> Iterable[Sequence[Any]]:
    len_each = -1
    last = None
    for each in generator:
        if last is None:
            last = each
            len_each = len(each)
            continue
        assert len(each) == len_each
        y = tuple(each[i] if i not in indices_float else 0. if 0. >= float(last[i]) or 0. >= float(each[i]) else float(each[i] / last[i]) for i in range(len_each))
        yield y


INPUT_VALUES = Sequence[float]
TARGET_CLASS = int
EXAMPLE = Tuple[INPUT_VALUES, TARGET_CLASS]

INFO_INVESTMENT = Tuple[int, int, Sequence[Tuple[int, float, float, float, float]]]        # timestamp, target, (output, error, knowledgeability, decidedness, roi)


def simulate_investment(
        classifications: Sequence[Classification],
        examples: Iterable[SNAPSHOT_BINANCE],
        names_assets: Sequence[str],
        fees: float,
        stop_training_at: int = -1) -> Generator[INFO_INVESTMENT, None, None]:

    amount_asset = [1. for _ in classifications]
    index_asset = [-1 for _ in classifications]

    indices_rates = range(1, len(names_assets) + 1)
    # examples_changes = extract_changes(examples, indices_rates)

    snapshot_last = None

    for i, snapshot in enumerate(examples):
        timestamp = snapshot[0]

        rate_changes = [1. for _ in indices_rates] if snapshot_last is None \
            else [float("inf") if 0. >= snapshot_last[i] else snapshot[i] / snapshot_last[i] for i in indices_rates]

        target = snapshot[-1]
        index_target = names_assets.index(target)

        rates = [snapshot[i] for i in indices_rates]

        classification_stats = []
        for j, each_classification in enumerate(classifications):
            output_class = each_classification.output(rate_changes)

            rate_hold = rates[index_asset[j]]
            rate_switch = rates[output_class]

            if index_asset[j] < 0:
                index_asset[j] = output_class
                amount_asset[j] = (1. - fees) * amount_asset[j] / rate_switch

            elif index_asset[j] != output_class:
                amount_asset[j] = amount_asset[j] * (1. - fees) * rate_hold / rate_switch
                index_asset[j] = index_target

            details = each_classification.get_details_last_output()
            output_raw = details["raw output"]
            error = MultivariateRegression.error_distance(output_raw, tuple(float(i == index_target) for i in output_raw))

            roi = amount_asset[j] * rates[index_asset[j]]

            if -1 >= stop_training_at or timestamp < stop_training_at:
                each_classification.fit(rate_changes, index_target, i + 1)

            stats = names_assets[output_class], error, details["knowledgeability"], details["decidedness"], roi
            classification_stats.append(stats)

        y = timestamp, target, classification_stats
        yield y

    # print()


INFO_PREDICTION = Tuple[int, Sequence[float], Sequence[Tuple[Sequence[float], float]]]        # timestamp, output, target, error


def predict_rate(approximations: Sequence[Approximation[Sequence[float]]], examples: Iterable[SNAPSHOT_BINANCE], names_assets: Sequence[str]) -> Generator[INFO_PREDICTION, None, None]:
    last_snapshot = None

    indices_rates = range(1, len(names_assets) + 1)
    example_changes = extract_changes(examples, indices_rates)

    for i, snapshot_changes in enumerate(example_changes):
        if last_snapshot is None:
            last_snapshot = snapshot_changes
            continue
        input_values = tuple(last_snapshot[i] for i in indices_rates)
        target_values = tuple(snapshot_changes[i] for i in indices_rates)

        approximator_stats = []
        for j, each_approximation in enumerate(approximations):
            output_values = each_approximation.output(input_values)
            error = MultivariateRegression.error_distance(output_values, target_values)
            # error = float((1. < output_values[0]) != (1. < target_values[0]))

            each_approximation.fit(input_values, target_values, i)
            approximator_stats.append((output_values, error))

        timestamp = last_snapshot[0]

        y = timestamp, target_values, approximator_stats
        yield y


def learn_timeseries(time_range: Optional[Tuple[int, int]] = None):
    pairs = ("AGI", "ETH"),
    stats = "close",

    names_assets = tuple(f"{each_pair[0].upper():s}-{each_pair[1].upper()}" for each_pair in pairs)
    columns = binance_columns(names_assets, stats)

    examples = iterate_snapshots("../../data/examples/binance_examples_small.csv", columns, types_binance)

    # learners = MultivariatePolynomialRegression(1, 3, 1), MultivariatePolynomialRecurrentRegression(1, 3, 1)
    learners = MultivariatePolynomialRegression(1, 4, 1), MultivariatePolynomialRecurrentRegression(1, 4, 1)

    approximations = predict_rate(learners, examples, names_assets)

    fig, ax = pyplot.subplots()
    max_size = 20
    times = []
    error = [[] for _ in learners]
    error_total = [-1. for _ in learners]

    # pyplot.ion()
    for i, (timestamp, target_values, approximator_stats) in enumerate(approximations):
        if Timer.time_passed(1000):
            print(f"finished reading {i:d} examples...")
            print(f"{timestamp:d}: {str(target_values):s}")

            times.append(i)
            for j, (each_output, each_error) in enumerate(approximator_stats):
                error_total[j] = each_error if len(times) < 2 else smear(error_total[j], each_error, 10)
                error[j].append(error_total[j])

            if max_size < len(times):
                times = times[-max_size:]
                for j, each_error in enumerate(error):
                    error[j] = each_error[-max_size:]

            ax.clear()
            for j, each_error in enumerate(error):
                ax.plot(times, each_error, label=f"{learners[j].__class__.__name__:s} {j:d}", alpha=.5)

            ax.set_ylim([0, min(max(each_error) for each_error in error) * 1.2])

            pyplot.legend()
            pyplot.pause(.05)


def learn_investment(stop_training_at: int = -1):
    pairs = get_pairs_from_filesystem()[:10]

    stats = STATS
    stats = "close",

    names_assets = tuple(f"{each_pair[0].upper():s}-{each_pair[1].upper()}" for each_pair in pairs)
    columns = tuple(binance_columns(names_assets, stats)) + ("target", )

    # all assets polynomial for all assets is too much
    classifications = PolynomialClassification(len(pairs), 2, len(pairs)), RecurrentPolynomialClassification(len(pairs), 2, len(pairs))

    examples = iterate_snapshots("../../data/examples/binance_examples_small.csv", columns, types_binance)
    simulation = simulate_investment(classifications, examples, names_assets, .01, stop_training_at=stop_training_at)

    fig, ax = pyplot.subplots()
    max_size = 20
    times = []
    error = [[] for _ in classifications]
    error_total = [-1. for _ in classifications]
    roi = [[] for _ in classifications]

    for i, (timestamp, target_asset, classification_stats) in enumerate(simulation):
        if Timer.time_passed(1000):
            print(f"finished reading {i:d} examples...")

            print(f"{timestamp:d}: {target_asset:s}")
            if timestamp >= stop_training_at:
                print("stopped training")
            else:
                print(f"stopping training in {stop_training_at - timestamp:d}")

            times.append(i)
            for j, (each_output, each_error, each_k, each_d, each_roi) in enumerate(classification_stats):
                error_total[j] = each_error if len(times) < 2 else smear(error_total[j], each_error, 10)
                error[j].append(error_total[j])
                roi[j].append(each_roi)

            if max_size < len(times):
                times = times[-max_size:]
                for j in range(len(classifications)):
                    error[j] = error[j][-max_size:]
                    roi[j] = roi[j][-max_size:]

            ax.clear()
            for j in range(len(classifications)):
                # ax.plot(times, error[j], label=f"error {classifications[j].__class__.__name__:s}", alpha=.5)
                ax.plot(times, roi[j], label=f"roi {classifications[j].__class__.__name__:s}", alpha=.5)

            # ax.set_ylim([0, min(max(each_error) for each_error in error) * 1.2])
            ax.set_ylim([max(min(each_roi) for each_roi in roi) * .8, min(max(each_roi) for each_roi in roi) * 1.2])

            pyplot.legend()
            pyplot.pause(.05)


def simple_predict(approximations: Sequence[Approximation[Sequence[float]]], pairs: Sequence[Tuple[str, str]], safety: float):
    # todo: implement simple rate change predict from current rate change, take best, invest all
    # use error distance normalized as error, use recurrency

    time_range = 1532491200000, 1577836856000     # full

    interval_minutes = 1

    timestamps = []
    results_errors = [[] for _ in approximations]
    results_amounts = [[] for _ in approximations]
    amounts_tmp = [0. for _ in approximations]
    errors_tmp = [0. for _ in approximations]

    assets_current = [-1 for _ in approximations]
    amounts = [1. for _ in approximations]
    errors = [0. for _ in approximations]

    fee = .01

    no_datapoints = (time_range[1] - time_range[0]) // (interval_minutes * 60000)

    fig, ax_amount = pyplot.subplots()
    ax_error = ax_amount.twinx()

    max_size = 20

    header_rates = tuple("-".join(each_pair) + "_close" for each_pair in pairs)

    t_last = -1
    ratios_last = None
    rates_last = None
    for t, snapshot in enumerate(merge_generator(pairs, timestamp_range=time_range, interval_minutes=interval_minutes)):
        # get data
        ts = snapshot["close_time"]

        rates_this = tuple(snapshot[column] for column in header_rates)

        if rates_last is None:
            rates_last = rates_this
            continue

        ratios_this = tuple(0. if 0. >= each_last else each_this / each_last for each_this, each_last in zip(rates_this, rates_last))

        if ratios_last is None:
            ratios_last = ratios_this
            continue

        # cycle
        for i, each_approximation in enumerate(approximations):
            # first predict
            input_values = ratios_last
            output_values = each_approximation.output(input_values)

            # then act
            asset_output, asset_ratio = max(enumerate(output_values), key=lambda x: x[1])
            if asset_output != assets_current[i] and 0. < rates_this[asset_output]:
                if safety * fee < asset_ratio - 1. or assets_current[i] < 0:
                    amounts[i] *= (1. - fee)
                    assets_current[i] = asset_output

            # then learn
            target_values = ratios_this
            asset_target, _ = max(enumerate(target_values), key=lambda x: x[1])
            # each_approximation.fit(input_values, target_values, 60 * 24)
            each_approximation.fit(input_values, target_values, t)

            # updates
            if 0. < ratios_this[assets_current[i]]:
                amounts[i] *= ratios_this[assets_current[i]]
            errors[i] = float(asset_output == asset_target)

            amounts_tmp[i] += amounts[i]
            errors_tmp[i] += errors[i]

        # update
        rates_last = rates_this
        ratios_last = ratios_this

        # plotting
        if Timer.time_passed(1000):
            print(f"finished {100. * t / no_datapoints:5.2f}% total...")
            for i, approximation in enumerate(approximations):
                print(f"current asset for approximation {i:d}: {'-'.join(pairs[assets_current[i]]):s} ({approximation.__class__.__name__:s})")

            # update
            if t_last < 0:
                t_last = t
                continue

            interval_plot = t - t_last
            t_last = t

            # generate data
            timestamps.append(ts)
            timestamps = timestamps[-max_size:]
            for i in range(len(approximations)):
                results_amounts[i].append(amounts_tmp[i] / interval_plot)
                results_amounts[i] = results_amounts[i][-max_size:]
                amounts_tmp[i] = 0.

                results_errors[i].append(errors_tmp[i] / interval_plot)
                results_errors[i] = results_errors[i][-max_size:]
                errors_tmp[i] = 0.

            # plot
            ax_amount.clear()
            ax_error.clear()

            ax_amount.set_xlabel("time")
            ax_amount.set_ylabel("value in ETH")

            ax_error.set_ylabel("average error during interval")
            ax_error.yaxis.label.set_color("grey")

            ax_amount.xaxis.set_major_formatter(dates.DateFormatter("%d.%m.%Y %H:%M"))
            ax_amount.xaxis.set_major_locator(MaxNLocator(10))

            ax_error.xaxis.set_major_formatter(dates.DateFormatter("%d.%m.%Y %H:%M"))
            ax_error.xaxis.set_major_locator(MaxNLocator(10))

            plots_error = []
            plots_amount = []
            datetime_axis = tuple(datetime.datetime.utcfromtimestamp(x // 1000) for x in timestamps)
            for i, approximation in enumerate(approximations):
                p_e, = ax_error.plot(datetime_axis, results_errors[i], label=f"error {approximation.__class__.__name__:s}", alpha=.25)
                p_a, = ax_amount.plot(datetime_axis, results_amounts[i], label=f"{approximation.__class__.__name__:s}", alpha=1.)

                #plots_error.append(p_e)
                plots_amount.append(p_a)

            ax_error.set_ylim((0., 1.))

            val_min_amount, val_max_amount = min(min(each_amounts) for each_amounts in results_amounts), max(max(each_amounts) for each_amounts in results_amounts)
            ax_amount.set_ylim([val_min_amount - .2 * (val_max_amount - val_min_amount),  val_max_amount + .2 * (val_max_amount - val_min_amount)])

            pyplot.setp(ax_amount.xaxis.get_majorticklabels(), rotation=-45, ha="left", rotation_mode="anchor", fontsize="10")
            pyplot.legend(plots_error + plots_amount, tuple(line.get_label() for line in plots_error + plots_amount))
            pyplot.tight_layout()
            pyplot.pause(.05)


def running_simple():
    random.seed(235245)
    safety = 1.2
    pairs = get_pairs_from_filesystem()
    pairs = random.sample(pairs, 10)
    print(pairs)

    no_assets = len(pairs)
    learners = MultivariatePolynomialRegression(no_assets, 2, no_assets), MultivariatePolynomialRecurrentRegression(no_assets, 2, no_assets)
    simple_predict(learners, pairs, safety)


if __name__ == "__main__":
    # learn_investment(stop_training_at=1532491200000 + (60000 * 60 * 24 * 7))
    # learn_timeseries()
    running_simple()
