import queue
from typing import Iterable, Sequence, Tuple, Generator, Union, Collection, Callable, Type, Any

from matplotlib import pyplot

from source.new.binance_examples import STATS, get_pairs
from source.new.learning import Classification, MultivariateRegression, PolynomialClassification, RecurrentPolynomialClassification, Approximation, smear, \
    MultivariatePolynomialRegression, MultivariatePolynomialRecurrentRegression
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


def simulate_investment(classifications: Sequence[Classification], examples: Iterable[SNAPSHOT_BINANCE], names_assets: Sequence[str], fees: float) -> Generator[INFO_INVESTMENT, None, None]:
    amount_asset = 1.
    index_asset = -1

    indices_rates = range(1, len(names_assets) + 1)
    examples_changes = extract_changes(examples, indices_rates)

    classification_stats = []

    for i, snapshot_changes in enumerate(examples_changes):
        timestamp = snapshot_changes[0]
        rates = [snapshot_changes[i] for i in indices_rates]
        target = snapshot_changes[-1]
        index_target = names_assets.index(target)

        rate_hold = rates[index_asset]
        rate_switch = rates[index_target]

        if index_asset < 0:
            index_asset = index_target
            amount_asset = (1. - fees) * amount_asset / rate_switch

        elif index_asset != index_target:
            amount_asset = amount_asset * (1. - fees) * rate_hold / rate_switch
            index_asset = index_target

        for j, each_classification in enumerate(classifications):
            output_class = each_classification.output(rates)

            details = each_classification.get_details_last_output()
            output_raw = details["raw output"]
            error = MultivariateRegression.error(output_raw, tuple(float(i == index_target) for i in output_raw))

            roi = amount_asset * rates[index_asset]

            each_classification.fit(rates, index_target, i + 1)

            stats = names_assets[output_class], error, details["knowledgeability"], details["decidedness"], roi
            classification_stats.append(stats)

        y = timestamp, target, classification_stats
        yield y


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
            error = MultivariateRegression.error(output_values, target_values)
            # error = float((1. < output_values[0]) != (1. < target_values[0]))

            each_approximation.fit(input_values, target_values, i)
            approximator_stats.append((output_values, error))

        timestamp = last_snapshot[0]

        y = timestamp, target_values, approximator_stats
        yield y


def learn_timeseries():
    pairs = ("AGI", "ETH"),
    stats = "close",

    names_assets = tuple(f"{each_pair[0].upper():s}-{each_pair[1].upper()}" for each_pair in pairs)
    columns = binance_columns(names_assets, stats)

    examples = iterate_snapshots("../../data/examples/binance_examples.csv", columns, types_binance)

    regression_non_recurrent = MultivariatePolynomialRegression(1, 4, 1)
    regression_recurrent = MultivariatePolynomialRecurrentRegression(1, 4, 1)
    learners = regression_non_recurrent, regression_recurrent

    approximations = predict_rate(learners, examples, names_assets)

    fig, ax = pyplot.subplots()
    max_size = 20
    times = []
    error = [[] for _ in learners]
    error_total = [-1. for _ in learners]

    # pyplot.ion()
    for i, (timestamp, target_values, approximator_stats) in enumerate(approximations):
        # print(snapshot)

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
                ax.plot(times, each_error, label=f"{learners[j].__class__.__name__:s}", alpha=.5)

            ax.set_ylim([0, min(max(each_error) for each_error in error) * 1.2])

            pyplot.legend()
            pyplot.pause(.05)


def learn_investment():
    pairs = get_pairs()

    stats = STATS
    stats = "close", "target"

    names_assets = tuple(f"{each_pair[0].upper():s}-{each_pair[1].upper()}" for each_pair in pairs)
    columns = binance_columns(names_assets, stats)

    # all assets polynomial for all assets is too much
    classifications = PolynomialClassification(len(pairs), 1, len(pairs)), RecurrentPolynomialClassification(len(pairs), 1, len(pairs))

    examples = iterate_snapshots("../../data/examples/binance_examples.csv", columns, types_binance)
    simulation = simulate_investment(classifications, examples, names_assets, .01)

    fig, ax = pyplot.subplots()
    max_size = 20
    times = []
    error = [[] for _ in classifications]
    error_total = [-1. for _ in classifications]
    roi = [[] for _ in classifications]
    roi_total = [-1. for _ in classifications]

    for i, (timestamp, target_asset, classification_stats) in enumerate(simulation):
        if Timer.time_passed(1000):
            print(f"finished reading {i:d} examples...")

            print(f"{timestamp:d}: {target_asset:s}")

            times.append(i)
            for j, (each_output, each_error, each_k, each_d, each_roi) in enumerate(classification_stats):
                error_total[j] = each_error if len(times) < 2 else smear(error_total[j], each_error, 10)
                error[j].append(error_total[j])

                roi_total[j] = each_roi if len(times) < 2 else smear(roi_total[j], each_roi, 10)
                roi[j].append(roi_total[j])

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
            ax.set_ylim([0, min(max(each_roi) for each_roi in roi) * 1.2])

            pyplot.legend()
            pyplot.pause(.05)


if __name__ == "__main__":
    learn_investment()
    # learn_timeseries()