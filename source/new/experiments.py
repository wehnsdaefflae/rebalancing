import queue
from typing import Iterable, Sequence, Tuple, Generator, Union, Collection, Callable, Type, Any, TypeVar

from matplotlib import pyplot

from source.new.binance_examples import STATS, get_pairs
from source.new.learning import Classification, MultivariateRegression, PolynomialClassification, RecurrentPolynomialClassification, Approximation, smear, \
    MultivariatePolynomialRegression, MultivariatePolynomialRecurrentRegression
from source.new.optimal_trading import generate_multiple_changes
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
    ) + ("target", )


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
            yield tuple(t(split[i]) for t, i in zip(types, indices))


def extract_changes(generator: Iterable[Sequence[Any]], indices_float: Collection[int]) -> Iterable[Sequence[Any]]:
    len_each = -1
    last = None
    for each in generator:
        if last is None:
            last = each
            len_each = len(each)
            continue
        assert len(each) == len_each
        yield tuple(each[i] if i not in indices_float else 0. if 0. >= float(last[i]) or 0. >= float(each[i]) else float(each[i] / last[i]) for i in range(len_each))


INPUT_VALUES = Sequence[float]
TARGET_CLASS = int
EXAMPLE = Tuple[INPUT_VALUES, TARGET_CLASS]

INFO_INVESTMENT = Tuple[int, int, int, float, float, float, float]        # timestamp, output, target, error, knowledgeability, decidedness, roi


def simulate_investment(classification: Classification, examples: Iterable[SNAPSHOT_BINANCE], names_assets: Sequence[str], fees: float) -> Generator[INFO_INVESTMENT, None, None]:
    amount_asset = 1.
    index_asset = -1

    indices_rates = range(1, len(names_assets) + 1)
    examples_changes = extract_changes(examples, indices_rates)

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

        output_class = classification.output(rates)

        details = classification.get_details_last_output()
        output_raw = details["raw output"]
        error = MultivariateRegression.error(output_raw, tuple(float(i == index_target) for i in output_raw))
        roi = amount_asset * rates[index_asset]

        classification.fit(rates, index_target, i + 1)

        yield timestamp, names_assets[output_class], names_assets[index_target], error, details["knowledgeability"], details["decidedness"], roi


INFO_PREDICTION = Tuple[int, int, int, float, float]        # timestamp, output, target, error, knowledgeability, decidedness


def predict_rate(regression: Approximation[Sequence[float]], examples: Iterable[SNAPSHOT_BINANCE], names_assets: Sequence[str]) -> Generator[INFO_PREDICTION, None, None]:
    last_snapshot = None
    error_total = -1.

    indices_rates = range(1, len(names_assets) + 1)
    example_changes = extract_changes(examples, indices_rates)

    for i, snapshot_changes in enumerate(example_changes):
        if last_snapshot is None:
            last_snapshot = snapshot_changes
            continue
        input_values = tuple(last_snapshot[i] for i in indices_rates)
        output_values = regression.output(input_values)
        target_values = tuple(snapshot_changes[i] for i in indices_rates)

        error = MultivariateRegression.error(output_values, target_values)

        timestamp = last_snapshot[0]
        error_total = smear(error_total, error, i - 1)

        regression.fit(input_values, target_values, i)

        yield timestamp, output_values, target_values, error, error_total


def learn_timeseries():
    pairs = ("EOS", "ETH"),
    stats = "close",

    names_assets = tuple(f"{each_pair[0].upper():s}-{each_pair[1].upper()}" for each_pair in pairs)
    columns = binance_columns(names_assets, stats)

    examples = iterate_snapshots("../../data/examples/binance_examples.csv", columns, types_binance)

    regression_non_recurrent = MultivariatePolynomialRegression(1, 4, 1)
    simulation_non_recurrent = predict_rate(regression_non_recurrent, examples, names_assets)

    regression_recurrent = MultivariatePolynomialRecurrentRegression(1, 4, 1)
    simulation_recurrent = predict_rate(regression_recurrent, examples, names_assets)

    fig, ax = pyplot.subplots()

    times = queue.Queue(maxsize=100)
    error_a = queue.Queue(maxsize=100)
    error_b = queue.Queue(maxsize=100)

    print_legend = True
    # pyplot.ion()
    for i, (snapshot_a, snapshot_b) in enumerate(zip(simulation_non_recurrent, simulation_recurrent)):
        # print(snapshot)

        if Timer.time_passed(1000):
            print(f"finished reading {i:d} examples...")

            times.put(i)
            error_a.put(snapshot_a[-1])
            error_b.put(snapshot_b[-1])

            ax.clear()
            t = list(times.queue)
            ax.plot(t, list(error_a.queue), label="non recurrent", color="black", alpha=.5)
            ax.plot(t, list(error_b.queue), label="recurrent", color="blue", alpha=.5)

            if print_legend:
                pyplot.legend()
                print_legend = False
            pyplot.pause(.05)

            #print(print("non_recurrent: " + str(snapshot_a)))
            #print(print("recurrent: " + str(snapshot_b)))


def learn_investment():
    pairs = get_pairs()

    stats = STATS
    stats = "close",

    names_assets = tuple(f"{each_pair[0].upper():s}-{each_pair[1].upper()}" for each_pair in pairs)
    columns = binance_columns(names_assets, stats)

    # all assets polynomial for all assets is too much
    # classification = PolynomialClassification(len(pairs), 1, len(pairs))
    classification = RecurrentPolynomialClassification(len(pairs), 2, len(pairs))

    examples = iterate_snapshots("../../data/examples/binance_examples.csv", columns, types_binance)
    simulation = simulate_investment(classification, examples, names_assets, .01)

    for i, snapshot in enumerate(simulation):
        print(snapshot)

        if Timer.time_passed(2000):
            print(f"finished reading {i:d} examples...")


if __name__ == "__main__":
    # learn_investment()
    learn_timeseries()
