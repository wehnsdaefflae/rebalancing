import glob
import os
from typing import Tuple, Sequence, Iterable

from source.tactics.optimal_trading import split_time_and_data, make_source_matrix, make_path_from_sourcematrix, get_crypto_rates
from source.tools.timer import Timer

from flyingcircus import base


def get_pairs() -> Sequence[Tuple[str, str]]:
    pairs = (
        ("bcc", "eth"), ("bnb", "eth"), ("dash", "eth"), ("icx", "eth"),
        ("iota", "eth"), ("ltc", "eth"), ("nano", "eth"), ("poa", "eth"),
        ("qtum", "eth"), ("theta", "eth"), ("tusd", "eth"), ("xmr", "eth")
    )

    directory = "../../data/binance/"
    files = sorted(glob.glob(directory + "*.csv"))

    names_base = (os.path.basename(x) for x in files)
    names_first = (os.path.splitext(x)[0] for x in names_base)
    pairs = tuple((x[:-3], x[-3:]) for x in names_first)
    return pairs


def store_matrix(path_file: str, matrix: Iterable[Tuple[Sequence[int], Tuple[int, float]]], no_datapoints: int):
    t_last = 0
    with open(path_file, mode="a") as file:
        for t, (snapshot, (best, roi)) in enumerate(matrix):
            line = "\t".join(f"{a:d}" for a in snapshot) + f", {best:d}: {roi:.8f}\n"
            file.write(line)

            if Timer.time_passed(2000):
                speed_per_sec = (t - t_last) // 2
                secs_remaining = (no_datapoints - t) // speed_per_sec
                minutes_remaining = secs_remaining // 60

                print(f"finished {100. * t / no_datapoints:5.2f}% of saving matrix ({t:d} of {no_datapoints:d} total). {minutes_remaining:d} minutes remaining...")
                t_last = t


def generate_path(path_file: str) -> Sequence[int]:
    path = []
    with open(path_file, mode="a") as file:
        asset_last = -1
        for i, line in enumerate(base.readline(file, reverse=True)):
            stripped = line.strip()
            snapshot_str, rest_str = stripped.split(", ")
            snapshot_split = snapshot_str.split("\t")
            rest = rest.split(": ")

            if i < 1:
                asset_last = int(rest[0])
                path.append(asset_last)

            asset_last = int(snapshot_split[asset_last])
            path.insert(0, asset_last)

        if Timer.time_passed(2000):
            print(f"finished reading {i:d} time steps of path...")

    return path


def main():
    pairs = get_pairs()

    stats = (
        # "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        # "close_time",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base_asset_volume",
        "taker_buy_quote_asset_volume",
        "ignore",
    )

    path_directory = "../../data/"

    time_range = 1532491200000, 1577836856000
    # time_range = 1532491200000, 1532491800000

    interval_minutes = 1
    no_datapoints = (time_range[1] - time_range[0]) // (interval_minutes * 60000)

    timestamps = []
    no_assets = len(pairs)
    generate_rates_for_actions = (
        split_time_and_data(x, timestamp_storage=timestamps)
        for x in get_crypto_rates(pairs, ("close", ), timestamp_range=time_range, interval_minutes=interval_minutes, directory_data=path_directory)
    )

    matrix = make_source_matrix(no_assets, generate_rates_for_actions, fees=.01)
    store_matrix("../../data/examples/matrix.csv", matrix, no_datapoints)
    path = generate_path("../../data/examples/matrix.csv")

    names_pairs = tuple(f"{x[0]:s}-{x[1]}" for x in pairs)

    generate_rates_for_examples = (
        split_time_and_data(x)
        for x in get_crypto_rates(pairs, stats, timestamp_range=time_range, interval_minutes=interval_minutes, directory_data=path_directory)
    )

    print("writing examples...")
    header = ("timestamp",) + tuple(f"{each_pair:s}_{each_column:s}" for each_pair in names_pairs for each_column in stats) + ("target",)  # todo: reward at some point?
    with open("../../data/examples/binance.csv", mode="a") as file:
        file.write("\t".join(header) + "\n")

        for i, (ts, rates, target) in enumerate(zip(timestamps, generate_rates_for_examples, path)):
            line = [f"{ts:d}"] + [f"{x:.8f}" for x in rates] + [names_pairs[target]]
            file.write("\t".join(line) + "\n")
            del line

            if Timer.time_passed(2000):
                print(f"finished {i * 100. / len(timestamps):5.2f}% of writing examples...")


if __name__ == "__main__":
    main()
