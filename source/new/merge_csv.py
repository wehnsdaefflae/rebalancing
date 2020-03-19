from __future__ import annotations

import glob
import os
from typing import Tuple, Sequence, Generator, Union, Optional, Iterable

from source.tools.timer import Timer

STATS_TYPES = Tuple[Union[int, float], ...]

STAT_COLUMNS = (
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base_asset_volume",
    "taker_buy_quote_asset_volume",
    "ignore",
)

stat_empty = -1, -1., -1., -1., -1., -1., -1, -1., -1, -1., -1., -1.

indices_ints = 0, 6, 8


def stat_from_line(line: str, indices: Optional[Sequence[int]] = None) -> STATS_TYPES:
    stripped = line.strip()
    split = stripped.split("\t")
    if indices is None:
        return int(split[0]), float(split[1]), float(split[2]), \
               float(split[3]), float(split[4]), float(split[5]), \
               int(split[6]), float(split[7]), int(split[8]), \
               float(split[9]), float(split[10]), float(split[11])

    return tuple(int(split[i]) if i in indices_ints else float(split[i]) for i in indices)


def get_close_time(line: str) -> int:
    split = line[:-1].split("\t")
    return int(split[6])


def make_values(line: str, indices: Sequence[int]) -> Sequence[Union[int, float]]:
    strip = line.strip()
    split = strip.split("\t")
    return tuple(int(split[i]) if i in indices_ints else float(split[i]) for i in indices)


def make_empty(timestamp_open: int, timestamp_close: int, indices: Sequence[int]) -> Sequence[Union[int, float]]:
    return tuple(timestamp_open if i == 0 else timestamp_close if i == 6 else stat_empty[i] for i in indices)


def generator_file(
        path_file: str,
        timestamp_range: Optional[Tuple[int, int]],
        interval_minutes: int,
        indices: Sequence[int],
        data_difference_timestamp: int = 60000) -> Generator[STATS_TYPES, None, None]:

    interval_timestamp = interval_minutes * data_difference_timestamp
    no_ranges = (timestamp_range[1] - timestamp_range[0]) // interval_timestamp
    ranges = tuple(
        (timestamp_range[0] + i * interval_timestamp, timestamp_range[0] + (i + 1) * interval_timestamp)
        for i in range(no_ranges)
    )

    index_next_range = 0
    with open(path_file, mode="r") as file:
        line = next(file, None)
        while line is not None and index_next_range < no_ranges:
            current_range = ranges[index_next_range]
            timestamp_close = get_close_time(line)

            # get next lines until line fits
            while current_range[0] >= timestamp_close:
                line = next(file, None)
                if Timer.time_passed(2000):
                    print(f"skipping lines in {os.path.basename(path_file):s} until reaching timestamp {current_range[0]:d}...")

                if line is None:
                    break

                timestamp_close = get_close_time(line)

            if line is None:
                break

            # current line fits, yield it, proceed to next range
            if current_range[0] < timestamp_close <= current_range[1]:
                yield make_values(line, indices)
                index_next_range += 1

            # yield empties for timestamps not covered by data
            while current_range[1] < timestamp_close:
                yield make_empty(current_range[1] - data_difference_timestamp, current_range[1] - 1, indices)
                index_next_range += 1
                current_range = ranges[index_next_range]
                if Timer.time_passed(2000):
                    print(f"filling gaps in {os.path.basename(path_file):s} until reaching timestamp {timestamp_close:d}...")

            if Timer.time_passed(2000):
                print(f"generated {index_next_range:d} of {no_ranges:d} continual data points for {os.path.basename(path_file):s}...")

    # fill rest
    for index_current_range in range(index_next_range, no_ranges):
        current_range = ranges[index_current_range]
        yield make_empty(current_range[1] - data_difference_timestamp, current_range[1] - 1, indices)
        if Timer.time_passed(2000):
            print(f"finished filling {100. * index_current_range / (no_ranges - index_next_range):5.2f}% of final gaps...")


def round_down_timestamp(timestamp: int, round_to: int) -> int:
    return timestamp // round_to * round_to


def get_timestamp_close_boundaries(files: Sequence[str]) -> Tuple[int, int]:
    timestamp_close_min = -1
    timestamp_close_max = -1
    for i, each_file in enumerate(files):
        print(f"getting timestamp boundaries of {each_file:s}...")
        with open(each_file, mode="r") as file:
            for each_line in file:
                stripped = each_line.strip()
                split = stripped.split("\t")

                timestamp_close = int(split[6])

                if timestamp_close_min < 0 or timestamp_close < timestamp_close_min:
                    timestamp_close_min = timestamp_close
                if timestamp_close_max < 0 or timestamp_close_max < timestamp_close:
                    timestamp_close_max = timestamp_close

    return timestamp_close_min, timestamp_close_max


def get_header(pairs: Sequence[Tuple[str, str]]) -> Tuple[str, ...]:
    return tuple("_".join(each_pair) + "_" + each_stat for each_pair in pairs for each_stat in STAT_COLUMNS)


def merge_generator(
        pairs: Optional[Iterable[Tuple[str, str]]] = None,
        timestamp_range: Optional[Tuple[int, int]] = None,
        interval_minutes: int = 1,
        header: Tuple[str, ...] = ("close_time", "close", ),
        directory_data: str = "../../data/") -> Generator[Sequence[Sequence[Union[int, float]]], None, None]:

    directory_csv = directory_data + "binance/"
    if pairs is None:
        files = sorted(glob.glob(f"{directory_csv:s}*.csv"))
    else:
        files = sorted(f"{directory_csv:s}{each_pair[0].upper():s}{each_pair[-1].upper():s}.csv" for each_pair in pairs)

    if timestamp_range is None:
        print(f"determining timestamp boundaries...")
        timestamp_range = get_timestamp_close_boundaries(files)
        print(f"timestamp start {timestamp_range[0]:d}, timestamp end {timestamp_range[1]:d}")

    else:
        assert timestamp_range[0] < timestamp_range[1]

    indices = tuple(STAT_COLUMNS.index(x) for x in header)
    generators_all = tuple(generator_file(each_file, timestamp_range, interval_minutes, indices) for each_file in files)

    yield from zip(*generators_all)


def main_single():
    files = ['../../data/binance/BCCETH.csv', '../../data/binance/BNBETH.csv', '../../data/binance/TUSDETH.csv']
    timestamp_range = 1501113780000, 1577836860000
    interval_minutes = 1
    indices = 6, 4,
    generators_all = tuple(generator_file(each_file, timestamp_range, interval_minutes, indices=indices) for each_file in files)

    value_lists = tuple([] for _ in generators_all)

    for i, each_generator in enumerate(generators_all):
        print(f"generator {i:d}")
        for v in each_generator:
            value_lists[i].append(v)

        if Timer.time_passed(2000):
            print(f"processed {i:d} elements")

    print([len(x) for x in value_lists])
    print([x[-1] for x in value_lists])
    print()


def main():
    g = merge_generator(
        (
            ("bcc", "eth"), ("bnb", "eth"), ("tusd", "eth"),
        ),
        interval_minutes=1,
        header=("close_time", "close",),
        timestamp_range=(1577836260000, 1577836860000),
    )
    v = None
    for i, v in enumerate(g):
        print(v)
        if Timer.time_passed(2000):
            print(f"iterated over {i:d} elements")
    print(v)


if __name__ == "__main__":
    main()
