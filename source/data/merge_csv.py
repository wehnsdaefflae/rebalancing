from __future__ import annotations
import glob
import os
from typing import Tuple, Sequence, Generator, Union, Optional, Iterable

from source.tools.timer import Timer

STATS = Tuple[Union[int, float], ...]

stats = (
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


def stat_from_line(line: str, indices: Optional[Sequence[int]] = None) -> STATS:
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
        indices: Optional[Sequence[int]] = None) -> Generator[STATS, None, None]:

    if indices is None:
        indices = 0, 6, 1, 4

    data_difference_timestamp = 60000
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

    # fill rest
    for index_current_range in range(index_next_range, no_ranges - 1):
        current_range = ranges[index_current_range]
        yield make_empty(current_range[1] - data_difference_timestamp, current_range[1] - 1, indices)


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


def get_pairs(files: Sequence[str]) -> Sequence[Tuple[str, str]]:
    print(f"getting pairs...")
    pairs = []
    for each_file in files:
        name_base = os.path.basename(each_file)
        name_first = os.path.splitext(name_base)[0]
        asset_from = name_first[:-3]
        asset_to = name_first[-3:]
        each_pair = asset_from, asset_to
        pairs.append(each_pair)
    return pairs


def get_header(pairs: Sequence[Tuple[str, str]]) -> Tuple[str, ...]:
    return tuple("_".join(each_pair) + "_" + each_stat for each_pair in pairs for each_stat in stats)


def data_generator(asset_from: str, asset_to: str, interval_minutes: int = 1, data: Tuple[str, ...] = ("close",)) -> Generator[Sequence[float], None, None]:
    directory_data = "../../data/"
    directory_merged = directory_data + "merged/"
    files = sorted(glob.glob(directory_merged + "merged_*.csv"))
    iterator = 0
    for each_file in files:
        with open(each_file, mode="r") as file:
            header = next(file)
            stripped = header.strip()
            split = stripped.split("\t")
            indices = tuple(split.index(asset_from.upper() + "_" + asset_to.upper() + "_" + each_datum.lower()) for each_datum in data)
            indices = (split.index("timestamp_close"), ) + indices

            for line in file:
                iterator += 1
                if iterator % interval_minutes != 0:
                    continue
                iterator = 0

                split = line[:-1].split("\t")
                yield tuple(float(split[i]) for i in indices)


def merge_generator(
        pairs: Iterable[Tuple[str, str]],
        timestamp_start: int = -1,
        timestamp_end: int = -1,
        interval_minutes: int = 1,
        header: Tuple[str, ...] = ("close",)) -> Generator[Sequence[Union[int, float]], None, None]:

    indices = tuple(stats.index(x) for x in ("open_time", "close_time", ) + header)

    directory_data = "../../data/"
    directory_csv = directory_data + "binance/"
    files = sorted(f"{directory_csv:s}{each_pair[0].upper():s}{each_pair[-1].upper():s}.csv" for each_pair in pairs)

    if 0 < timestamp_start and 0 < timestamp_end:
        assert timestamp_start < timestamp_end

    else:
        print(f"determining timestamp boundaries...")
        ts_close_min, ts_close_max = get_timestamp_close_boundaries(files)
        timestamp_start = ts_close_min if timestamp_start < 0 else timestamp_start
        timestamp_end = ts_close_max if timestamp_end < 0 else timestamp_end

    timestamp_range = timestamp_start, timestamp_end
    generators_all = tuple(generator_file(each_file, timestamp_range, interval_minutes, indices=indices) for each_file in files)

    no_generators = len(generators_all)
    while True:
        stats_all = tuple(next(each_generator, None) for each_generator in generators_all)
        no_nones = stats_all.count(None)
        if no_nones == no_generators:
            break

        assert no_nones == 0

        yield stats_all


def main_single():
    g = generator_file("../../data/binance/ADAETH.csv", (1577836620000, 1577837220000), 1, (0, 6, 4,))
    for v in g:
        print(v)


def main():
    g = merge_generator(
        (("ada", "eth"), ("adx", "eth")),
        interval_minutes=1,
        header=("close",),
        # timestamp_start=1512044700000,
        # timestamp_end=1512045300000,
    )
    v = None
    for i, v in enumerate(g):
        # print(v)
        if Timer.time_passed(2000):
            print(f"iterated over {i:d} elements")
    print(v)

if __name__ == "__main__":
    main()
