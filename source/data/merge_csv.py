from __future__ import annotations
import glob
import os
from typing import Tuple, Sequence, Generator, Union

from source.tools.timer import Timer

STATS = Tuple[int, float, float, float, float, float, int, float, int, float, float, float]

stat_empty = -1, -1., -1., -1., -1., -1., -1, -1., -1, -1., -1., -1.


def stat_from_line(line: str) -> STATS:
    stripped = line.strip()
    split = stripped.split("\t")
    assert len(split) == 12
    return int(split[0]), float(split[1]), float(split[2]), \
           float(split[3]), float(split[4]), float(split[5]), \
           int(split[6]), float(split[7]), int(split[8]), \
           float(split[9]), float(split[10]), float(split[11])


def generator_file(path_file: str) -> Generator[STATS, None, None]:
    with open(path_file, mode="r") as file:
        for line in file:
            yield stat_from_line(line)

    while True:
        yield stat_empty


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
    stats = [
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
    ]

    return tuple("_".join(each_pair) + "_" + each_stat for each_pair in pairs for each_stat in stats)


def data_generator(asset_from: str, asset_to: str, interval_minutes: int = 1, data: Sequence[str] = ("close",)) -> Generator[Sequence[float], None, None]:
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

            for line in file:
                iterator += 1
                if iterator % interval_minutes != 0:
                    continue
                iterator = 0

                split = line[:-1].split("\t")
                yield tuple(float(split[i]) for i in indices)


def merge():
    interval_timestamp = 60000

    directory_data = "../../data/"

    directory_csv = directory_data + "binance/"
    files = sorted(glob.glob(directory_csv + "*.csv"))

    directory_merged = directory_data + "merged/"

    ts_close_min, ts_close_max = get_timestamp_close_boundaries(files)
    timestamp_from = round_down_timestamp(ts_close_min, interval_timestamp) + interval_timestamp
    timestamp_to = round_down_timestamp(ts_close_max, interval_timestamp) + 2 * interval_timestamp

    generators_all = tuple(generator_file(each_file) for each_file in files)
    stats_all = [next(each_generator) for each_generator in generators_all]

    proceed = [False for _ in generators_all]

    pairs = get_pairs(files)
    header = ("timestamp_close", "timestamp_open") + get_header(pairs)

    length_segment = 100000
    no_entries = (timestamp_to - timestamp_from) // interval_timestamp + 1
    print(f"total number of entries: {no_entries:d}")
    ranges = tuple(
        (i * length_segment, min((i + 1) * length_segment, no_entries))
        for i in range(no_entries // length_segment + 1)
    )

    for j, each_range in enumerate(ranges):
        print(f"writing entries # {each_range[0]:d} to {each_range[1]:d}...")
        path_merged = directory_merged + f"merged_{j:05d}.csv"

        print(f"from timestamps {each_range[0] * interval_timestamp + timestamp_from:d} to {each_range[0] * interval_timestamp + timestamp_from:d}...")

        with open(path_merged, mode="a") as file:
            file.write("\t".join(header) + "\n")

            for reference_timestamp_close in range(*each_range):
                reference_timestamp_close *= interval_timestamp
                reference_timestamp_close += timestamp_from

                reference_timestamp_open = reference_timestamp_close - interval_timestamp

                values = [reference_timestamp_close, reference_timestamp_open]

                for i, each_stat in enumerate(stats_all):
                    each_timestamp_close = each_stat[6]
                    if reference_timestamp_open < each_timestamp_close <= reference_timestamp_close:
                        values.extend(each_stat)
                        proceed[i] = True

                    elif each_timestamp_close < reference_timestamp_open:
                        values.extend(stat_empty)
                        proceed[i] = True

                    elif reference_timestamp_close < each_timestamp_close:
                        values.extend(stat_empty)

                for i, p in enumerate(proceed):
                    if not p:
                        continue
                    proceed[i] = False
                    stats_all[i] = next(generators_all[i])

                file.write("\t".join(str(x) for x in values) + "\n")

                if Timer.time_passed(2000):
                    print(f"finished {j+1:d}/{len(ranges):d} {(reference_timestamp_close - timestamp_from) * 100. / (timestamp_to - timestamp_from):5.2f}%...")


def main():
    g = data_generator("ada", "eth", interval_minutes=100, data=("close",))
    s = []
    for v in g:
        s.append(v[0])
        if Timer.time_passed(2000):
            print(f"length so far {len(s):d}")

    print(s)


if __name__ == "__main__":
    main()
