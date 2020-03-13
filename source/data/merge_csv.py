from __future__ import annotations
import glob
import os
from typing import Tuple, Sequence

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


def round_down_timestamp(timestamp: int, round_to: int) -> int:
    return timestamp // round_to * round_to


def get_timestamp_close_boundaries(files: Sequence[str]) -> Tuple[int, int]:
    print(f"getting timestamp boundaries...")
    timestamp_close_min = -1
    timestamp_close_max = -1
    for i, each_file in enumerate(files):
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


def write_header(path_file: str, pairs: Sequence[Tuple[str, str]]):
    with open(path_file, mode="a") as file:
        header = ("timestamp_close", "timestamp_open") + get_header(pairs)
        file.write("\t".join(header) + "\n")


def write_timestamps(file_path: str, files: Sequence[str], interval_timestamp: int):
    print(f"writing time stamps...")
    with open(file_path, mode="a") as file:
        ts_close_min, ts_close_max = get_timestamp_close_boundaries(files)
        timestamp_from = round_down_timestamp(ts_close_min, interval_timestamp)
        timestamp_to = round_down_timestamp(ts_close_max, interval_timestamp)
        for each_ts_close in range(timestamp_from, timestamp_to + interval_timestamp, interval_timestamp):
            values = f"{each_ts_close:d}", f"{each_ts_close - interval_timestamp:d}"
            file.write("\t".join(values) + "\n")


def timestamps_base_line(line: str) -> Tuple[int, int]:
    stripped = line.strip()
    split = stripped.split("\t", maxsplit=2)
    return int(split[1]), int(split[0])


def add_file(path_basis: str, path_extension: str, interval_timestamp: int):
    print(f"adding {path_extension:s} to {path_basis:s}...")
    path_tmp = os.path.dirname(path_basis) + "/temp.tmp"
    if os.path.isfile(path_tmp):
        os.remove(path_tmp)

    with open(path_basis, mode="r") as file_basis, open(path_extension, mode="r") as file_extension, open(path_tmp, mode="a") as file_temp:
        header = next(file_basis)
        file_temp.write(header)

        for i, line in enumerate(file_extension):
            stats = stat_from_line(line)
            timestamp_ext_close = stats[6]

            try:
                line_basis = next(file_basis)
            except StopIteration:
                print(f"base file terminated prematurely at the {i+1:d}th line of {file_extension:s}.")
                break

            stripped_basis = line_basis.strip()
            _, timestamp_base_close = timestamps_base_line(stripped_basis)

            while timestamp_ext_close >= timestamp_base_close:
                line_new = stripped_basis + "\t" + "\t".join(str(x) for x in stat_empty)
                file_temp.write(line_new + "\n")
                line_basis = next(file_basis)
                stripped_basis = line_basis.strip()
                _, timestamp_base_close = timestamps_base_line(stripped_basis)

            file_temp.write(stripped_basis + "\t" + line)

    os.remove(path_basis)
    os.rename(path_tmp, path_basis)


def main():
    interval_timestamp = 60000

    directory_data = "../../data/"

    directory_csv = directory_data + "binance/"
    files = sorted(glob.glob(directory_csv + "*.csv"))

    directory_merged = directory_data + "merged/"

    path_merged = directory_merged + "merged.csv"

    pairs = get_pairs(files)
    write_header(path_merged, pairs)
    write_timestamps(path_merged, files, interval_timestamp)

    for i, each_file in enumerate(files):
        print(f"adding {i + 1:d} / {len(files)} ({each_file:s})...")
        add_file(path_merged, each_file, interval_timestamp)


if __name__ == "__main__":
    main()
