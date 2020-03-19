import glob
import os
from typing import Tuple, Sequence, Iterable, Generator, Collection

from source.new.optimal_trading import get_crypto_rates, generate_multiple_changes, generate_matrix
from source.tools.timer import Timer

# from flyingcircus import base

PATH_DIRECTORY_DATA = "../../data/"
RAW_BINANCE_DIR = PATH_DIRECTORY_DATA + "binance/"


STATS = (
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


def get_pairs() -> Collection[Tuple[str, str]]:
    files = sorted(glob.glob(RAW_BINANCE_DIR + "*.csv"))

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


def read_reverse_order(file_name: str) -> Generator[str, None, None]:
    # https://thispointer.com/python-read-a-file-in-reverse-order-line-by-line/
    # Open file for reading in binary mode
    with open(file_name, 'rb') as read_obj:
        # Move the cursor to the end of the file
        read_obj.seek(0, os.SEEK_END)
        # Get the current position of pointer i.e eof
        pointer_location = read_obj.tell()
        # Create a buffer to keep the last read line
        buffer = bytearray()
        # Loop till pointer reaches the top of the file
        while pointer_location >= 0:
            # Move the file pointer to the location pointed by pointer_location
            read_obj.seek(pointer_location)
            # Shift pointer location by -1
            pointer_location = pointer_location -1
            # read that byte / character
            new_byte = read_obj.read(1)
            # If the read byte is new line character then it means one line is read
            if new_byte == b'\n':
                # Fetch the line from buffer and yield it
                yield buffer.decode()[::-1]
                # Reinitialize the byte array to save next line
                buffer = bytearray()
                # todo: buffer.clear() ?
            else:
                # If last read character is not eol then add it in buffer
                buffer.extend(new_byte)

        # As file is read completely, if there is still data in buffer, then its the first line.
        if len(buffer) > 0:
            # Yield the first line too
            yield buffer.decode()[::-1]


def generate_path(path_file: str) -> Sequence[int]:
    path = []
    asset_last = -1
    #with open(path_file, mode="r") as file:

    size_total = os.path.getsize(path_file)
    size_read_last = -1
    size_read = 0
    for i, line in enumerate(read_reverse_order(path_file)):
        #for i, line in enumerate(base.readline(file, reverse=True)):
        stripped = line.strip()
        if len(stripped) < 1:
            continue
        snapshot_str, rest_str = stripped.split(", ")
        snapshot_split = snapshot_str.split("\t")
        rest = rest_str.split(": ")

        if i < 1:
            asset_last = int(rest[0])
            path.append(asset_last)

        asset_last = int(snapshot_split[asset_last])
        path.insert(0, asset_last)

        size_read += len(line)

        if Timer.time_passed(2000):
            if size_read_last < 0:
                min_str = "??"

            else:
                speed = (size_read - size_read_last) // 2
                seconds_remaining = (size_total - size_read) // speed
                min_str = f"{seconds_remaining // 60:d}"

            print(f"finished reading {100. * size_read / size_total:5.2f}% percent of path. {min_str:s} minutes remaining...")
            size_read_last = size_read

    return path


def binance_matrix(pairs: Collection[Tuple[str, str]], time_range: Tuple[int, int], interval_minutes: int) -> Generator[Tuple[Sequence[int], Tuple[int, float]], None, None]:
    no_assets = len(pairs)
    rates_no_timestamps = (
        tuple(each_asset[0] for each_asset in snapshot)
        for timestamp, snapshot in get_crypto_rates(pairs, ("close", ), timestamp_range=time_range, interval_minutes=interval_minutes)
    )
    matrix_change = generate_multiple_changes(rates_no_timestamps)
    return generate_matrix(no_assets, matrix_change, .01, bound=100)


def write_examples(interval_minutes: int, pairs: Collection[Tuple[str, str]], path_investment: Sequence[int], file_path: str, stats: Sequence[str], time_range: Tuple[int, int]):
    len_path = len(path_investment)
    print(f"length path: {len_path:d}")

    generate_stats = get_crypto_rates(pairs, stats, timestamp_range=time_range, interval_minutes=interval_minutes, directory_data=PATH_DIRECTORY_DATA)

    print("writing examples...")
    names_pairs = tuple(f"{x[0]:s}-{x[1]}" for x in pairs)
    header = ("timestamp",) + tuple(f"{each_pair:s}_{each_column:s}" for each_pair in names_pairs for each_column in stats) + ("target",)  # todo: reward at some point?
    with open(file_path, mode="a") as file:
        file.write("\t".join(header) + "\n")

        last_i = -1
        for i, ((ts, snapshot), target) in enumerate(zip(generate_stats, path_investment)):
            segment_data = [
                f"{each_value:d}" if stats[i] in ("open_time", "close_time", "number_of_trades") else f"{each_value:.8f}"
                for each_asset in snapshot
                for i, each_value in enumerate(each_asset)
            ]
            line = [f"{ts:d}"] + segment_data + [names_pairs[target]]
            file.write("\t".join(line) + "\n")

            if Timer.time_passed(2000):
                if last_i < 0:
                    min_str = "??"
                else:
                    speed = (i - last_i) // 2
                    seconds_remaining = (len_path - i) // speed
                    min_str = f"{seconds_remaining // 60:d}"

                print(f"finished {i * 100. / len_path:5.2f}% of writing examples. {min_str:s} minutes remaining...")
                last_i = i


def main():
    pairs = get_pairs()
    # pairs = pairs[:5]

    time_range = 1532491200000, 1577836856000     # full
    # time_range = 1532491200000, 1532491800000       # short

    interval_minutes = 1
    no_datapoints = (time_range[1] - time_range[0]) // (interval_minutes * 60000)

    matrix = binance_matrix(pairs, time_range, interval_minutes)

    file_path_matrix = PATH_DIRECTORY_DATA + "examples/binance_matrix.csv"
    store_matrix(file_path_matrix, matrix, no_datapoints)

    path = generate_path(file_path_matrix)

    write_examples(interval_minutes, pairs, path, PATH_DIRECTORY_DATA + "examples/binance_examples.csv", STATS, time_range)


if __name__ == "__main__":
    main()
