import glob
import os
from typing import Tuple, Sequence, Iterable, Generator, Collection, Union

from source.config import RAW_BINANCE_DIR, PATH_DIRECTORY_DATA, STATS_NO_TIME
from source.data.generators.snapshots_binance import rates_binance_generator

# from flyingcircus import base
from source.strategies.infer_investment_path.optimal_trading import generate_multiple_changes, generate_matrix
from source.tools.timer import Timer


def get_pairs_from_filesystem() -> Sequence[Tuple[str, str]]:
    return tuple(
        (x[:-3], x[-3:])
        for x in (
            os.path.splitext(y)[0]
            for y in (
                os.path.basename(z)
                for z in sorted(glob.glob(RAW_BINANCE_DIR + "*.csv"))
            )
        )
    )


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
            # If the read byte is source line character then it means one line is read
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


def generate_path(matrix: Sequence[Tuple[Sequence[int], Tuple[int, float]]]) -> Sequence[int]:
    path = []
    asset_last = -1

    len_matrix = len(matrix)

    for t in range(len_matrix - 1, -1, -1):
        assets_from, (asset_max, _) = matrix[t]
        if t >= len_matrix - 1:
            asset_last = asset_max
            path.append(asset_last)

        asset_last = assets_from[asset_last]
        path.insert(0, asset_last)

        if Timer.time_passed(2000):
            print(f"finished reading {100. * (t - len_matrix) / len_matrix:5.2f}% percent of path...")

    return path


def generate_path_from_file(path_file: str) -> Sequence[int]:
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

    header_rates_only = tuple(f"{each_pair[0]:s}-{each_pair[1]:s}_close" for each_pair in pairs)
    rates = (
        tuple(snapshot[x] for x in header_rates_only)
        for snapshot in rates_binance_generator(pairs=pairs, timestamp_range=time_range, interval_minutes=interval_minutes, header=("close_time", "close"))
    )
    matrix_change = generate_multiple_changes(rates)
    return generate_matrix(no_assets, matrix_change, .01, bound=100)


def combine_assets_header(pairs: Sequence[Tuple[str, str]], header: Sequence[str]) -> Sequence[str]:
    return ("close_time", ) + tuple(f"{each_pair[0]:s}-{each_pair[1]:s}_{column:s}" for each_pair in pairs for column in header if "close_time" not in column)


def convert_to_string(value: Union[int, float]) -> str:
    if isinstance(value, int):
        return f"{value:d}"

    elif isinstance(value, float):
        return f"{value:.8f}"

    raise ValueError("inconvertible")


def write_examples(interval_minutes: int, pairs: Sequence[Tuple[str, str]], path_investment: Sequence[int], file_path: str, header: Sequence[str], time_range: Tuple[int, int]):
    len_path = len(path_investment)
    print(f"length path: {len_path:d}")

    generate_stats = rates_binance_generator(pairs=pairs, timestamp_range=time_range, header=header, interval_minutes=interval_minutes, directory_data=PATH_DIRECTORY_DATA)
    header_combined = tuple(combine_assets_header(pairs, header))

    print("writing examples...")
    names_pairs = tuple(f"{x[0]:s}-{x[1]}" for x in pairs)
    with open(file_path, mode="a") as file:
        file.write("\t".join(header_combined + ("target", )) + "\n")

        last_i = -1
        for i, (snapshot, target) in enumerate(zip(generate_stats, path_investment)):
            line = tuple(convert_to_string(snapshot[column]) for column in header_combined) + (names_pairs[target], )
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
    pairs = get_pairs_from_filesystem()[:10]
    # pairs = pairs[:5]

    # time_range = 1532491200000, 1577836856000     # full
    # time_range = 1532491200000, 1554710537999     # 7/10ths
    time_range = 1532491200000, 1532491800000       # short

    interval_minutes = 1
    no_datapoints = (time_range[1] - time_range[0]) // (interval_minutes * 60000)

    matrix = binance_matrix(pairs, time_range, interval_minutes)

    file_path_matrix = PATH_DIRECTORY_DATA + "examples/binance_matrix_small.csv"
    store_matrix(file_path_matrix, matrix, no_datapoints)

    path = generate_path_from_file(file_path_matrix)

    write_examples(interval_minutes, pairs, path, PATH_DIRECTORY_DATA + "examples/binance_examples_small.csv", STATS_NO_TIME, time_range)


if __name__ == "__main__":
    main()
