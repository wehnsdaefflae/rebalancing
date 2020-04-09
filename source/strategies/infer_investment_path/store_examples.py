from typing import Sequence, Tuple

from source.config import PATH_DIRECTORY_DATA, STATS_NO_TIME
from source.data.generators.snapshots_binance import rates_binance_generator
from source.strategies.infer_investment_path.optimal_trading_filesystem import binance_matrix, store_matrix, \
    generate_path_from_file
from source.tools.functions import get_pairs_from_filesystem, combine_assets_header, convert_to_string
from source.tools.timer import Timer


def write_examples_path(interval_minutes: int, pairs: Sequence[Tuple[str, str]], path_investment: Sequence[int], file_path: str, header: Sequence[str], time_range: Tuple[int, int]):
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

    write_examples_path(interval_minutes, pairs, path, PATH_DIRECTORY_DATA + "examples/binance_examples_small.csv", STATS_NO_TIME, time_range)


if __name__ == "__main__":
    main()
