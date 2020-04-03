import os
from typing import Optional, Sequence, Tuple, Generator, Dict, Any, Collection

from source.new.config import STATS_TYPES, STAT_COLUMNS
from source.tools.timer import Timer


def get_int(line: str, index: int) -> int:
    strip = line.strip()
    split = strip.split("\t")
    return int(split[index])


def generator_file(
        path_file: str,
        timestamp_range: Optional[Tuple[int, int]],
        interval_minutes: int,
        header: Sequence[str],
        data_difference_timestamp: int = 60000) -> Generator[Dict[str, Any], None, None]:

    indices_columns = tuple(STAT_COLUMNS.index(x) for x in header)

    interval_timestamp = interval_minutes * data_difference_timestamp
    no_ranges = (timestamp_range[1] - timestamp_range[0]) // interval_timestamp
    ranges = tuple(
        (timestamp_range[0] + i * interval_timestamp, timestamp_range[0] + (i + 1) * interval_timestamp)
        for i in range(no_ranges)
    )

    index_time_close = STAT_COLUMNS.index("close_time")
    index_next_range = 0
    with open(path_file, mode="r") as file:
        line = next(file, None)
        while line is not None and index_next_range < no_ranges:
            current_range = ranges[index_next_range]
            timestamp_close = get_int(line)

            # get next lines until line fits
            while current_range[0] >= timestamp_close:
                line = next(file, None)
                if Timer.time_passed(2000):
                    print(f"skipping lines in {os.path.basename(path_file):s} until reaching timestamp {current_range[0]:d}...")

                if line is None:
                    break

                timestamp_close = get_int(line, index_time_close)

            if line is None:
                break

            # current line fits, yield it, proceed to next range
            if current_range[0] < timestamp_close <= current_range[1]:
                stripped = line.strip()
                split = stripped.split("\t")
                yield {column: STATS_TYPES[i](split[i]) for column, i in zip(header, indices_columns)}
                index_next_range += 1

            # yield empties for timestamps not covered by data
            while current_range[1] < timestamp_close:
                time_open = current_range[1] - data_difference_timestamp
                time_close = current_range[1] - 1
                yield {
                    column: time_open if column == "open_time" else time_close if column == "close_time" else STATS_TYPES[i](False)
                    for column, i in zip(header, indices_columns)
                }
                index_next_range += 1
                current_range = ranges[index_next_range]
                if Timer.time_passed(2000):
                    print(f"filling gaps in {os.path.basename(path_file):s} until reaching timestamp {timestamp_close:d}...")

            if Timer.time_passed(2000):
                print(f"generated {index_next_range:d} of {no_ranges:d} continual data points for {os.path.basename(path_file):s}...")

    # fill rest
    for index_current_range in range(index_next_range, no_ranges):
        time_open = current_range[1] - data_difference_timestamp
        time_close = current_range[1] - 1
        yield {
            column: time_open if column == "time_open" else time_close if column == "time_close" else STATS_TYPES[i](False)
            for column, i in zip(header, indices_columns)
        }
        if Timer.time_passed(2000):
            print(f"finished filling {100. * index_current_range / (no_ranges - index_next_range):5.2f}% of final gaps...")


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


def get_pairs_from_filenames(paths_files: Collection[str]) -> Collection[Tuple[str, str]]:
    return tuple(
        (x[:-3], x[-3:])

        for x in (
            os.path.splitext(y)[0]

            for y in (
                os.path.basename(z)

                for z in paths_files
            )
        )
    )
