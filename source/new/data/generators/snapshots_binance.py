from __future__ import annotations

import glob
from typing import Tuple, Sequence, Generator, Optional, Iterable

from source.new.config import PATH_DIRECTORY_DATA
from source.new.data._abstract import SNAPSHOT
from source.new.data.tools import generator_file, get_timestamp_close_boundaries, get_pairs_from_filenames


def merge_generator(
        pairs: Optional[Iterable[Tuple[str, str]]] = None,
        timestamp_range: Optional[Tuple[int, int]] = None,
        interval_minutes: int = 1,
        header: Sequence[str] = ("close_time", "close", ),
        directory_data: str = PATH_DIRECTORY_DATA) -> Generator[SNAPSHOT, None, None]:

    directory_csv = directory_data + "binance/"
    if pairs is None:
        files = sorted(glob.glob(f"{directory_csv:s}*.csv"))
        pairs = get_pairs_from_filenames(files)

    else:
        files = sorted(f"{directory_csv:s}{each_pair[0].upper():s}{each_pair[-1].upper():s}.csv" for each_pair in pairs)

    if timestamp_range is None:
        print(f"determining timestamp boundaries...")
        timestamp_range = get_timestamp_close_boundaries(files)
        print(f"timestamp start {timestamp_range[0]:d}, timestamp end {timestamp_range[1]:d}")

    else:
        assert timestamp_range[0] < timestamp_range[1]

    header = ("close_time", ) + tuple(header) if "close_time" not in header else header
    generators_all = tuple(generator_file(each_file, timestamp_range, interval_minutes, header) for each_file in files)

    names_pair = tuple(f"{each_pair[0].upper():s}-{each_pair[1].upper():s}" for each_pair in pairs)
    for snapshots in zip(*generators_all):
        d = {}
        for i, each_snapshot in enumerate(snapshots):
            if i < 1:
                d["close_time"] = each_snapshot["close_time"]
            d.update({f"{names_pair[i]:s}_{k:s}": v for k, v in each_snapshot.items()})

        yield d
