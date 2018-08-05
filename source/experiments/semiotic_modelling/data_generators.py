import datetime
import json
from math import sin, cos
from typing import Generator, Tuple, Sequence

from dateutil.tz import tzutc

from source.data.data_generation import series_generator
from source.experiments.semiotic_modelling.modelling import TIME, EXAMPLE


def debug_series() -> Generator[Tuple[TIME, Sequence[EXAMPLE]], None, None]:
    with open("../../../configs/time_series.json", mode="r") as file:
        config = json.load(file)

    base_symbol = "ETH"                                         # type: str
    asset_symbols = ("EOS", "SNT", "QTUM", "BNT")[:1]           # type: Tuple[str, ...]

    source_paths = (config["data_dir"] + "{:s}{:s}.csv".format(_x, base_symbol) for _x in asset_symbols)

    start = datetime.datetime.fromtimestamp(1501113780, tz=tzutc())  # maybe divide by 1000?
    end = datetime.datetime.fromtimestamp(1503712000, tz=tzutc())
    # end = datetime.datetime.fromtimestamp(1529712000, tz=tzutc())
    start_str, end_str = str(start), str(end)

    generators = [series_generator(_x, interval_minutes=1, start_time=start_str, end_time=end_str) for _x in source_paths]
    last_values = [0. for _ in asset_symbols]
    while True:
        time_stamps = set()
        examples = []
        for _i, each_generator in enumerate(generators):
            try:
                t, f = next(each_generator)
            except StopIteration as e:
                raise e
            time_stamps.add(t)
            each_example = last_values[_i], f
            examples.append(each_example)
            last_values[_i] = f

        assert len(time_stamps) == 1
        t, = time_stamps

        yield t, examples


def debug_trig() -> Generator[Tuple[TIME, Sequence[EXAMPLE]], None, None]:
    for t in range(100000):
        examples = [(sin(t / 10.), cos(t / 10.))]
        yield t, examples
