import datetime
import json
import time
from math import sin, cos
from typing import Generator, Tuple, Sequence, Iterable, TypeVar, Generic, Sized

from dateutil.tz import tzutc

from source.data.data_generation import series_generator
from source.experiments.semiotic_modelling.modelling import TIME, EXAMPLE

INPUT_DEF = TypeVar("INPUT_DEF")
OUTPUT_DEF = TypeVar("OUTPUT_DEF")


class SequentialExampleGeneratorFactory(Generic[INPUT_DEF, OUTPUT_DEF]):
    def __init__(self, input_definition: Iterable[INPUT_DEF], output_definition: Iterable[OUTPUT_DEF]):
        self.input_definition = input_definition        # type: Iterable[INPUT_DEF]
        self.output_definition = output_definition      # type: Iterable[OUTPUT_DEF]

    def get_generator(self) -> Generator[Tuple[TIME, Sequence[EXAMPLE]], None, None]:
        raise NotImplementedError


class ExchangeRateGeneratorFactory(SequentialExampleGeneratorFactory[str, str]):
    def __init__(self, input_definition: Iterable[str], output_definition: Iterable[str]):
        super().__init__(input_definition, output_definition)
        with open("../../../configs/time_series.json", mode="r") as file:
            config = json.load(file)

        self.data_dir = config["data_dir"]

    def get_generator(self) -> Generator[Tuple[TIME, Sequence[EXAMPLE]], None, None]:
        base_symbol = "ETH"                     # type: str
        all_symbols = set(self.input_definition) | set(self.output_definition)

        source_paths = {_s: self.data_dir + "{:s}{:s}.csv".format(_s, base_symbol) for _s in all_symbols}

        # here

        start = datetime.datetime.fromtimestamp(1501113780, tz=tzutc())
        end = datetime.datetime.fromtimestamp(1503712000, tz=tzutc())
        # end = datetime.datetime.fromtimestamp(1529712000, tz=tzutc())
        start_str, end_str = str(start), str(end)

        generators = {_s: series_generator(_x, interval_minutes=1, start_time=start_str, end_time=end_str) for _s, _x in source_paths.items()}
        inputs = tuple(0. for _ in input_symbols)


def debug_multiple_inputs() -> Generator[Tuple[TIME, Sequence[EXAMPLE]], None, None]:
    with open("../../../configs/time_series.json", mode="r") as file:
        config = json.load(file)

    base_symbol = "ETH"                                         # type: str
    input_symbols = "EOS", "SNT", "QTUM", "BNT"                 # type: Tuple[str, ...]
    target_symbols = "EOS", "SNT"  # , "QTUM", "BNT"
    all_symbols = set(input_symbols) | set(target_symbols)

    source_paths = {_s: config["data_dir"] + "{:s}{:s}.csv".format(_s, base_symbol) for _s in all_symbols}

    start = datetime.datetime.fromtimestamp(1501113780, tz=tzutc())
    end = datetime.datetime.fromtimestamp(1503712000, tz=tzutc())
    # end = datetime.datetime.fromtimestamp(1529712000, tz=tzutc())
    start_str, end_str = str(start), str(end)

    generators = {_s: series_generator(_x, interval_minutes=1, start_time=start_str, end_time=end_str) for _s, _x in source_paths.items()}
    inputs = tuple(0. for _ in input_symbols)
    while True:
        values = {_s: next(each_generator) for _s, each_generator in generators.items()}
        time_set = set(t for t, v in values.values())
        assert len(time_set) == 1
        t, = time_set

        target_values = [values[_s] for _s in target_symbols]
        yield t, [(inputs, each_target) for _, each_target in target_values]

        input_values = [values[_s] for _s in input_symbols]
        inputs = tuple(each_input for _, each_input in input_values)


def debug_multiple_states() -> Generator[Tuple[TIME, Sequence[EXAMPLE]], None, None]:
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
    for t in range(50000):
        # examples = [(sin(t / 100.), cos(t / 70.)*3. + sin(t/13.)*.7)]
        examples = [((sin(t / 100.), ), cos(t / 100.))]
        yield t, examples


if __name__ == "__main__":
    gen = debug_multiple_inputs()
    for each_time, each_ex in gen:
        print("{:s}:\t{:s}".format(str(each_time), str(each_ex)))
        time.sleep(.5)
