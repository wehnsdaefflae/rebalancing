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
    def __init__(self, input_definition: Tuple[INPUT_DEF, ...], output_definition: Tuple[OUTPUT_DEF, ...]):
        self.input_definition = input_definition        # type: Tuple[INPUT_DEF, ...]
        self.output_definition = output_definition      # type: Tuple[OUTPUT_DEF, ...]

    def get_generator(self) -> Generator[Tuple[TIME, Sequence[EXAMPLE]], None, None]:
        raise NotImplementedError


class ExchangeRateGeneratorFactory(SequentialExampleGeneratorFactory[str, str]):
    def __init__(self, input_definition: Tuple[str, ...], output_definition: Tuple[str, ...]):
        super().__init__(input_definition, output_definition)
        with open("../../../configs/time_series.json", mode="r") as file:
            config = json.load(file)

        data_dir = config["data_dir"]
        base_symbol = "ETH"                     # type: str
        all_symbols = set(self.input_definition) | set(self.output_definition)
        self.source_paths = {_s: data_dir + "{:s}{:s}.csv".format(_s, base_symbol) for _s in all_symbols}

        start = datetime.datetime.fromtimestamp(1501113780, tz=tzutc())
        # end = datetime.datetime.fromtimestamp(1503712000, tz=tzutc())
        end = datetime.datetime.fromtimestamp(1529712000, tz=tzutc())
        self.start_str, self.end_str = str(start), str(end)

    def get_generator(self) -> Generator[Tuple[TIME, Sequence[EXAMPLE]], None, None]:
        generators = {_s: series_generator(_x, interval_minutes=1, start_time=self.start_str, end_time=self.end_str)
                      for _s, _x in self.source_paths.items()}

        inputs = tuple(0. for _ in self.input_definition)

        while True:
            values = {_s: next(each_generator) for _s, each_generator in generators.items()}
            time_set = set(t for t, v in values.values())
            assert len(time_set) == 1
            t, = time_set

            target_values = [values[_s] for _s in self.output_definition]
            yield t, [(inputs, each_target) for _, each_target in target_values]

            input_values = [values[_s] for _s in self.input_definition]
            inputs = tuple(each_input for _, each_input in input_values)


def debug_trig() -> Generator[Tuple[TIME, Sequence[EXAMPLE]], None, None]:
    for t in range(50000):
        # examples = [(sin(t / 100.), cos(t / 70.)*3. + sin(t/13.)*.7)]
        examples = [((sin(t / 100.), ), cos(t / 100.))]
        yield t, examples


if __name__ == "__main__":
    symbols = "EOS", "SNT", "QTUM", "BNT"                 # type: Tuple[str, ...]
    factory = ExchangeRateGeneratorFactory(symbols[:1], symbols[:1])
    gen = factory.get_generator()
    for each_time, each_ex in gen:
        print("{:s}:\t{:s}".format(str(each_time), str(each_ex)))
        time.sleep(.5)
