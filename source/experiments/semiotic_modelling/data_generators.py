import datetime
import json
import time
from math import sin, cos
from typing import Generator, Tuple, Sequence, TypeVar, Generic

from matplotlib import pyplot

from source.data.data_generation import series_generator, equisample
from source.experiments.semiotic_modelling.modelling import TIME, EXAMPLE

INPUT_DEF = TypeVar("INPUT_DEF")
OUTPUT_DEF = TypeVar("OUTPUT_DEF")


class SequentialExampleGeneratorFactory(Generic[INPUT_DEF, OUTPUT_DEF]):
    def __init__(self, input_definition: Tuple[INPUT_DEF, ...], output_definition: Tuple[OUTPUT_DEF, ...]):
        self.input_definition = input_definition        # type: Tuple[INPUT_DEF, ...]
        self.output_definition = output_definition      # type: Tuple[OUTPUT_DEF, ...]

    def get_generator(self) -> Generator[Tuple[TIME, Sequence[EXAMPLE]], None, None]:
        raise NotImplementedError


class SingularTrigonometryGeneratorFactory(SequentialExampleGeneratorFactory[str, str]):
    def __init__(self, length: int):
        super().__init__(("sin", ), ("cos", ))
        self.length = length

    def get_generator(self) -> Generator[Tuple[TIME, Sequence[EXAMPLE]], None, None]:
        for t in range(self.length):
            # examples = [(sin(t / 100.), cos(t / 70.)*3. + sin(t/13.)*.7)]
            examples = [((sin(t / 100.), ), cos(t / 100.))]
            yield t, examples


class SingularExchangeRateGeneratorFactory(SequentialExampleGeneratorFactory[str, str]):
    def __init__(self, input_definition: Tuple[str, ...], output_definition: Tuple[str, ...], length: int = -1):
        super().__init__(input_definition, output_definition)
        with open("../../../configs/time_series.json", mode="r") as file:
            config = json.load(file)

        data_dir = config["data_dir"]
        base_symbol = "ETH"                     # type: str
        all_symbols = set(self.input_definition) | set(self.output_definition)
        self.source_paths = {_s: data_dir + "{:s}{:s}.csv".format(_s, base_symbol) for _s in all_symbols}

        self.length = length

        self.start_ts = 1501113780
        self.end_ts = 1532508240

    def get_generator(self) -> Generator[Tuple[TIME, Sequence[EXAMPLE]], None, None]:
        generators = {_s: equisample(series_generator(_x, start_timestamp=self.start_ts, end_timestamp=self.end_ts), target_delta=60)
                      for _s, _x in self.source_paths.items()}

        inputs = tuple(0. for _ in self.input_definition)
        last_target_values = tuple(0. for _ in self.output_definition)

        iterations = 0

        while True:
            values = {_s: next(each_generator) for _s, each_generator in generators.items()}
            time_set = set(t for t, v in values.values())
            if len(time_set) == 1:
                t, = time_set
            else:
                max_time = max(time_set)
                delta = max_time - min(time_set)
                if delta >= 60.:
                    for each_symbol, each_value in values.items():
                        print("{:s}\t{:s}".format(str(each_symbol), str(each_value)))
                    raise ValueError("time stamps more than 60 seconds apart")
                else:
                    t = max_time

            target_values = [values[_s][-1] for _s in self.output_definition]

            change = tuple(0. if _last == 0. else _this / _last - 1. for _last, _this in zip(last_target_values, target_values))
            yield t, [(inputs, each_change) for each_change in change]
            #yield t, [(inputs, each_target) for each_target in target_values]

            iterations += 1
            if 0 < self.length <= iterations:
                raise StopIteration

            last_target_values = target_values
            input_values = [values[_s][-1] for _s in self.input_definition]
            inputs = tuple(input_values)


def generator_testing():
    symbols = "EOS", "SNT", "QTUM", "BNT"                 # type: Tuple[str, ...]
    factory = SingularExchangeRateGeneratorFactory(symbols[:1], symbols[:1])
    gen = factory.get_generator()
    for each_time, each_ex in gen:
        print("{:s}:\t{:s}".format(str(each_time), str(each_ex)))
        time.sleep(.5)


if __name__ == "__main__":
    generator_testing()
