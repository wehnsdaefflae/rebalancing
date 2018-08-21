import json
import time
from math import sin, cos
from typing import Generator, Tuple, Sequence, TypeVar, Generic, Dict

from source.data.data_processing import series_generator, equisample

TIME = TypeVar("TIME")
INPUT_TYPE = TypeVar("INPUT_DEF")
OUTPUT_TYPE = TypeVar("OUTPUT_DEF")
EXAMPLE = Tuple[INPUT_TYPE, OUTPUT_TYPE]


class SequenceFactory(Generic[INPUT_TYPE, OUTPUT_TYPE]):
    def get_dimensions(self) -> Tuple[int, int]:
        raise NotImplementedError

    def get_generator(self) -> Generator[Tuple[TIME, EXAMPLE], None, None]:
        raise NotImplementedError


class TrigonometricSequence(SequenceFactory[Tuple[float], Tuple[float]]):
    def __init__(self, length: int):
        self.length = length

    def get_dimensions(self) -> Tuple[int, int]:
        return 1, 1

    def get_generator(self) -> Generator[Tuple[TIME, Tuple[Tuple[float], Tuple[float]]], None, None]:
        for t in range(self.length):
            # examples = [(sin(t / 100.), cos(t / 70.)*3. + sin(t/13.)*.7)]
            input_value = sin(t / 100.),
            target_value = float(cos(t / 100.) >= 0.) * 2. - 1.,
            examples = input_value, target_value
            yield t, examples


class ExchangeRateSequence(SequenceFactory[Tuple[float, ...], Tuple[float, ...]]):
    @staticmethod
    def _get_current_values(values: Dict[str, Tuple[TIME, float]]) -> Tuple[TIME, Dict[str, float]]:
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
        return t, {_s: _v for _s, (_, _v) in values.items()}

    def __init__(self, input_symbols: Tuple[str, ...], output_symbols: Tuple[str, ...], start_timestamp: int = -1, end_timestamp: int = -1):
        self.input_symbols = input_symbols
        self.output_symbols = output_symbols
        with open("../../../configs/time_series.json", mode="r") as file:
            config = json.load(file)

        data_dir = config["data_dir"]
        all_symbols = set(input_symbols) | set(output_symbols)
        self.source_paths = {_s: data_dir + "{:s}ETH.csv".format(_s) for _s in all_symbols}
        self.start_timestamp = start_timestamp
        self.end_timestamp = end_timestamp

        self.generators = {
            _s: equisample(series_generator(_x, start_timestamp=self.start_timestamp, end_timestamp=self.end_timestamp), target_delta=60)
            for _s, _x in self.source_paths.items()
        }

    def get_dimensions(self) -> Tuple[int, int]:
        return len(self.input_symbols), len(self.output_symbols)

    def get_generator(self) -> Generator[Tuple[TIME, Tuple[Tuple[float, ...], Tuple[float, ...]]], None, None]:
        input_values = tuple(0. for _ in self.input_symbols)
        last_target_values = tuple(0. for _ in self.output_symbols)

        while True:
            snapshots = {_s: next(each_generator) for _s, each_generator in self.generators.items()}
            t, symbol_values = ExchangeRateSequence._get_current_values(snapshots)

            target_values = tuple(symbol_values[_s] for _s in self.output_symbols)
            change = tuple(0. if _last == 0. else _this / _last - 1. for _last, _this in zip(last_target_values, target_values))

            yield t, (input_values, change)
            #yield t, (input_values, target_values)

            last_target_values = target_values
            input_values = tuple(symbol_values[_s] for _s in self.input_symbols)


def generator_testing():
    symbols = "EOS", "SNT", "QTUM", "BNT"                 # type: Tuple[str, ...]

    factory = ExchangeRateSequence(symbols[:], symbols[:1], start_timestamp=1501113780, end_timestamp=1532508240)
    gen = factory.get_generator()
    for each_time, each_ex in gen:
        print("{:s}:\t{:s}".format(str(each_time), str(each_ex)))
        time.sleep(.5)


if __name__ == "__main__":
    generator_testing()
