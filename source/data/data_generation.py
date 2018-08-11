import datetime
import json
import os
import random
from math import sin, cos
from typing import Generator, Tuple, Iterator, Sequence

import numpy
from dateutil import parser
from dateutil.tz import tzutc
from matplotlib import pyplot

from source.experiments.semiotic_modelling.modelling import TIME, EXAMPLE
from source.tools.timer import Timer


def relative_brownian(initial=1., drift=.0, volatility=.01):  # relative normally distributed change
    a = initial
    while True:
        yield a
        # r = random.random()
        r = numpy.random.random()
        a = a * (1. + drift + volatility * (r - .5))


def absolute_brownian(initial=1., factor=1., relative_bias=0.):  # constant equiprobable change
    a = initial
    while True:
        yield a
        if 0. < a:
            # numpy.random.standard_cauchy
            rnd_value = random.gauss(0., .2 / 4.)
            rnd_value = 2. * factor * random.random() - factor + relative_bias * factor
            a = max(a + rnd_value / 100., .0)


def series_generator(file_path: str, start_timestamp: int = -1, end_timestamp: int = -1) -> Generator[Tuple[float, float], None, None]:
    print("Reading time series from {:s}...".format(file_path))

    with open(file_path, mode="r") as file:
        row_ts = -1
        for i, line in enumerate(file):
            row = line.strip().split("\t")
            row_ts = float(row[0]) / 1000.
            if -1 < start_timestamp:
                if start_timestamp < row_ts:
                    if i < 1:
                        first_date = datetime.datetime.fromtimestamp(row_ts, tz=tzutc())
                        start_time = datetime.datetime.fromtimestamp(start_timestamp, tz=tzutc())
                        msg = "Source {:s} starts after {:s} (at {:s})!"
                        raise ValueError(msg.format(file_path, start_time, str(first_date)))
                elif row_ts < end_timestamp:
                    continue

            if -1 < end_timestamp < row_ts:
                break

            close = float(row[4])
            yield row_ts, close

        if row_ts < end_timestamp:
            last_date = datetime.datetime.fromtimestamp(row_ts, tz=tzutc())
            end_time = datetime.datetime.fromtimestamp(end_timestamp, tz=tzutc())
            msg = "Source {:s} ends before {:s} (at {:s})!"
            raise ValueError(msg.format(file_path, end_time, str(last_date)))


def equisample(iterator: Iterator[Tuple[float, float]], target_delta: float) -> Generator[Tuple[float, float], None, None]:
    assert 0 < target_delta
    last_time = -1
    last_value = 0.
    for time_stamp, value in iterator:
        difference = time_stamp - last_time

        if difference < target_delta:
            continue

        elif difference == target_delta or last_time < 0:
            assert last_time < 0 or time_stamp == last_time + target_delta
            yield time_stamp, value

            last_value = value
            last_time = time_stamp

        else:
            value_change = (value - last_value) / difference
            no_intermediate_steps = round(difference // target_delta)
            for each_step in range(no_intermediate_steps):
                last_value += value_change
                last_time += target_delta
                yield last_time, last_value


def debug_trig() -> Generator[Tuple[TIME, Sequence[EXAMPLE]], None, None]:
    for t in range(50000):
        # examples = [(sin(t / 100.), cos(t / 70.)*3. + sin(t/13.)*.7)]
        examples = [((sin(t / 100.), ), cos(t / 100.))]
        yield t, examples


def time_stamp_test():
        with open("../../configs/time_series.json", mode="r") as file:
            config = json.load(file)

        start_ts = 1501113780
        #start_ts = 1521113780
        # end_ts = 1502508240
        end_ts = 1532508240

        data_dir = config["data_dir"]
        file_names = ["BNTETH.csv", "SNTETH.csv", "QTUMETH.csv", "EOSETH.csv"]
        # file_names = os.listdir(data_dir)[-10:]

        delta = 60

        time_axes = tuple([] for _ in file_names)
        value_axes = tuple([] for _ in file_names)
        for _i, each_file in enumerate(file_names):
            file_path = data_dir + each_file
            last_t = -1
            for t, v in equisample(series_generator(file_path, start_timestamp=start_ts, end_timestamp=end_ts), target_delta=delta):
                if 0 < last_t:
                    if not t - last_t == delta:
                        print("skip")
                last_t = t
                time_axes[_i].append(t)
                value_axes[_i].append(v)
                if Timer.time_passed(2000):
                    print("{:5.2f}% of file {:d}/{:d}".format(100. * (t - start_ts) / (end_ts - start_ts), _i + 1, len(file_names)))

        for _i, (each_time, each_value) in enumerate(zip(time_axes, value_axes)):
            pyplot.plot(each_time, each_value, label=file_names[_i])

        pyplot.legend()
        pyplot.show()

        for _i, each_axis in enumerate(time_axes):
            for _j in range(_i + 1, len(time_axes)):
                print("{:s} and {:s}: {:s}".format(file_names[_i], file_names[_j], str(each_axis == time_axes[_j])))


if __name__ == "__main__":
    time_stamp_test()
