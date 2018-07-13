import datetime
import json
import random
import numpy
from dateutil import parser
from dateutil.tz import tzutc


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


def series_generator(file_path, range_start=None, range_end=None, interval_minutes=1):
    print("Reading time series from {:s}...".format(file_path))
    start_timestamp = -1 if range_start is None else int(range_start.timestamp())
    end_timestamp = -1 if range_end is None else int(range_end.timestamp())
    with open(file_path, mode="r") as file:
        row_ts = -1
        for i, line in enumerate(file):
            if i % interval_minutes != 0:
                continue
            row = line[:-1].split("\t")
            row_ts = int(row[0]) / 1000
            if -1 < start_timestamp:
                if start_timestamp < row_ts:
                    if i < 1:
                        first_date = datetime.datetime.fromtimestamp(row_ts, tz=tzutc())
                        msg = "Source {:s} starts after {:s} (at {:s})!"
                        raise ValueError(msg.format(file_path, str(range_start), str(first_date)))
                elif row_ts < end_timestamp:
                    continue

            if -1 < end_timestamp < row_ts:
                break

            close = float(row[4])
            yield datetime.datetime.fromtimestamp(row_ts, tz=tzutc()), close

        if row_ts < end_timestamp:
            last_date = datetime.datetime.fromtimestamp(row_ts, tz=tzutc())
            msg = "Source {:s} ends before {:s} (at {:s})!"
            raise ValueError(msg.format(file_path, str(range_end), str(last_date)))


def DEBUG_SERIES(cur_a, cur_b, data_dir, interval_minutes, start_time, end_time):
    start_date = parser.parse(start_time)  # datetime.datetime.strptime(start_time, "%Y-%m-%d_%H:%M:%S_%Z")
    end_date = parser.parse(end_time)  # datetime.datetime.strptime(end_time, "%Y-%m-%d_%H:%M:%S_%Z")

    source_path = data_dir + "{:s}{:s}.csv".format(cur_a, cur_b)
    time_series = series_generator(source_path, range_start=start_date, range_end=end_date, interval_minutes=interval_minutes)
    return time_series


def main():
    print(len(list(DEBUG_SERIES("ADA"))))


if __name__ == "__main__":
    main()
