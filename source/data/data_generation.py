import random
import numpy


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
            rnd_value = random.gauss(0., .2 / 4.)
            rnd_value = 2. * factor * random.random() - factor + relative_bias * factor
            a = max(a + rnd_value / 100., .0)


def get_series(file_path, range_start=None, range_end=None, interval_minutes=1):
    print("Reading time series from {:s}...".format(file_path))
    series = []
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
                        raise ValueError("Source {} starts after {:s}!".format(file_path, str(range_start)))
                elif row_ts < end_timestamp:
                    continue

            if -1 < end_timestamp < row_ts:
                break

            close = float(row[4])
            series.append(close)

        if row_ts < end_timestamp:
            raise ValueError("Source {} ends before {:s}!".format(file_path, str(range_end)))

    return series
