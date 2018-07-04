import datetime
import json
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
                        raise ValueError("Source {} starts after {:s}!".format(file_path, str(range_start)))
                elif row_ts < end_timestamp:
                    continue

            if -1 < end_timestamp < row_ts:
                break

            close = float(row[4])
            yield datetime.datetime.utcfromtimestamp(row_ts), close

        if row_ts < end_timestamp:
            raise ValueError("Source {} ends before {:s}!".format(file_path, str(range_end)))


def DEBUG_SERIES(cur_a, cur_b="ETH"):
    with open("../../../configs/config.json", mode="r") as file:
        config = json.load(file)
    source_dir = config["data_dir"]     # "../../configs/23Jun2017-23Jun2018-1m/"
    target_dir = config["target_dir"]  # "../../results/dtw/2018-06-25/"
    interval_minutes = config["interval_minutes"]
    start_date = datetime.datetime.strptime(config["start_time"], "%Y-%m-%d_%H:%M:%S_%Z")
    end_date = datetime.datetime.strptime(config["end_time"], "%Y-%m-%d_%H:%M:%S_%Z")

    source_path = source_dir + "{:s}{:s}.csv".format(cur_a, cur_b)
    time_series = series_generator(source_path, range_start=start_date, range_end=end_date, interval_minutes=interval_minutes)
    return time_series


def main():
    print(len(list(DEBUG_SERIES("ADA"))))


if __name__ == "__main__":
    main()
