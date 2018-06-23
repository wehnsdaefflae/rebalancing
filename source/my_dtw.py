import datetime
import sys
from math import sin, cos

from matplotlib import pyplot

import numpy
from source.main import absolute_brownian, data_generator
import random

random.seed(133711)


def get_series(file_path, range_start=-1, range_end=-1):
    series = []
    with open(file_path, mode="r") as file:
        row_ts = -1
        for i, line in enumerate(file):
            row = line[:-1].split("\t")
            row_ts = int(row[0]) / 1000
            if -1 < range_start:
                if range_start < row_ts:
                    if i < 1:
                        raise ValueError("Source {} does not support range_start!".format(file_path))
                elif row_ts < range_end:
                    continue

            if -1 < range_end < row_ts:
                break

            close = float(row[4])
            series.append(close)

        if row_ts < range_end:
            raise ValueError("Source {} does not support range_end!".format(file_path))

    return series


def get_table(s_0, s_1, mode="default", distance=lambda v1, v2: (v1 - v2) ** 2):
    l_a, l_b = len(s_0), len(s_1)
    if mode == "derivative":
        series_a = [0.] + [-1. if 0. >= s_0[i+1] else 1. - s_0[i] / s_0[i+1] for i in range(l_a - 1)]
        series_b = [0.] + [-1. if 0. >= s_1[i+1] else 1. - s_1[i] / s_1[i+1] for i in range(l_b - 1)]
    elif mode == "normalized":
        max_a, min_a = max(s_0), min(s_0)
        series_a = [(x - min_a) / (max_a - min_a) for x in s_0]
        max_b, min_b = max(s_1), min(s_1)
        series_b = [(x - min_b) / (max_b - min_b) for x in s_1]
    else:
        series_a = s_0[:]
        series_b = s_1[:]

    """
    # https://docs.scipy.org/doc/numpy/user/quickstart.html
    table = numpy.zeros((l_a + 1, l_b + 1))
    table[0, 1:l_b] = [sys.float_info.max for _ in range(l_b)]
    table[1:l_a, 0] = [sys.float_info.max for _ in range(l_a)]
    print("shape: {}".format(table.shape))
    """
    table = [[sys.float_info.max if i == 0 or j == 0 else 0. for i in range(l_b + 1)] for j in range(l_a + 1)]
    table[0][0] = .0
    # """

    factor = 1.

    for i in range(l_a):
        row = table[i + 1]
        # print("finished {:05.2f}%".format(100. * i / (len(series_a) - 1)))
        for j in range(l_b):
            d = distance(series_a[i], series_b[j])
            row[j + 1] = d + min(table[i][j + 1] * factor, row[j] * factor, table[i][j])
            #print("{}, {}: {}".format(i, j, d))
            #print(table_str([[-1.] + series_a] + [[series_b[idx]] + table[idx] for idx in range(l_b)]))
            #print()
    return table


def get_path(table):
    path = []
    i, j = len(table) - 1, len(table[0]) - 1
    while 1 < i or 1 < j:
        x, y, z = table[i-1][j], table[i][j-1], table[i-1][j-1]
        min_i, min_v = min(enumerate([x, y, z]), key=lambda x_v: x_v[1])
        path.append([1, -1, 0][min_i])
        i, j = [(i-1, j), (i, j-1), (i-1, j-1)][min_i]
    return path[::-1]


def table_str(table):
    format_str = "\t".join(["{:08.5f}" for _ in table[0]])
    rows = [format_str.format(*[-1. if v == sys.float_info.max else v for v in x]) for x in table]
    return "\n".join(rows)


def stretch(series_a, series_b, path):
    new_a, new_b = [series_a[0]], [series_b[0]]
    i, j = 0, 0
    for i_dir, each_d in enumerate(path):
        if each_d == 0:
            i += 1
            j += 1
        elif each_d == 1:
            i += 1
        elif each_d == -1:
            j += 1
        new_a.append(series_a[i])
        new_b.append(series_b[j])
    return new_a, new_b


def plot_series(series_a, series_b, path, file_path=None):
    f, (ax1, ax2, ax3) = pyplot.subplots(3, sharex="all")

    ax1.plot(series_a, color="#1f77b4")
    ax1.set_ylabel("series_a", color="#1f77b4")
    ax11 = ax1.twinx()
    ax11.plot(series_b, color="#ff7f0e")
    ax11.set_ylabel("series_b", color="#ff7f0e")
    ax1.set_title("original")

    s_a, s_b = stretch(series_a, series_b, path)
    ax2.plot(s_a, color="#1f77b4")
    ax2.set_ylabel("a")
    ax21 = ax2.twinx()
    ax21.plot(s_b, color="#ff7f0e")
    ax21.set_ylabel("b")
    ax2.set_title("fitted")

    tendency = [sum(path[0:i]) for i in range(len(path))]
    ax3.plot(tendency)
    if file_path is not None:
        pyplot.savefig(file_path)
    else:
        pyplot.show()
    pyplot.clf()
    pyplot.close()


if __name__ == "__main__":
    """
    start_date = datetime.datetime(2018, 5, 20, 0, 0, 0, tzinfo=datetime.timezone.utc)
    # end_date = datetime.datetime(2018, 6, 20, 0, 0, 0, tzinfo=datetime.timezone.utc)
    end_date = datetime.datetime(2018, 5, 27, 0, 0, 0, tzinfo=datetime.timezone.utc)

    timestamp_start, timestamp_end = int(start_date.timestamp()), int(end_date.timestamp())
    fp_a = "../data/binance/20May2018-20Jun2018-1m/{}.csv".format("ADAETH")
    fp_b = "../data/binance/20May2018-20Jun2018-1m/{}.csv".format("ADXETH")
    a = get_series(fp_a, range_start=timestamp_start, range_end=timestamp_end)[7500:]
    b = get_series(fp_b, range_start=timestamp_start, range_end=timestamp_end)[:-7500]

    ""
    g = absolute_brownian(initial=1., factor=2., relative_bias=.1)
    _a = [next(g) for _ in range(11)]
    a = _a[:10]
    b = [x - .1 for x in _a][1:]

    """
    a = [sin(x/7.) for x in range(1000)]
    b = [cos(x/11.)/3 for x in range(1000)]
    #"""

    print("a: {}, b: {} ".format(len(a), len(b)))
    t = get_table(a, b, mode="normalized", distance=lambda _x, _y: abs(_x - _y))  # , distance=lambda v1, v2: abs(v1-v2))
    p = get_path(t)

    temporal_tendency = [sum(p[0:i]) for i in range(len(p))]
    total_tendency = [temporal_tendency[i] for i in range(len(temporal_tendency)) if p[i] == 0]
    ahead = 0. if len(total_tendency) < 1 else sum(total_tendency) / len(total_tendency)
    print("b is on average {:.2f} ahead of a".format(ahead))
    print("deviation: {:.2f}".format(t[-1][-1]))

    plot_series(a, b, p)


