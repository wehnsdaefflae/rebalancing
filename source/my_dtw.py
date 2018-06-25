import datetime
import math
import sys
from math import sin, cos

from matplotlib import pyplot
from scipy import stats

import numpy
from source.main import absolute_brownian, data_generator
import random

random.seed(133711)


def get_series(file_path, range_start=-1, range_end=-1, interval_minutes=1):
    series = []
    with open(file_path, mode="r") as file:
        row_ts = -1
        for i, line in enumerate(file):
            if i % interval_minutes != 0:
                continue
            row = line[:-1].split("\t")
            row_ts = int(row[0]) / 1000
            if -1 < range_start:
                if range_start < row_ts:
                    if i < 1:
                        raise ValueError("Source {} does not support range_start: {:d}!".format(file_path, range_start))
                elif row_ts < range_end:
                    continue

            if -1 < range_end < row_ts:
                break

            close = float(row[4])
            series.append(close)

        if row_ts < range_end:
            raise ValueError("Source {} does not support range_end: {:d}!".format(file_path, range_end))

    return series


def get_table(s_0, s_1, derivative=False, normalized=True, overlap=True, diag_factor=1., distance=lambda v1, v2: (v1 - v2) ** 2):
    l_a, l_b = len(s_0), len(s_1)
    if normalized:
        #max_a, min_a = max(s_0), min(s_0)
        #series_a = [(x - min_a) / (max_a - min_a) for x in s_0]
        #max_b, min_b = max(s_1), min(s_1)
        #series_b = [(x - min_b) / (max_b - min_b) for x in s_1]
        series_a = stats.zscore(s_0)
        series_b = stats.zscore(s_1)

    else:
        series_a = s_0[:]
        series_b = s_1[:]

    if derivative:
        series_a = [0.] + [0. if 0. >= series_a[i+1] else 1. - series_a[i] / series_a[i+1] for i in range(l_a - 1)]
        series_b = [0.] + [0. if 0. >= series_b[i+1] else 1. - series_b[i] / series_b[i+1] for i in range(l_b - 1)]


    """
    # https://docs.scipy.org/doc/numpy/user/quickstart.html
    table = numpy.zeros((l_a + 1, l_b + 1))
    table[0, 1:l_b] = [sys.float_info.max for _ in range(l_b)]
    table[1:l_a, 0] = [sys.float_info.max for _ in range(l_a)]
    print("shape: {}".format(table.shape))
    """
    table = [[0. for _ in range(l_b)] for _ in range(l_a)]
    # """

    for i in range(l_a):
        row = table[i]
        # print("finished {:05.2f}%".format(100. * i / (len(series_a) - 1)))
        for j in range(l_b):
            dist = distance(series_a[i], series_b[j])
            if j == i == 0 or (overlap and (i == 0 or j == 0)):
                row[j] = dist
            elif not overlap and j == 0:
                row[j] = dist + table[i-1][j]
            elif not overlap and i == 0:
                row[j] = dist + row[j-1]
            else:
                row[j] = dist + min(table[i-1][j], row[j-1], table[i-1][j-1] * diag_factor)
            #print("{}, {}: {}".format(i, j, d))
            #print(table_str([[-1.] + series_a] + [[series_b[idx]] + table[idx] for idx in range(l_b)]))
            #print()
    return table


def get_path(table, overlap=False):
    if overlap:
        last_row = {(len(table) - 1, _j) for _j in range(1, len(table[0]))}
        last_col = {(_i, len(table[0]) - 1) for _i in range(1, len(table))}
        i, j = min(last_row | last_col, key=lambda ij: table[ij[0]][ij[1]])
    else:
        i, j = len(table) - 1, len(table[0]) - 1

    fork = [(-1, 0), (0, -1), (-1, -1)]
    path = [(i, j)]

    while (overlap and not any(0 == x for x in (i, j))) or (not overlap and (i, j) != (0, 0)):
        if 0 < i and 0 < j:
            d_i, d_j = min(fork, key=lambda _ij: table[i+_ij[0]][j+_ij[1]])
        elif 0 < i:
            d_i, d_j = -1, 0
        else:
            d_i, d_j = 0, -1
        i += d_i
        j += d_j
        path.append((i, j))

    return path[::-1]


def table_str(table):
    format_str = "\t".join(["{:08.5f}" for _ in table[0]])
    rows = [format_str.format(*[-1. if v == sys.float_info.max else v for v in x]) for x in table]
    return "\n".join(rows)


def fit(series_a, series_b, path):
    fit_a, fit_b = [], []
    for i, j in path:
        fit_a.append(series_a[i])
        fit_b.append(series_b[j])
    return fit_a, fit_b


def plot_series(series_a, series_b, path, a_label="series a", b_label="series b", file_path=None):
    _, (ax1, ax2, ax3) = pyplot.subplots(3, sharex="all")

    ax1.set_title("original")
    ax11 = ax1.twinx()
    ax2.set_title("fitted")
    ax21 = ax2.twinx()
    ax3.set_title("path")

    ax1.plot(series_a, color="#1f77b4")
    ax1.set_ylabel(a_label, color="#1f77b4")
    ax11.plot(series_b, color="#ff7f0e")
    ax11.set_ylabel(b_label, color="#ff7f0e")

    start_offset = path[0]
    end_offset = path[-1]

    a_start = series_a[:start_offset[0]]
    a_mid = series_a[start_offset[0]:end_offset[0] + 1]
    a_end = series_a[end_offset[0] + 1:]

    b_start = series_b[:start_offset[1]]
    b_mid = series_b[start_offset[1]:end_offset[1] + 1]
    b_end = series_b[end_offset[1] + 1:]

    a_fit, b_fit = fit(a_mid, b_mid, [(_i - len(a_start), _j - len(b_start)) for _i, _j in path])

    fitted_a = a_start + a_fit + a_end
    fitted_b = b_start + b_fit + b_end

    ax2.plot(range(len(b_start), len(b_start) + len(fitted_a)), fitted_a, color="#1f77b4", alpha=.5)
    ax2.set_ylabel(a_label)

    ax21.plot(range(len(a_start), len(a_start) + len(fitted_b)), fitted_b, color="#ff7f0e", alpha=.5)
    ax21.set_ylabel(b_label)

    def _diff_path(_path):
        _diff = []
        for _index in range(len(_path) - 2):
            _last, _this = _path[_index:_index + 2]
            _dir = _last[0] - _this[0], _last[1] - _this[1]
            _diff.append(_dir)
        return _diff

    diff_path = _diff_path(path)
    acc_path = [_i - _j for _i, _j in diff_path]
    tendency = [sum(acc_path[:i]) for i in range(len(acc_path))]
    ax3.plot(range(len(a_start) + len(b_start), len(a_start) + len(b_start) + len(tendency)), tendency)

    pyplot.tight_layout()
    if file_path is not None:
        pyplot.savefig(file_path)
    else:
        pyplot.show()
    pyplot.clf()
    pyplot.close()


if __name__ == "__main__":
    #"""
    start_date = datetime.datetime(2018, 3, 1, 0, 0, 0, tzinfo=datetime.timezone.utc)
    # end_date = datetime.datetime(2018, 6, 20, 0, 0, 0, tzinfo=datetime.timezone.utc)
    end_date = datetime.datetime(2018, 6, 1, 0, 0, 0, tzinfo=datetime.timezone.utc)

    timestamp_start, timestamp_end = int(start_date.timestamp()), int(end_date.timestamp())
    fp_a = "../data/binance/23Jun2017-23Jun2018-1m/{}.csv".format("NEOETH")
    fp_b = "../data/binance/23Jun2017-23Jun2018-1m/{}.csv".format("QTUMETH")
    full_a = get_series(fp_a, range_start=timestamp_start, range_end=timestamp_end, interval_minutes=60)
    full_b = get_series(fp_b, range_start=timestamp_start, range_end=timestamp_end, interval_minutes=60)
    a = full_a  # full_a[200:]
    b = full_b  # [_x * .9 for _x in full_a[:-200]]
    print("a: {}, b: {} ".format(len(a), len(b)))

    """
    g = absolute_brownian(initial=1., factor=2., relative_bias=.1)
    a = [next(g) for _ in range(10)]
    b = [x - .1 for x in a][3:8]

    ""
    a = [sin(x/7.) for x in range(1000)]
    b = [cos(x/11.)/3 for x in range(1000)]
    #"""

    o = True
    n = True
    d = False
    f = .9  # 1. / math.sqrt(2)
    t = get_table(a, b, normalized=n, derivative=d, overlap=o, diag_factor=f, distance=lambda _x, _y: (_x - _y) ** 2.)
    p = get_path(t, overlap=o)

    #print("\t".join(["{:08.5}".format(_x) for _x in a]))
    #print("\t".join(["{:08.5}".format(_x) for _x in b]))
    #print(table_str(t))
    #print(p)

    #temporal_tendency = [sum(p[0:i]) for i in range(len(p))]
    #total_tendency = [temporal_tendency[i] for i in range(len(temporal_tendency)) if p[i] == 0]
    #ahead = 0. if len(total_tendency) < 1 else sum(total_tendency) / len(total_tendency)
    #print("b is on average {:.2f} ahead of a".format(ahead))
    #print("deviation: {:.2f}".format(t[-1][-1]))

    plot_series(a, b, p)


