import sys

from matplotlib import pyplot

from source.main import absolute_brownian
import random

# random.seed(1)


def get_table(series_a, series_b):
    # assert(len(series_a) == len(series_b))
    table = [[sys.float_info.max if i == 0 or j == 0 else 0. for i in range(len(series_b))] for j in range(len(series_a))]
    d_table = [["X" if i == j == 0 else "." for i in range(len(series_b))] for j in range(len(series_a))]
    table[0][0] = .0
    for i in range(1, len(series_a)):
        row = table[i]
        d_row = d_table[i]
        for j in range(1, len(series_b)):
            d = abs(series_a[i] - series_b[j])
            x, y, z = table[i-1][j], row[j-1], table[i-1][j-1]
            min_i, min_v = min(enumerate([x, y, z]), key=lambda x_v: x_v[1])
            row[j] = d + min_v
            d_row[j] = ["|", "-", "\\"][min_i]
    return table, d_table


def get_path(table):
    path = []
    i, j = len(table) - 1, len(table[0]) - 1
    while 0 < i or 0 < j:
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
    for each_d in path:
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


if __name__ == "__main__":
    g = absolute_brownian(initial=1., factor=2.)
    _a = [next(g) for _ in range(1100)]
    a = _a[:1000]
    g = absolute_brownian(initial=.9, factor=1.)
    b = [x - .1 for x in _a][100:]
    #b = [next(g) for _ in range(1000)]

    d_a = [1. - a[i] / a[i+1] for i in range(len(a) - 1)]
    d_b = [1. - b[i] / b[i+1] for i in range(len(b) - 1)]

    print(", ".join(["{:08.5f}".format(x) for x in a]))
    print(", ".join(["{:08.5f}".format(x) for x in b]))
    t, d_t = get_table(d_a, d_b)
    p = get_path(t)
    print(p)
    print()
    s_a, s_b = stretch(a[:-1], b[:-1], p)
    print(", ".join(["{:08.5f}".format(x) for x in s_a]))
    print(", ".join(["{:08.5f}".format(x) for x in s_b]))
    print()
    print(sum(p))

    f, (ax1, ax2, ax3) = pyplot.subplots(3, sharex="all")
    ax1.plot(a, label="a")
    ax1.plot(b, label="b")
    ax1.legend()
    ax1.set_title("original")
    ax2.plot(s_a)
    ax2.plot(s_b)
    ax2.set_title("fitted")
    temporal_tendency = [sum(p[0:i]) for i in range(len(p))]
    ax3.plot(temporal_tendency)
    total_tendency = sum(temporal_tendency) / len(temporal_tendency)
    print("b is {:.2f} ahead of a".format(total_tendency))
    pyplot.show()
