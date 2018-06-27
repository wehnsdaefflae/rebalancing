import sys

from matplotlib import pyplot
from scipy import stats

import random

random.seed(133711)


def get_table(s_0, s_1,
              derivative=False,
              normalized=True,
              overlap=True,
              w=0,
              diag_factor=1.,
              distance=lambda v1, v2: (v1 - v2) ** 2):
    l_a, l_b = len(s_0), len(s_1)
    if normalized:
        # max_a, min_a = max(s_0), min(s_0)
        # series_a = [(x - min_a) / (max_a - min_a) for x in s_0]
        # max_b, min_b = max(s_1), min(s_1)
        # series_b = [(x - min_b) / (max_b - min_b) for x in s_1]
        series_a = stats.zscore(s_0)
        series_b = stats.zscore(s_1)

    else:
        series_a = s_0[:]
        series_b = s_1[:]

    if derivative:
        series_a = [0.] + [0. if 0. >= series_a[i+1] else 1. - series_a[i] / series_a[i+1] for i in range(l_a - 1)]
        series_b = [0.] + [0. if 0. >= series_b[i+1] else 1. - series_b[i] / series_b[i+1] for i in range(l_b - 1)]

    if 0 < w:
        w = max(w, abs(l_a - l_b))
    """
    # https://docs.scipy.org/doc/numpy/user/quickstart.html
    table = numpy.zeros((l_a + 1, l_b + 1))
    table[0, 1:l_b] = [sys.float_info.max for _ in range(l_b)]
    table[1:l_a, 0] = [sys.float_info.max for _ in range(l_a)]
    print("shape: {}".format(table.shape))
    """
    table = [[0. for _ in range(l_b)] for _ in range(l_a)]
    # """

    diag_factors = [1., 1., diag_factor]

    for i in range(l_a):
        row = table[i]
        sub_range = (0, l_b) if 0 >= w else (max(0, i-w), min(l_b, i+w))
        for j in range(*sub_range):
            dist = distance(series_a[i], series_b[j])
            if j == i == 0 or (overlap and (i == 0 or j == 0)):
                row[j] = dist
            elif not overlap and j == 0:
                row[j] = dist + table[i-1][j]
            elif not overlap and i == 0:
                row[j] = dist + row[j-1]
            else:
                _i, prev_dist = min(enumerate([table[i-1][j], row[j-1], table[i-1][j-1]]), key=lambda _x: _x[1])
                row[j] = dist * diag_factors[_i] + prev_dist
            # print("{}, {}: {}".format(i, j, d))
            # print(table_str([[-1.] + series_a] + [[series_b[idx]] + table[idx] for idx in range(l_b)]))
            # print()
    return table


def get_path(table, overlap=False):
    r = False
    if overlap:
        # left right overlap
        # r_i, r_j = min({(len(table) - 1, _j) for _j in range(1, len(table[0]))}, key=lambda ij: table[ij[0]][ij[1]])
        # c_i, c_j = min({(_i, len(table[0]) - 1) for _i in range(1, len(table))}, key=lambda ij: table[ij[0]][ij[1]])

        # forced overlap
        r_i, r_j = min({(len(table) - 1, _j) for _j in range(len(table) // 2, len(table[0]))}, key=lambda ij: table[ij[0]][ij[1]])
        c_i, c_j = min({(_i, len(table[0]) - 1) for _i in range(len(table[0]) // 2, len(table))}, key=lambda ij: table[ij[0]][ij[1]])

        if table[r_i][r_j] < table[c_i][c_j]:
            i, j = r_i, r_j
            r = True
        else:
            i, j = c_i, c_j
    else:
        i, j = len(table) - 1, len(table[0]) - 1

    fork = [(-1, 0), (0, -1), (-1, -1)]
    path = [(i, j)]

    # left-right-overlap
    # while (overlap and ((r and not 0 == j) or (not r and not 0 == i))) or (not overlap and (i, j) != (0, 0)):

    # forced overlap
    while (overlap and ((r and not (0 == j and i < len(table) // 2)) or (not r and not (0 == i and j < len(table[0]) // 2)))) or (not overlap and (i, j) != (0, 0)):
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
    ax2.set_ylabel(a_label, color="#1f77b4")

    ax21.plot(range(len(a_start), len(a_start) + len(fitted_b)), fitted_b, color="#ff7f0e", alpha=.5)
    ax21.set_ylabel(b_label, color="#ff7f0e")

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


def get_fit(a, b, cur_a, cur_b,
            result_dir=None,
            overlap=False,
            w=0,
            normalized=True,
            derivative=False,
            diag_factor=1.,
            distance=lambda v1, v2: (v1 - v2) ** 2):
    print("{:s}: {:d}, {:s}: {:d} ".format(cur_a, len(a), cur_b, len(b)))

    t = get_table(a, b, normalized=normalized, derivative=derivative, overlap=overlap, diag_factor=diag_factor, distance=distance, w=w)
    p = get_path(t, overlap=overlap)

    target_path = None
    if result_dir is not None:
        target_path = result_dir + "{:s}_{:s}.png".format(cur_a, cur_b)

    plot_series(a, b, p, a_label=cur_a, b_label=cur_b, file_path=target_path)

    start_pos, end_pos = p[0], p[-1]
    deviation = t[end_pos[0]][end_pos[1]]
    b_range = start_pos[0], end_pos[0]
    a_range = start_pos[1], end_pos[1]
    return deviation, a_range, b_range
