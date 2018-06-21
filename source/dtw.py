import sys

from source.main import absolute_brownian


def main(series_a, series_b):
    assert(len(series_a) == len(series_b))
    table = [[sys.float_info.max if i == 0 or j == 0 else 0. for i in range(len(series_a))] for j in range(len(series_b))]
    d_table = [["X" if i == j == 0 else "." for i in range(len(series_a))] for j in range(len(series_b))]
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
    i, j = len(table[0]) - 1, len(table) - 1
    while 0 < i or 0 < j:
        x, y, z = table[i-1][j], table[i][j-1], table[i-1][j-1]
        min_i, min_v = min(enumerate([x, y, z]), key=lambda x_v: x_v[1])
        path.append(["d", "r", "o"][min_i])
        i, j = [(i-1, j), (i, j-1), (i-1, j-1)][min_i]
    return path[::-1]


def table_str(table):
    format_str = "\t".join(["{:8.5f}" for _ in table[0]])
    rows = [format_str.format(*[-1. if v == sys.float_info.max else v for v in x]) for x in table]
    return "\n".join(rows)


if __name__ == "__main__":
    g = absolute_brownian()
    a = [next(g) for _ in range(10)]
    g = absolute_brownian()
    b = [next(g) for _ in range(10)]
    # b = a[:]

    t, d_t = main(a, b)
    print(table_str(t))
    print()
    print("\n".join(["".join(x) for x in d_t]))
    print()
    print(get_path(t))
