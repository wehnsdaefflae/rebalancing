import os

from source.dtw import get_table, get_path


def check_range(pair):
    with open("../data/binance/{}.csv".format(pair), mode="r") as file:
        line = file.readline()
        row = line[:-1].split("\t")
        if row[0] != "1526774400000":
            return False
        for line in file:
            pass
        row = line[:-1].split("\t")
        if row[0] != "1529452800000":
            return False
    return True


def get_shift(series_a, series_b):
    d_a = [1. - series_a[i] / series_a[i+1] for i in range(len(series_a) - 1)]
    d_b = [1. - series_b[i] / series_b[i+1] for i in range(len(series_b) - 1)]

    t = get_table(d_a, d_b)
    p = get_path(t)

    temporal_tendency = [sum(p[0:i]) for i in range(len(p))]
    total_tendency = [temporal_tendency[i] for i in range(len(temporal_tendency)) if p[i] == 0]
    return sum(total_tendency) / len(total_tendency), t[-1][-1]


def get_series(pair):
    series = []
    with open("../data/binance/{}.csv".format(pair), mode="r") as file:
        for line in file:
            row = line[:-1].split("\t")
            series.append(float(row[0]))
    return series


def main():
    data_dir = "../data/binance/"
    all_pairs = [os.path.splitext(f)[0] for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
    all_pairs = sorted(x for x in all_pairs if x[-3:] == "ETH")

    matrix = dict()

    for i, each_pair in enumerate(all_pairs):
        print(each_pair)
        if not check_range(each_pair):
            continue
        each_series = get_series(each_pair)

        sub_dict = matrix.get(each_pair)
        if sub_dict is None:
            sub_dict = dict()
            matrix[each_pair] = sub_dict

        for j in range(i+1, len(all_pairs)):
            every_pair = all_pairs[j]
            print(every_pair)
            if not check_range(every_pair):
                continue
            every_series = get_series(every_pair)

            shift, certainty = get_shift(each_series[:1000], every_series[:1000])
            sub_dict[every_pair] = (shift, certainty)
            print("shift: {}, certainty: {}".format(shift, certainty))
            print()

    print(matrix)


if __name__ == "__main__":
    main()
