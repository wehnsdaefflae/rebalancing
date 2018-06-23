import datetime
import os

from source.dtw import get_table, get_path, get_series, plot_series


def get_shift(p):
    temporal_tendency = [sum(p[0:i]) for i in range(len(p))]
    total_tendency = [temporal_tendency[i] for i in range(len(temporal_tendency)) if p[i] == 0]
    return sum(total_tendency) / len(total_tendency)


def main():
    project_dir = "20May2018-20Jun2018-1m"
    data_path = "../data/binance/{}/".format(project_dir)
    result_path = "../results/dtw/{}/".format(project_dir)

    if os.path.isdir(result_path):
        print("result path {} already exists!".format(result_path))
        exit()

    os.mkdir(result_path)

    start_date = datetime.datetime(2018, 5, 20, 0, 0, 0, tzinfo=datetime.timezone.utc)
    # end_date = datetime.datetime(2018, 6, 20, 0, 0, 0, tzinfo=datetime.timezone.utc)
    end_date = datetime.datetime(2018, 5, 27, 0, 0, 0, tzinfo=datetime.timezone.utc)

    timestamp_start, timestamp_end = int(start_date.timestamp()), int(end_date.timestamp())

    print("From {} to {}.".format(start_date.strftime("%Y-%m-%d %H:%M:%S"), end_date.strftime("%Y-%m-%d %H:%M:%S")))

    all_pairs = [os.path.splitext(f)[0] for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))]
    all_pairs = sorted(x for x in all_pairs if x[-3:] == "ETH") # [:10]

    iterations = 0
    total_pairs = len(all_pairs) * (len(all_pairs) - 1) // 2
    matrix = dict()
    for i, each_pair in enumerate(all_pairs):
        try:
            each_series = get_series(data_path + each_pair + ".csv", range_start=timestamp_start, range_end=timestamp_end)
        except ValueError as e:
            print(e)
            continue

        sub_dict = matrix.get(each_pair)
        if sub_dict is None:
            sub_dict = dict()
            matrix[each_pair] = sub_dict

        for j in range(i + 1, len(all_pairs)):
            every_pair = all_pairs[j]
            try:
                every_series = get_series(data_path + every_pair + ".csv", range_start=timestamp_start, range_end=timestamp_end)
            except ValueError as e:
                print(e)
                continue

            t = get_table(each_series, every_series, derivative=True)
            p = get_path(t)
            shift = get_shift(p)
            deviation = t[-1][-1]
            sub_dict[every_pair] = (shift, deviation)
            print("shift {} to {}: {}, deviation: {}".format(every_pair, each_pair, shift, deviation))

            plot_series(each_series, every_series, p, file_path=result_path + "{}_{}.png".format(each_pair, every_pair))

            iterations += 1
            print("{:5.2f}% done...".format(100. * iterations / total_pairs))

    with open(result_path + "shifts.csv", mode="w") as file:
        keys = sorted(matrix.keys())
        file.write("symbol\t" + "\t".join(keys) + "\n")
        for each_key in keys:
            sub_dict = matrix.get(each_key)
            if sub_dict is not None:
                row = [each_key] + [str(sub_dict.get(x, "-")) for x in keys]
                file.write("\t".join(row) + "\n")


if __name__ == "__main__":
    main()
