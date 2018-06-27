import datetime
import json
import os
from math import sin, cos

from source.dtw.my_dtw import get_fit
from source.data.parse_data import get_series


def fit_exchange_rates(cur_a, cur_b, start_date, end_date, interval, parameters, data_dir, result_dir=None):
    timestamp_start, timestamp_end = int(start_date.timestamp()), int(end_date.timestamp())
    fp_a = data_dir + "{}.csv".format(cur_a)
    fp_b = data_dir + "{}.csv".format(cur_b)
    a = get_series(fp_a, range_start=timestamp_start, range_end=timestamp_end, interval_minutes=interval)
    b = get_series(fp_b, range_start=timestamp_start, range_end=timestamp_end, interval_minutes=interval)
    if len(a) != len(b):
        msg = "{:s} and {:s} from {:s} to {:s}: sample number different ({:d} vs. {:d})!"
        raise ValueError(msg.format(fp_a, fp_b, str(timestamp_start), str(timestamp_end), len(a), len(b)))

    error, overlap, offset = get_fit(a, b, cur_a, cur_b, result_dir=result_dir, **parameters)

    msg = "{:s} precedes {:s} with an offset of {:d} and an overlap for {:d} with a deviation of {:.4f}"
    print(msg.format(cur_a, cur_b, offset, overlap, error, offset))
    return error, overlap, offset


def fit_test_data():
    """
    g = absolute_brownian(initial=1., factor=2., relative_bias=.1)
    a = [next(g) for _ in range(10)]
    b = [x - .1 for x in a][3:8]

    """
    cur_a, cur_b = "sin", "cos"
    a = [sin(x/7.) for x in range(1000)]
    b = [cos(x/11.)/3 for x in range(1000)]
    # """

    parameters = {"overlap": True, "normalized": True, "derivative": False, "diag_factor": .5}
    error, overlap, offset = get_fit(a, cur_a, b, cur_b, **parameters)

    msg = "{:s} precedes {:s} with an offset of {:d} and an overlap for {:d} with a deviation of {:.4f}"
    print(msg.format(cur_a, cur_b, offset, overlap, error, offset))


def train_dtw():
    # "2018-06-01_00:00:00_UTC"
    # "2018-06-07_00:00:00_UTC"

    parameters = {"overlap": True,
                  "normalized": True,
                  "derivative": False,
                  "diag_factor": .05,
                  "w": 0,
                  "distance": lambda v1, v2: (v1 - v2) ** 2}

    with open("config.json", mode="r") as file:
        config = json.load(file)
    source_dir = config["data_dir"]     # "../../data/binance/23Jun2017-23Jun2018-1m/"
    target_dir = config["target_dir"]  # "../../results/dtw/2018-06-25/"
    interval_minutes = config["interval_minutes"]
    start_date = datetime.datetime.strptime(config["start_date"], "%Y-%m-%d_%H:%M:%S_%Z")
    end_date = datetime.datetime.strptime(config["end_date"], "%Y-%m-%d_%H:%M:%S_%Z")

    all_pairs = [os.path.splitext(f)[0] for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]
    all_pairs = sorted(x for x in all_pairs if x[-3:] == "ETH")

    if not os.path.isdir(target_dir):
        os.mkdir(target_dir)

    if not os.path.exists(target_dir + "results.csv"):
        with open(target_dir + "results.csv", mode="w") as file:
            file.write("time\tcurrency_a\tcurrency_b\terror\toverlap\toffset\n")

    iterations = 0
    total_pairs = len(all_pairs) * (len(all_pairs) - 1) // 2

    for i, each_cur in enumerate(all_pairs):
        for j in range(i + 1, len(all_pairs)):
            every_cur = all_pairs[j]
            iterations += 1
            print("starting {:s} X {:s} ({:d}/{:d})...".format(each_cur, every_cur, iterations, total_pairs))

            try:
                if os.path.isfile(target_dir + "{:s}_{:s}.png".format(each_cur, every_cur)):
                    print("Currency pair {:s} X {:s} already fitted. Skipping...".format(each_cur, every_cur))
                    continue
                e, o, ofs = fit_exchange_rates(each_cur, every_cur,
                                               start_date, end_date,
                                               interval_minutes, parameters, source_dir, result_dir=target_dir)

                row = [str(datetime.datetime.now()),
                       each_cur, every_cur, "{:.5f}".format(e), "{:d}".format(o), "{:d}".format(ofs)]

            except ValueError as e:
                row = [str(datetime.datetime.now()), each_cur, every_cur, str(e)]

            with open(target_dir + "results.csv", mode="a") as file:
                file.write("\t".join(row) + "\n")


def single_run():
    with open("config.json", mode="r") as file:
        config = json.load(file)
    source_dir = config["data_dir"]     # "../../data/binance/23Jun2017-23Jun2018-1m/"
    target_dir = config["target_dir"]  # "../../results/dtw/2018-06-25/"
    interval_minutes = config["interval_minutes"]
    start_date = datetime.datetime.strptime(config["start_date"], "%Y-%m-%d_%H:%M:%S_%Z")
    end_date = datetime.datetime.strptime(config["end_date"], "%Y-%m-%d_%H:%M:%S_%Z")

    parameters = {"overlap": True,
                  "normalized": True,
                  "derivative": False,
                  "diag_factor": .05,
                  "w": 0,
                  "distance": lambda v1, v2: (v1 - v2) ** 2}

    fit_exchange_rates("ADAETH", "ADXETH", start_date, end_date, interval_minutes, parameters, source_dir)


def test_dtw():
    with open("config.json", mode="r") as file:
        config = json.load(file)
    source_dir = config["data_dir"]         # "../../data/binance/23Jun2017-23Jun2018-1m/"
    target_dir = config["target_dir"]       # "../../results/dtw/2018-06-25/"
    interval = config["interval_minutes"]   # 10

    results = []
    with open(target_dir + "results.csv", mode="r") as file:
        line = file.readline()
        header = line[1:-1].split("\t")
        for each_line in file:
            row = each_line[1:-1].split("\t")
            try:
                each_result = {_h: float(_c) if not _h.startswith("cur") else _c for _h, _c in zip(header, row)}
            except ValueError:
                continue
            results.append(each_result)

    for each_result in results:
        cur_a, cur_b = each_result["currency_a"], each_result["currency_b"]
        error, overlap, offset = each_result["error"], each_result["overlap"], each_result["offset"]


def main():
    # single_run()
    train_dtw()


if __name__ == "__main__":
    main()
