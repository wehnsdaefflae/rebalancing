import datetime
import json
import os
from math import sin, cos

from source.experiements.dtw import get_fit
from source.data.data_generation import series_generator


def fit_exchange_rates(cur_a, cur_b, start_date, end_date, interval, parameters, data_dir, result_dir=None):
    fp_a = data_dir + "{}.csv".format(cur_a)
    fp_b = data_dir + "{}.csv".format(cur_b)
    a = list(series_generator(fp_a, range_start=start_date, range_end=end_date, interval_minutes=interval))
    b = list(series_generator(fp_b, range_start=start_date, range_end=end_date, interval_minutes=interval))
    if len(a) != len(b):
        msg = "{:s} and {:s} from {:s} to {:s}: sample number different ({:d} vs. {:d})!"
        raise ValueError(msg.format(fp_a, fp_b, str(start_date), str(end_date), len(a), len(b)))

    error, a_range, b_range = get_fit(a, b, cur_a, cur_b, result_dir=result_dir, **parameters)

    msg = "{:s} range: {:d}-{:d}, {:s} range: {:d}-{:d}, deviation {:.4f}"
    print(msg.format(cur_a, *a_range, cur_b, *b_range, error))
    return error, a_range, b_range


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
    error, a_range, b_range = get_fit(a, cur_a, b, cur_b, **parameters)

    msg = "{:s} range: {:d}-{:d}, {:s} range: {:d}-{:d}, deviation {:.4f}"
    print(msg.format(cur_a, *a_range, cur_b, *b_range, error))


def train_dtw():
    # "2018-06-01_00:00:00_UTC"
    # "2018-06-07_00:00:00_UTC"

    parameters = {"overlap": True,
                  "normalized": True,
                  "derivative": False,
                  "diag_factor": .05,
                  "w": 0,
                  "distance": lambda v1, v2: (v1 - v2) ** 2}

    with open("../../configs/config.json", mode="r") as file:
        config = json.load(file)
    source_dir = config["data_dir"]     # "../../configs/23Jun2017-23Jun2018-1m/"
    target_dir = config["target_dir"]  # "../../results/dtw/2018-06-25/"
    interval_minutes = config["interval_minutes"]
    start_date = datetime.datetime.strptime(config["start_time"], "%Y-%m-%d_%H:%M:%S_%Z")
    end_date = datetime.datetime.strptime(config["end_time"], "%Y-%m-%d_%H:%M:%S_%Z")

    all_pairs = [os.path.splitext(f)[0] for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]
    all_pairs = sorted(x for x in all_pairs if x[-3:] == "ETH")

    if not os.path.isdir(target_dir):
        os.mkdir(target_dir)

    if not os.path.exists(target_dir + "results.csv"):
        with open(target_dir + "results.csv", mode="w") as file:
            file.write("time\tcurrency_a\tcurrency_b\tstart_a\tend_a\tstart_b\tend_b\terror\n")

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
                d, a_r, b_r = fit_exchange_rates(each_cur, every_cur,
                                                 start_date, end_date,
                                                 interval_minutes, parameters, source_dir, result_dir=target_dir)

                row = [str(datetime.datetime.now()),
                       each_cur, every_cur,
                       *["{:d}".format(_x) for _x in a_r],
                       *["{:d}".format(_x) for _x in b_r],
                       "{:.5f}".format(d)]

            except ValueError as e:
                row = [str(datetime.datetime.now()), each_cur, every_cur, str(e), "", "", "", ""]

            with open(target_dir + "results.csv", mode="a") as file:
                file.write("\t".join(row) + "\n")


def single_run():
    with open("../../configs/config.json", mode="r") as file:
        config = json.load(file)
    source_dir = config["data_dir"]     # "../../configs/23Jun2017-23Jun2018-1m/"
    target_dir = config["target_dir"]  # "../../results/dtw/2018-06-25/"
    interval_minutes = config["interval_minutes"]
    start_date = datetime.datetime.strptime(config["start_time"], "%Y-%m-%d_%H:%M:%S_%Z")
    end_date = datetime.datetime.strptime(config["end_time"], "%Y-%m-%d_%H:%M:%S_%Z")

    parameters = {"overlap": True,
                  "normalized": True,
                  "derivative": False,
                  "diag_factor": .05,
                  "w": 0,
                  "distance": lambda v1, v2: (v1 - v2) ** 2}

    fit_exchange_rates("ADAETH", "ADXETH", start_date, end_date, interval_minutes, parameters, source_dir)


def test_dtw():
    with open("../../configs/config.json", mode="r") as file:
        config = json.load(file)
    source_dir = config["data_dir"]     # "../../configs/23Jun2017-23Jun2018-1m/"
    target_dir = config["target_dir"]  # "../../results/dtw/2018-06-25/"
    interval_minutes = config["interval_minutes"]
    start_date = datetime.datetime.strptime(config["start_time"], "%Y-%m-%d_%H:%M:%S_%Z")
    end_date = datetime.datetime.strptime(config["end_time"], "%Y-%m-%d_%H:%M:%S_%Z")

    results = []
    with open(target_dir + "results.csv", mode="r") as file:
        line = file.readline()
        header = line[1:-1].split("\t")
        for each_line in file:
            row = each_line[1:-1].split("\t")
            try:
                each_result = dict()
                for _h, _c in zip(header, row):
                    if _h.startswith("start_") or _h.startswith("end_"):
                        each_result[_h] = int(_c)
                    elif _h == "error":
                        each_result[_h] = float(_c)
                    else:
                        each_result[_h] = _c

            except ValueError:
                continue

            results.append(each_result)

    if not os.path.exists(target_dir + "final_results.csv"):
        with open(target_dir + "final_results.csv", mode="w") as file:
            header = "time", "currency_a", "currency_b", "start_a", "end_a", "start_b", "end_b", "error", "prediction_error"
            file.write("\t".join(header) + "\n")

    prediction_dir = target_dir + "predictions/"
    if not os.path.isdir(prediction_dir):
        os.makedirs(prediction_dir)

    for no_pairs, each_result in enumerate(results):
        cur_a, cur_b = each_result["currency_a"], each_result["currency_b"]
        print("{:s} vs {:s} ({:d}/{:d})".format(cur_a, cur_b, no_pairs + 1, len(results)))
        range_a = each_result["start_a"], each_result["end_a"]
        range_b = each_result["start_b"], each_result["end_b"]
        error = each_result["error"]

        delta = range_a[1] - range_b[1]
        overlap = abs(delta)
        if 1 >= overlap:
            with open(target_dir + "final_results.csv", mode="a") as file:
                row = [str(datetime.datetime.now()),
                       cur_a, cur_b,
                       *["{:d}".format(_x) for _x in range_a],
                       *["{:d}".format(_x) for _x in range_b],
                       "{:.5f}".format(error),
                       "overlap too small"]
                file.write("\t".join(row) + "\n")
            continue

        overlap_time = datetime.timedelta(minutes=overlap * interval_minutes)
        output_start_date = end_date - overlap_time
        target_end_date = end_date + overlap_time

        fp_b = source_dir + "{}.csv".format(cur_b)
        fp_a = source_dir + "{}.csv".format(cur_a)

        a_desc, b_desc = (cur_a + "-t", cur_b + "-o") if delta < 0 else (cur_a + "-o", cur_b + "t")
        if os.path.isfile(prediction_dir + "{:s}_{:s}.png".format(a_desc, b_desc)):
            print("pair {:s} X {:s} already exist. Skipping...".format(a_desc, b_desc))
            continue

        output, target = (fp_b, fp_a) if delta < 0 else (fp_a, fp_b)
        output_series = series_generator(output,
                                         range_start=output_start_date, range_end=end_date,
                                         interval_minutes=interval_minutes)
        target_series = series_generator(target,
                                         range_start=end_date, range_end=target_end_date,
                                         interval_minutes=interval_minutes)

        prediction_error, _, _ = get_fit(output_series, target_series, a_desc, b_desc, result_dir=prediction_dir)

        with open(target_dir + "final_results.csv", mode="a") as file:
            row = [str(datetime.datetime.now()),
                   cur_a, cur_b,
                   *["{:d}".format(_x) for _x in range_a],
                   *["{:d}".format(_x) for _x in range_b],
                   "{:.5f}".format(error),
                   "{:.5f}".format(prediction_error)]
            file.write("\t".join(row) + "\n")


def main():
    # single_run()
    # train_dtw()
    test_dtw()


if __name__ == "__main__":
    main()
