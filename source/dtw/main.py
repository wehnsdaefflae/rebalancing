import datetime
import os
from math import sin, cos

from source.dtw.my_dtw import get_series, get_fit


def fit_exchange_rates(cur_a, cur_b, start_date, end_date, interval, parameters, result_dir=None):
    timestamp_start, timestamp_end = int(start_date.timestamp()), int(end_date.timestamp())
    fp_a = "../../data/binance/23Jun2017-23Jun2018-1m/{}.csv".format(cur_a)
    fp_b = "../../data/binance/23Jun2017-23Jun2018-1m/{}.csv".format(cur_b)
    a = get_series(fp_a, range_start=timestamp_start, range_end=timestamp_end, interval_minutes=interval)
    b = get_series(fp_b, range_start=timestamp_start, range_end=timestamp_end, interval_minutes=interval)
    if len(a) != len(b):
        msg = "{:s} and {:s} from {:s} to {:s}: sample number different ({:d} vs. {:d})!"
        raise ValueError(msg.format(fp_a, fp_b, str(timestamp_start), str(timestamp_end), len(a), len(b)))

    temp_offset, error = get_fit(a, b, cur_a, cur_b, result_dir=result_dir, **parameters)

    msg = "{} is {:d} ahead of {} with an overlap deviation of {:.4f}"
    print(msg.format(cur_a, temp_offset, cur_b, error))
    return temp_offset, error


def fit_test_data():
    """
    g = absolute_brownian(initial=1., factor=2., relative_bias=.1)
    a = [next(g) for _ in range(10)]
    b = [x - .1 for x in a][3:8]

    """
    cur_a, cur_b = "sin", "cos"
    a = [sin(x/7.) for x in range(1000)]
    b = [cos(x/11.)/3 for x in range(1000)]
    #"""

    parameters = {"overlap": True, "normalized": True, "derivative": False, "diag_factor": .5}
    temp_offset, error = get_fit(a, cur_a, b, cur_b, **parameters)

    msg = "{} is {:d} ahead of {} with an overlap deviation of {:.4f}"
    print(msg.format(cur_a, temp_offset, cur_b, error))


def batch():
    start_date = datetime.datetime(2018, 6, 1, 0, 0, 0, tzinfo=datetime.timezone.utc)
    end_date = datetime.datetime(2018, 6, 7, 0, 0, 0, tzinfo=datetime.timezone.utc)
    interval_minutes = 10

    parameters = {"overlap": True,
                  "normalized": True,
                  "derivative": False,
                  "diag_factor": 2.,
                  "w": 0,
                  "distance": lambda v1, v2: (v1 - v2) ** 2}

    source_dir = "../../data/binance/23Jun2017-23Jun2018-1m/"
    target_dir = "../../results/dtw/2018-06-25/"

    all_pairs = [os.path.splitext(f)[0] for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]
    all_pairs = sorted(x for x in all_pairs if x[-3:] == "ETH")

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
                o, e = fit_exchange_rates(each_cur, every_cur,
                                          start_date, end_date,
                                          interval_minutes, parameters, result_dir=target_dir)

                row = [str(datetime.datetime.now()), each_cur, every_cur, "{:d}".format(o), "{:.5f}".format(e)]
            except ValueError as e:
                row = [str(datetime.datetime.now()), each_cur, every_cur, str(e)]

            with open(target_dir + "results.csv", mode="a") as file:
                file.write("\t".join(row) + "\n")


def single_run():
    start_date = datetime.datetime(2018, 6, 1, 0, 0, 0, tzinfo=datetime.timezone.utc)
    end_date = datetime.datetime(2018, 6, 7, 0, 0, 0, tzinfo=datetime.timezone.utc)
    interval_minutes = 10

    parameters = {"overlap": True,
                  "normalized": True,
                  "derivative": False,
                  "diag_factor": 2.,
                  "w": 0,
                  "distance": lambda v1, v2: (v1 - v2) ** 2}
    fit_exchange_rates("HSRETH", "WANETH", start_date, end_date, interval_minutes, parameters=parameters)


def main():
    # single_run()
    batch()


if __name__ == "__main__":
    main()
