# implement
#   rebalancing
#   dtw predictions
#   rational semiotic model
#   learn avrg deviation to get turn points

import datetime
import json

from matplotlib import pyplot

from source.data.data_generation import get_series
from source.signal_system.signals import ChannelSignal


def main():
    with open("../../configs/config.json", mode="r") as file:
        config = json.load(file)
    source_dir = config["data_dir"]     # "../../configs/23Jun2017-23Jun2018-1m/"
    target_dir = config["target_dir"]  # "../../results/dtw/2018-06-25/"
    interval_minutes = config["interval_minutes"]
    start_date = datetime.datetime.strptime(config["start_time"], "%Y-%m-%d_%H:%M:%S_%Z")
    end_date = datetime.datetime.strptime(config["end_time"], "%Y-%m-%d_%H:%M:%S_%Z")

    cur_a, cur_b = "ADA", "ETH"
    source_path = source_dir + "{:s}{:s}.csv".format(cur_a, cur_b)
    time_series = get_series(source_path, range_start=start_date, range_end=end_date, interval_minutes=interval_minutes)
    time_axis = [start_date + datetime.timedelta(minutes=interval_minutes * _x) for _x in range(len(time_series))]

    pyplot.clf()
    pyplot.close()

    _, (channel_axis, signal_axis, asset_axis) = pyplot.subplots(3, sharex="all")

    channel_signal = ChannelSignal(cur_a, 50)
    signals = [channel_signal.get_tendency({cur_a: _x}) for _x in time_series]
    channel_axis.set_ylabel("{:s} / {:s}".format(cur_a, cur_b))
    channel_signal.plot(time_axis, channel_axis)

    signal_axis.plot(time_axis, signals, label="signal")
    signal_axis.set_ylabel("trading signal")
    signal_axis.legend()

    value_a, value_b = 0., 10.
    volume_a = []
    volume_b = []
    trading_actions = []
    risk = .1
    for t, (rate, signal) in enumerate(zip(time_series, signals)):
        if t >= channel_signal.window_size:
            if signal >= 1.:
                # b to a
                diff_b = value_b * risk
                value_a += diff_b * rate * .975
                value_b -= diff_b
                now_time = start_date + datetime.timedelta(minutes=interval_minutes * t)
                trading_actions.append(now_time)

            elif -1. >= signal:
                # a to b
                diff_a = value_a * risk
                value_b += (diff_a / rate) * .975
                value_a -= diff_a
                now_time = start_date + datetime.timedelta(minutes=interval_minutes * t)
                trading_actions.append(now_time)

        volume_a.append(value_a / rate)
        volume_b.append(value_b)

    asset_axis.stackplot(time_axis, volume_a, volume_b, labels=[cur_a, cur_b])
    asset_axis.set_ylabel("value in {:s}".format(cur_b))
    for d in trading_actions:
        asset_axis.axvline(x=d, linewidth=1, color="green", alpha=.2)

    asset_axis.legend()

    pyplot.gcf().autofmt_xdate()
    pyplot.tight_layout()
    pyplot.show()


if __name__ == "__main__":
    main()
