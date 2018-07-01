# implement
#   rebalancing
#   dtw predictions
#   rational semiotic model
#   learn avrg deviation to get turn points
import datetime
import json
from typing import Sequence, Tuple

from matplotlib import pyplot

from source.data.data_generation import get_series


class TradingSignal(object):
    def __init__(self, state_path: str = None):
        self.state_path = state_path

    def get_tendency(self, source_info: object) -> float:
        # return a value between -1 and 1
        raise NotImplementedError()

    def train(self, arguments: object):
        if self.state_path is not None:
            raise NotImplementedError()
        else:
            raise TypeError("Stateless signals cannot be trained.")


class ChannelSignal(TradingSignal):
    def __init__(self, inertia):
        super().__init__()
        self.e = -1.
        self.d = -1.
        self.is_running = False
        self.inertia = inertia

    def get_tendency(self, source_info: object) -> float:
        if not isinstance(source_info, float):
            raise TypeError("{:s} requires single float inputs.".format(self.__class__.__name__))
        v = source_info
        if not self.is_running:
            self.e = v
            self.d = 0.
            self.is_running = True
        else:
            self.e = self.e * self.inertia + v * (1. - self.inertia)
            self.d = self.d * self.inertia + abs(self.e - v) * (1. - self.inertia)
        return max(-1., min(1., (self.e - v) / self.d)) if self.d != 0. else 0.


def get_channel(time_series: Sequence[float], inertia=.5) -> Tuple[float, float]:
    e = -1.
    d = -1.
    for i, v in enumerate(time_series):
        if i < 1:
            e = v
            d = 0.
        else:
            e = e * inertia + v * (1. - inertia)
            d = d * inertia + abs(e - v) * (1. - inertia)
        yield e, d


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

    expectation, deviation = zip(*[(_e, _d) for _e, _d in get_channel(time_series, inertia=.9)])

    _, (ax1, ax2, ax3) = pyplot.subplots(3, sharex="all")

    ax1.plot(time_series, label="original")
    ax1.plot(expectation, label="expectation")
    ax1.plot([_x - _d for _x, _d in zip(expectation, deviation)], label="lower bound")
    ax1.plot([_x + _d for _x, _d in zip(expectation, deviation)], label="upper bound")
    ax1.legend()

    channel_signal = ChannelSignal(.9)
    signals = [channel_signal.get_tendency(_x) for _x in time_series]
    ax2.plot(signals, label="signal")
    ax2.legend()

    value_a, value_b = 0., 10.
    volume_a = []
    volume_b = []
    trading_actions = []
    for t, (rate, signal) in enumerate(zip(time_series, signals)):
        if t >= 100:
            if signal >= 1.:
                # b to a
                diff_b = value_b * 1.
                value_a += diff_b * rate * .975
                value_b -= diff_b
                trading_actions.append(t)

            elif -1. >= signal:
                # a to b
                diff_a = value_a * 1.
                value_b += (diff_a / rate) * .975
                value_a -= diff_a
                trading_actions.append(t)

        volume_a.append(value_a / rate)
        volume_b.append(value_b)

    ax3.stackplot(range(len(volume_a)), volume_a, volume_b, labels=[cur_a, cur_b])
    for d in trading_actions:
        ax3.axvline(x=d, linewidth=1, color="green", alpha=.2)

    ax3.legend()

    pyplot.tight_layout()
    pyplot.show()


if __name__ == "__main__":
    main()
