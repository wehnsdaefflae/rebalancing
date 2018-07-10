import datetime
from typing import Sequence, Any, Dict, TypeVar, Generic, Iterable

from matplotlib import pyplot
from matplotlib.axes import Axes

from source.data.data_generation import DEBUG_SERIES

SIGNAL_INPUT = TypeVar("SIGNAL_INPUT")
RATE_INFO = Dict[str, float]


class TradingSignal(Generic[SIGNAL_INPUT]):
    def __init__(self, initialization: int = 0, plot_log: bool = True):
        self.initialization = initialization
        self.plot_log = plot_log
        self.iterations = 0

    def _get_signal(self, source_info: SIGNAL_INPUT) -> float:
        # return a value between -1 and 1
        raise NotImplementedError()

    def _log(self, source_info: SIGNAL_INPUT):
        raise NotImplementedError()

    def get_tendency(self, source_info: SIGNAL_INPUT) -> float:
        signal = self._get_signal(source_info)
        if self.iterations < self.initialization:
            signal = 0.

        if self.plot_log:
            self._log(source_info)

        self.iterations += 1
        return min(1., max(-1., signal))

    def train(self, state_path: str, arguments: Any):
        raise NotImplementedError()

    def reset_log(self):
        raise NotImplementedError()

    def plot(self, time: Sequence[datetime.datetime], axis: Axes, axis_label: str = ""):
        if not self.plot_log:
            raise AttributeError("Signal is not set up for plotting.")
        if 0 < len(axis_label):
            axis.set_ylabel(axis_label)
        self._plot(time, axis)

    def _plot(self, time: Sequence[datetime.datetime], axis: Axes):
        raise NotImplementedError()


class StatelessMixin(object):
    def train(self, state_path: str, arguments: Any):
        raise TypeError("This is a stateless signal.")


class SymmetricChannelSignal(StatelessMixin, TradingSignal[float]):
    def __init__(self, window_size: int = 50):
        if window_size < 1:
            raise ValueError("Window must be > 1!")
        TradingSignal.__init__(self, initialization=window_size)
        StatelessMixin.__init__(self)
        self.e = -1.
        self.d = -1.
        self.is_running = False
        self.window_size = window_size
        self.last_position = 0
        self.original = []
        self.expect, self.dev = [], []

    def _log(self, source_info: float):
        v = source_info
        self.original.append(v)
        if self.iterations < self.initialization:
            return
        self.dev.append(self.d)
        self.expect.append(self.e)

    def _get_signal(self, source_info: float) -> float:
        v = source_info
        if not self.is_running:
            self.e = v
            self.d = 0.
            self.is_running = True
        else:
            self.e = (self.e * (self.window_size - 1) + v) / self.window_size
            self.d = (self.d * (self.window_size - 1) + abs(self.e - v)) / self.window_size

        if v >= self.e + self.d:
            position = 1
        elif v < self.e - self.d:
            position = -1
        else:
            position = 0

        tendency = 0.
        if position == 0:
            if self.last_position == 1:
                tendency = -1.
            elif self.last_position == -1:
                tendency = 1.
        self.last_position = position
        return tendency

    def _plot(self, time: Sequence[datetime.datetime], axis: Axes):
        axis.plot(time, self.original, label="original")
        axis.plot(time[self.initialization:], self.expect, label="expectation")
        lower = [_e - _d for _e, _d in zip(self.expect, self.dev)]
        axis.plot(time[self.initialization:], lower, label="lower bound")
        upper = [_e + _d for _e, _d in zip(self.expect, self.dev)]
        axis.plot(time[self.initialization:], upper, label="upper bound")
        axis.legend()

    def reset_log(self):
        self.original.clear()
        self.expect.clear()
        self.dev.clear()


class RelativeStrengthIndexSignal(StatelessMixin, TradingSignal[float]):
    def __init__(self, history_length: int = 144):
        if history_length < 1:
            raise ValueError("history_length must be equal to or greater than 1.")
        TradingSignal.__init__(self, initialization=history_length)
        StatelessMixin.__init__(self)
        self.history_length = history_length
        self.history = []

        self.avrg_up, self.avrg_dn = .0, .0
        self.original = []
        self.upper, self.lower = [], []

    def _get_signal(self, source_info: float) -> float:
        if self.history_length < len(self.history):
            self.history.pop(0)

        if len(self.history) == self.history_length:
            pos_dev = [max(self.history[_i + 1] - self.history[_i], .0) for _i in range(self.history_length - 1)]
            neg_dev = [min(self.history[_i + 1] - self.history[_i], .0) for _i in range(self.history_length - 1)]

            sum_up = sum(pos_dev)
            sum_dn = sum(neg_dev)

            avrg_dev_up = sum_up / self.history_length
            avrg_dev_dn = sum_dn / self.history_length

            self.avrg_up = source_info + avrg_dev_up
            self.avrg_dn = source_info + avrg_dev_dn

            total = avrg_dev_up + avrg_dev_dn
            if total != 0.:
                signal = avrg_dev_up / total
            else:
                signal = 0.

        else:
            signal = .0

        self.history.append(source_info)
        return signal / 100.

    def _log(self, source_info: float):
        self.original.append(source_info)
        if self.iterations < self.initialization:
            return
        self.upper.append(self.avrg_up)
        self.lower.append(self.avrg_dn)

    def reset_log(self):
        self.original.clear()
        self.upper.clear()
        self.lower.clear()

    def _plot(self, time: Sequence[datetime.datetime], axis: Axes):
        axis.plot(time, self.original, label="original")
        axis.plot(time[self.initialization:], self.lower, label="lower bound")
        axis.plot(time[self.initialization:], self.upper, label="upper bound")
        axis.legend()

    def train(self, state_path: str, arguments: Any):
        raise TypeError("This is a stateless signal.")


class FakeSignal(StatelessMixin, TradingSignal[float]):
    def __init__(self, signal_data: Sequence[float]):
        TradingSignal.__init__(self, initialization=0)
        StatelessMixin.__init__(self)
        self.signal_data = [signal_data[_i + 1] / signal_data[_i] - 1. for _i in range(len(signal_data) - 1)]
        self.index = 0
        self.original = []

    def _get_signal(self, source_info: SIGNAL_INPUT) -> float:
        if self.index >= len(self.signal_data):
            return 0.
        signal = self.signal_data[self.index]
        self.index += 1
        return signal

    def _log(self, source_info: SIGNAL_INPUT):
        self.original.append(source_info)

    def reset_log(self):
        self.original.clear()

    def _plot(self, time: Sequence[datetime.datetime], axis: Axes):
        axis.plot(time, self.original, label="original")
        axis.legend()


def main():
    cur = "ADA"
    # signal = RelativeStrengthIndexSignal(history_length=60)
    signal = SymmetricChannelSignal(window_size=150)
    # signal = FakeSignal([_x[1] for _x in DEBUG_SERIES(cur, config_path="../../../configs/config.json")])

    time_axis = []
    signal_axis = []
    value_a, value_b = 0., 1.
    each_rate = 1.

    buys = []
    sells = []
    total_value = []

    for each_date, each_rate in DEBUG_SERIES(cur, config_path="../../../configs/config.json"):
        tendency = signal.get_tendency(each_rate)
        if tendency >= .8:
            if 0. < value_a:
                buys.append(each_date)
            value_b += value_a * each_rate
            value_a = 0.
        elif -.8 >= tendency:
            if 0. < value_b:
                sells.append(each_date)
            value_a += value_b / each_rate
            value_b = 0.

        time_axis.append(each_date)
        signal_axis.append(tendency)
        total_value.append(value_b + value_a * each_rate)

    print("success: {:.5f}".format(total_value[-1]))
    pyplot.clf()
    pyplot.close()

    fig, (ax1, ax2, ax3) = pyplot.subplots(3, sharex="all")
    signal.plot(time_axis, ax1, axis_label=cur)
    for each_buy in buys:
        ax1.axvline(x=each_buy, color="green", alpha=.2)
    for each_sell in sells:
        ax1.axvline(x=each_sell, color="red", alpha=.2)
    ax2.plot(time_axis, signal_axis, label="signal")
    ax3.plot(time_axis, total_value, label="total value in {:s}".format(cur))
    ax2.legend()

    pyplot.tight_layout()
    pyplot.show()


if __name__ == "__main__":
    main()
