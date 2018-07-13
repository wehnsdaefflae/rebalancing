import datetime
from typing import Sequence, Any, Dict, TypeVar, Generic, Iterable, Tuple

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
        self.original = []

    def _get_signal(self, source_info: SIGNAL_INPUT) -> float:
        # return a value between -1 and 1
        raise NotImplementedError()

    def _log(self):
        raise NotImplementedError()

    def get_tendency(self, source_info: SIGNAL_INPUT) -> float:
        signal = self._get_signal(source_info)
        if self.iterations < self.initialization:
            signal = 0.

        if self.plot_log:
            self.original.append(source_info)
            self._log()

        self.iterations += 1
        return min(1., max(-1., signal))

    def train(self, state_path: str, arguments: Any):
        raise NotImplementedError()

    def plot(self, time: Sequence[datetime.datetime], axis: Axes, axis_label: str = ""):
        if not self.plot_log:
            raise AttributeError("Signal is not set up for plotting.")
        if 0 < len(axis_label):
            axis.set_ylabel(axis_label)
        axis.plot(time, self.original, label="original")
        self._plot(time, axis)
        axis.legend()

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
        self.expect, self.dev = [], []

    def _log(self):
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
        axis.plot(time[self.initialization:], self.expect, label="expectation")
        lower = [_e - _d for _e, _d in zip(self.expect, self.dev)]
        axis.plot(time[self.initialization:], lower, label="lower bound")
        upper = [_e + _d for _e, _d in zip(self.expect, self.dev)]
        axis.plot(time[self.initialization:], upper, label="upper bound")


class RelativeStrengthIndexSignal(StatelessMixin, TradingSignal[float]):
    def __init__(self, history_length: int = 144):
        if history_length < 1:
            raise ValueError("history_length must be equal to or greater than 1.")
        TradingSignal.__init__(self, initialization=history_length)
        StatelessMixin.__init__(self)
        self.history_length = history_length
        self.history = []

        self.avrg_up, self.avrg_dn = .0, .0
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

    def _log(self):
        if self.iterations < self.initialization:
            return
        self.upper.append(self.avrg_up)
        self.lower.append(self.avrg_dn)

    def _plot(self, time: Sequence[datetime.datetime], axis: Axes):
        axis.plot(time[self.initialization:], self.lower, label="lower bound")
        axis.plot(time[self.initialization:], self.upper, label="upper bound")

    def train(self, state_path: str, arguments: Any):
        raise TypeError("This is a stateless signal.")


class HillValleySignal(StatelessMixin, TradingSignal[float]):
    def __init__(self, window_size: int):
        TradingSignal.__init__(self, initialization=window_size)
        StatelessMixin.__init__(self)
        self.window = [0. for _ in range(self.initialization)]

    def _get_signal(self, source_info: SIGNAL_INPUT) -> float:
        min_index, min_value = -1, .0
        max_index, max_value = -1, .0
        for each_index, each_value in enumerate(self.window):
            if min_index < 0:
                min_index, min_value = each_index, each_value
                max_index, max_value = each_index, each_value
                continue

            elif each_value < min_value:
                min_index, min_value = each_index, each_value

            elif max_value < each_value:
                max_index, max_value = each_index, each_value

        self.window.pop(0)
        self.window.append(source_info)

        if 0. >= min_value:
            return 0.

        if min_index < max_index:
            return 1. - max_value / min_value

        elif max_index < min_index:
            return max_value / min_value - 1.

        return 0.

    def _log(self):
        pass

    def _plot(self, time: Sequence[datetime.datetime], axis: Axes):
        pass


class DynamicTimeWarpingSignal(TradingSignal[float]):
    def __init__(self):
        super().__init__()

    def _get_signal(self, source_info: SIGNAL_INPUT) -> float:
        pass

    def _log(self):
        pass

    def _plot(self, time: Sequence[datetime.datetime], axis: Axes):
        pass

    def train(self, state_path: str, arguments: Any):
        pass


class SemioticModelSignal(TradingSignal[float]):
    def __init__(self):
        super().__init__()

    def _get_signal(self, source_info: SIGNAL_INPUT) -> float:
        pass

    def _log(self):
        pass

    def _plot(self, time: Sequence[datetime.datetime], axis: Axes):
        pass

    def train(self, state_path: str, arguments: Any):
        pass


class FakeSignal(StatelessMixin, TradingSignal[float]):
    def __init__(self, signal_data: Sequence[float]):
        TradingSignal.__init__(self, initialization=0)
        StatelessMixin.__init__(self)
        self.signal_data = [signal_data[_i + 1] / signal_data[_i] - 1. for _i in range(len(signal_data) - 1)]
        self.index = 0

    def _get_signal(self, source_info: SIGNAL_INPUT) -> float:
        if self.index >= len(self.signal_data):
            return 0.
        signal = self.signal_data[self.index]
        self.index += 1
        return signal

    def _log(self):
        pass

    def _plot(self, time: Sequence[datetime.datetime], axis: Axes):
        pass


def optimize_signal(signal: TradingSignal, time_series: Iterable[Tuple[datetime.datetime, float]]):
    pass

def main():
    cur = "EOS"
    # signal = RelativeStrengthIndexSignal(history_length=100)
    # signal = SymmetricChannelSignal(window_size=500)
    signal = HillValleySignal(window_size=50)
    # signal = FakeSignal([_x[1] for _x in DEBUG_SERIES(cur, config_path="../../../configs/config.json")])

    time_axis = []
    signal_axis = []
    value_eth, value_cur = 1., 0.

    buys = []
    sells = []
    total_value = []
    other_value = []
    amount_cur = -1.
    tolerance = .1

    for each_date, each_rate in DEBUG_SERIES(cur, config_path="../../../configs/config.json"):
        tendency = signal.get_tendency(each_rate)
        if tendency >= tolerance:
            if 0. < value_eth:
                buys.append(each_date)
                value_cur += value_eth / each_rate
                value_eth = 0.
        elif -tolerance >= tendency:
            if 0. < value_cur:
                sells.append(each_date)
                value_eth += value_cur * each_rate
                value_cur = 0.

        if amount_cur < 0.:
            amount_cur = 1. / each_rate
        other_value.append(amount_cur * each_rate)
        time_axis.append(each_date)
        signal_axis.append(tendency)
        total_value.append(value_eth + value_cur * each_rate)

    print("success: {:.5f}".format(total_value[-1]))
    pyplot.clf()
    pyplot.close()

    fig, (ax1, ax2, ax3) = pyplot.subplots(3, sharex="all")
    signal.plot(time_axis, ax1, axis_label=cur)
    ax2.plot(time_axis, signal_axis)
    ax2.set_ylabel("signal")
    ax3.plot(time_axis, total_value, label="total value")
    ax3.plot(time_axis, other_value, label="all {:s} value".format(cur))
    ax3.set_ylabel("total value in ETH")
    ax3.legend()

    for each_buy in buys:
        ax1.axvline(x=each_buy, color="red", alpha=.2)
        ax2.axvline(x=each_buy, color="red", alpha=.2)
        ax3.axvline(x=each_buy, color="red", alpha=.2)
    for each_sell in sells:
        ax1.axvline(x=each_sell, color="green", alpha=.2)
        ax2.axvline(x=each_sell, color="green", alpha=.2)
        ax3.axvline(x=each_sell, color="green", alpha=.2)

    pyplot.tight_layout()
    pyplot.show()


if __name__ == "__main__":
    main()
