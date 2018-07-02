import datetime
from typing import Sequence, Any, Dict, TypeVar, Generic, Tuple

from matplotlib.axes import Axes


SIGNAL_INPUT = TypeVar("T")
RATE_INFO = Dict[str, float]
PORTFOLIO_INFO = Dict[str, float]
TECH_INFO = Tuple[RATE_INFO, PORTFOLIO_INFO]
SIGNAL_OUTPUT = Dict[str, float]


class TradingSignal(Generic[SIGNAL_INPUT]):
    def __init__(self, asset_name: str, initialization: int = 50, plot_log: bool = True, state_path: str = None):
        self.asset_name = asset_name
        self.plot_log = plot_log
        self.state_path = state_path
        self.initialization = initialization

    def _get_signal(self, source_info: SIGNAL_INPUT) -> float:
        # return a value between -1 and 1
        raise NotImplementedError()

    def _log(self, source_info: SIGNAL_INPUT):
        raise NotImplementedError()

    def get_tendency(self, source_info: SIGNAL_INPUT) -> float:
        if 0 < self.initialization:
            self.initialization -= 1
            signal = 0.

        else:
            signal = self._get_signal(source_info)

        if self.plot_log:
            self._log(source_info)
        return signal

    def train(self, arguments: Any):
        if self.state_path is not None:
            raise NotImplementedError()
        else:
            raise TypeError("Stateless signals cannot be trained.")

    def reset_log(self):
        raise NotImplementedError()

    def plot(self, time: Sequence[datetime.datetime], axis: Axes):
        if not self.plot_log:
            raise AttributeError("Signal is not set up for plotting.")
        self._plot(time, axis)

    def _plot(self, time: Sequence[datetime.datetime], axis: Axes):
        raise NotImplementedError()


class ChannelSignal(TradingSignal[RATE_INFO]):
    def __init__(self, asset_name: str, window_size: int = 10):
        if window_size < 1:
            raise ValueError("Window must be > 1!")
        super().__init__(asset_name)
        self.e = -1.
        self.d = -1.
        self.is_running = False
        self.window_size = window_size
        self.last_position = 0
        self.original = []
        self.upper, self.expect, self.lower = [], [], []

    def _log(self, source_info: RATE_INFO):
        rates = source_info
        v = rates[self.asset_name]
        self.original.append(v)
        self.upper.append(self.e + self.d)
        self.expect.append(self.e)
        self.lower.append(self.e - self.d)

    def _get_signal(self, source_info: RATE_INFO) -> float:
        rates = source_info
        v = rates[self.asset_name]
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
        axis.plot(time, self.expect, label="expectation")
        axis.plot(time, self.lower, label="lower bound")
        axis.plot(time, self.upper, label="upper bound")
        axis.legend()

    def reset_log(self):
        self.original.clear()
        self.expect.clear()
        self.lower.clear()
        self.upper.clear()
