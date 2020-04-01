import datetime
import time
from typing import Sequence, Optional, Tuple

from matplotlib import pyplot, dates
from matplotlib.axis import Axis
from matplotlib.ticker import MaxNLocator

from source.new.learning.tools import smear


class MovingGraph:
    def __init__(self,
                 name_axis_primary: str,
                 names_plots_primary: Sequence[str],
                 name_axis_secondary: str,
                 names_plots_secondary: Sequence[str],
                 size_window: int, interval_ms: int = 1000,
                 limits_primary: Optional[Tuple[float, float]] = None,
                 limits_secondary: Optional[Tuple[float, float]] = None):
        assert 5000. >= interval_ms >= 0.
        self.names_plots_primary = names_plots_primary
        self.names_plots_secondary = names_plots_secondary

        self.interval_ms = interval_ms

        self.no_plots_primary = len(names_plots_primary)
        self.no_plots_secondary = len(names_plots_secondary)

        self.time = None
        self.values_primary = [0. for _ in names_plots_primary]
        self.values_secondary = [0. for _ in names_plots_secondary]

        self.time_range = []
        self.plots_primary = tuple([] for _ in names_plots_primary)
        self.plots_secondary = tuple([] for _ in names_plots_secondary)

        self.size_window = size_window
        self.limits_primary = limits_primary
        self.limits_secondary = limits_secondary

        self.fig, self.ax_primary = pyplot.subplots()
        self.ax_secondary = self.ax_primary.twinx()

        self.name_axis_primary = name_axis_primary
        self.name_axis_secondary = name_axis_secondary

        self.iterations_since_draw = 0

        self.time_last = -1.

    def add_snapshot(self, now: datetime.datetime, points_primary: Sequence[float], points_secondary: Sequence[float]):
        assert len(points_primary) == self.no_plots_primary
        assert len(points_secondary) == self.no_plots_secondary

        for i, (each_value, each_point) in enumerate(zip(self.values_primary, points_primary)):
            self.values_primary[i] = smear(each_value, each_point, self.iterations_since_draw)

        for i, (each_value, each_point) in enumerate(zip(self.values_secondary, points_secondary)):
            self.values_secondary[i] = smear(each_value, each_point, self.iterations_since_draw)

        self.time = now
        self.iterations_since_draw += 1

        time_now = time.time() * 1000.
        if self.time_last < 0. or time_now - self.time_last >= self.interval_ms:
            self.draw()
            self.time_last = time_now

    @staticmethod
    def _set_limits(axis: Axis.axes, plots: Sequence[Sequence[float]]):
        val_min = min(min(each_plot) for each_plot in plots)
        val_max = max(max(each_plot) for each_plot in plots)
        val_d = .2 * (val_max - val_min)
        axis.set_ylim([val_min - val_d, val_max + val_d])

    def draw(self):
        self.ax_primary.clear()
        self.ax_secondary.clear()

        self.ax_primary.set_xlabel("time")
        self.ax_primary.set_ylabel(self.name_axis_primary)
        self.ax_secondary.set_ylabel(self.name_axis_secondary)

        self.time_range.append(self.time)
        del(self.time_range[:-self.size_window])

        self.ax_primary.xaxis.set_major_formatter(dates.DateFormatter("%d.%m.%Y %H:%M"))
        self.ax_primary.xaxis.set_major_locator(MaxNLocator(10))

        self.ax_secondary.xaxis.set_major_formatter(dates.DateFormatter("%d.%m.%Y %H:%M"))
        self.ax_secondary.xaxis.set_major_locator(MaxNLocator(10))

        primary_lines = []
        for i, (each_name, each_plot, each_value) in enumerate(zip(self.names_plots_primary, self.plots_primary, self.values_primary)):
            each_plot.append(each_value)
            del(each_plot[:-self.size_window])
            l, = self.ax_primary.plot(self.time_range, each_plot, label=f"{each_name:s}")
            primary_lines.append(l)

        for i, (each_name, each_plot, each_value) in enumerate(zip(self.names_plots_secondary, self.plots_secondary, self.values_secondary)):
            each_plot.append(each_value)
            del(each_plot[:-self.size_window])
            self.ax_secondary.plot(self.time_range, each_plot, label=f"{each_name:s}", alpha=.2)

        if self.limits_primary:
            self.ax_primary.set_ylim(self.limits_primary)
        else:
            self._set_limits(self.ax_primary, self.plots_primary)

        if self.limits_secondary:
            self.ax_secondary.set_ylim(self.limits_secondary)
        else:
            self._set_limits(self.ax_secondary, self.plots_secondary)

        pyplot.setp(self.ax_primary.xaxis.get_majorticklabels(), rotation=90, ha="right", rotation_mode="anchor")

        pyplot.legend(primary_lines, tuple(line.get_label() for line in primary_lines))
        pyplot.tight_layout()
        pyplot.pause(.05)

        self.iterations_since_draw = 0