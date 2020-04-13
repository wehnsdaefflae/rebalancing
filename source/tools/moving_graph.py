import datetime
import time
from typing import Sequence, Optional, Tuple, Dict

from matplotlib import pyplot, dates
from matplotlib.ticker import MaxNLocator

from source.tools.functions import smear

NAME_Y = str
NAMES_PLOTS = Sequence[str]
MOVING_AVERAGE = Optional[bool]
LIMITS = Optional[Tuple[float, float]]
INFO_AXIS = Tuple[NAME_Y, NAMES_PLOTS, MOVING_AVERAGE, LIMITS]


class MovingGraph:
    def __init__(self,
                 axes_info: Sequence[INFO_AXIS],
                 size_window: int,
                 interval_draw_ms: int = 1000,
                 ):
        assert 5000. >= interval_draw_ms >= 0.
        no_subplots = len(axes_info)
        self.fig, self.subplots = pyplot.subplots(nrows=no_subplots, ncols=1, sharex="all")

        self.names_axes, self.names_plots, self.moving_averages, self.limits = zip(*axes_info)

        self.interval_draw_ms = interval_draw_ms
        self.size_window = size_window

        self.no_axes = len(self.names_axes)
        self.no_plots = tuple(len(names) for names in self.names_plots)

        self.time_current = None
        self.values_current = tuple({each_name: 0. for each_name in each_names_plot} for each_names_plot in self.names_plots)

        self.time_window = []
        self.values_windows = tuple({each_name: [] for each_name in each_names_plot} for each_names_plot in self.names_plots)

        self.iterations_since_draw = [0 for _ in self.names_axes]
        self.time_last = -1.

    def add_snapshot(self, now: datetime.datetime, points: Sequence[Dict[str, float]]):
        assert len(points) == self.no_axes

        self.time_current = now
        for i, (value_current, limits, is_moving_average, points_axis) in enumerate(zip(self.values_current, self.limits, self.moving_averages, points)):
            if is_moving_average is None:
                for name_plot in value_current:
                    value_current[name_plot] = points_axis[name_plot]
            else:
                for name_plot, value_last in value_current.items():
                    value_current[name_plot] = smear(value_last, points_axis[name_plot], self.iterations_since_draw[i])

            self.iterations_since_draw[i] += 1

        time_now = time.time() * 1000.
        if self.time_last < 0. or time_now - self.time_last >= self.interval_draw_ms or 0 >= self.interval_draw_ms:
            self.draw()
            self.time_last = time_now

    def _set_limits(self, index_subplot: int):
        windows = self.values_windows[index_subplot]
        val_min = min(min(each_plot) for each_plot in windows.values())
        val_max = max(max(each_plot) for each_plot in windows.values())
        val_d = .2 * (val_max - val_min)

        axis_subplot = self.subplots[index_subplot]
        axis_subplot.set_ylim([val_min - val_d, val_max + val_d])

    def _draw_subplot(self, index_subplot: int):
        axis_subplot = self.subplots[index_subplot]
        axis_subplot.clear()

        axis_subplot.set_xlabel("time")
        axis_subplot.set_ylabel(self.names_axes[index_subplot])

        axis_subplot.ticklabel_format(useOffset=False)

        axis_subplot.xaxis.set_major_formatter(dates.DateFormatter("%d.%m.%Y %H:%M"))
        axis_subplot.xaxis.set_major_locator(MaxNLocator(10))

        lines = []
        window_subplot = self.values_windows[index_subplot]
        current_subplot = self.values_current[index_subplot]
        for i, (each_name, each_plot, each_value) in enumerate(zip(self.names_plots[index_subplot], window_subplot.values(), current_subplot.values())):
            each_plot.append(each_value)
            del(each_plot[:-self.size_window])
            l, = axis_subplot.plot(self.time_window, each_plot, label=f"{each_name:s}")
            lines.append(l)

        if self.limits[index_subplot] is not None:
            axis_subplot.set_ylim(ymin=min(self.limits[index_subplot]), ymax=max(self.limits[index_subplot]))
        else:
            self._set_limits(index_subplot)

        pyplot.setp(axis_subplot.xaxis.get_majorticklabels(), rotation=90, ha="right", rotation_mode="anchor")

        if self.moving_averages[index_subplot] is not None and self.moving_averages[index_subplot]:
            self.iterations_since_draw[index_subplot] = 0

        axis_subplot.legend(lines, tuple(each_line.get_label() for each_line in lines))

    def draw(self):
        self.time_window.append(self.time_current)
        del(self.time_window[:-self.size_window])

        for i in range(self.no_axes):
            self._draw_subplot(i)

        pyplot.tight_layout()
        pyplot.pause(.05)
