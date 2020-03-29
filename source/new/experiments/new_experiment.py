import time
from typing import Sequence

from matplotlib import pyplot

from source.new.learning.approximation import Approximation


class MovingGraph:
    def __init__(self, names_plots: Sequence[str], size_window: int, interval_ms: int = 1000):
        assert 5000. >= interval_ms >= 0.
        self.names_plots = names_plots
        self.plots = tuple([] for _ in names_plots)
        self.size_window = size_window
        self.fig, self.ax = pyplot.subplots()
        self.timestep_plot = 0
        self.interval_ms = interval_ms
        self.time_last = -1.

    def add_snapshot(self, points: Sequence[float]):
        for each_plot, each_value in zip(self.plots, points):
            each_plot.append(each_value)
            del(each_plot[:-self.size_window])
        self.timestep_plot += 1

        time_this = time.time() * 1000.
        if time_this - self.time_last >= self.interval_ms or self.time_last < 0.:
            self.draw()
            self.time_last = time_this

    def draw(self):
        self.ax.clear()
        x_coordinates = list(range(max(0, self.timestep_plot - self.size_window), self.timestep_plot))
        for each_name, each_plot in zip(self.names_plots, self.plots):
            self.ax.plot(x_coordinates, each_plot, label=f"{each_name:s}")

        val_min = min(min(each_plot) for each_plot in self.plots)
        val_max = max(max(each_plot) for each_plot in self.plots)

        self.ax.set_ylim([val_min - .2 * (val_max - val_min),  val_max + .2 * (val_max - val_min)])

        pyplot.legend()
        pyplot.tight_layout()
        pyplot.pause(.05)


class ModulesProcessing:
    def process(self, input_values: Sequence[float], target_values: Sequence[float], output_values: Sequence[float], error: float, name: str):
        raise NotImplementedError()


class ModulePlotting(MovingGraph, ModulesProcessing):
    def __init__(self, no_plots: int, size_window: int):
        super().__init__(no_plots, size_window)

    def process(self, input_values: Sequence[float], target_values: Sequence[float], output_values: Sequence[float], error: float, name: str):
        pass


class Experiment:
    def __int__(self, approximations: Sequence[Approximation], processing: Sequence[ModulesProcessing]):
        self.approximations = approximations
        self.processing = processing

    def _next_example(self):
        raise NotImplementedError()

    def _error(self, output_value: Sequence[float], target_value: Sequence[float]) -> float:
        raise NotImplementedError()

    def start(self):
        example = self._next_example()
        if example is None:
            return
        input_value, target_value = example
        for each_approximation in self.approximations:
            output_value = each_approximation.output(input_value)
            error = self._error(output_value, target_value)
            for each_module in self.processing:
                each_module.process(input_value, target_value, output_value, error, str(each_approximation))
