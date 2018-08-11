import datetime
import math
from typing import List, Tuple, Sequence

from matplotlib import pyplot
from matplotlib.colors import hsv_to_rgb
from matplotlib.dates import date2num
from matplotlib.patches import Rectangle

from source.experiments.semiotic_modelling.modelling import MODEL, STATE, TRACE, TIME
from source.tools.helper_functions import distribute_circular, smoothing_generator
from source.tools.timer import Timer


class ComparativeEvaluation:
    def __init__(self, output_dimension: int, method_names: Sequence[str]):
        self.output_dimension = output_dimension
        self.targets = tuple([] for _ in range(output_dimension))

        self.method_names = method_names
        self.outputs = {each_method: tuple([] for _ in range(output_dimension)) for each_method in method_names}

        self.errors = tuple([] for _ in method_names)
        self.time_axis = []

    def log_predictors(self, time: TIME, all_output_values: Sequence[Tuple[float, ...]], target_value: Tuple[float, ...]):
        assert len(all_output_values) == len(self.method_names)
        self.time_axis.append(time)
        for _i, each_target in enumerate(target_value):
            target_list = self.targets[_i]
            target_list.append(each_target)

        for _i, output_value in enumerate(all_output_values):
            assert len(output_value) == len(target_value)
            each_method = self.method_names[_i]
            output_tuples = self.outputs[each_method]
            for _j, _o in enumerate(output_value):
                output_list = output_tuples[_j]
                output_list.append(_o)
            error_list = self.errors[_i]

            each_error = math.sqrt(sum((_o - _t) ** 2. for (_o, _t) in zip(output_value, target_value)))
            error_list.append(each_error)

    def _convert_time(self):
        type_set = {type(_x) for _x in self.time_axis}
        assert len(type_set) == 1
        time_type, = type_set
        if time_type == datetime.datetime:
            is_datetime = True
            for _i, each_datetime in enumerate(self.time_axis):
                self.time_axis[_i] = date2num(each_datetime)
        else:
            is_datetime = False
        return is_datetime

    def _plot_errors(self, axis: pyplot.Axes.axes):
        for _i, method_name in enumerate(self.method_names):
            cumulative_error = []
            for each_error in self.errors[_i]:
                if len(cumulative_error) < 1:
                    cumulative_error.append(each_error ** 2.)
                else:
                    cumulative_error.append(each_error ** 2. + cumulative_error[-1])
            axis.plot(self.time_axis, cumulative_error, label=method_name, alpha=.5)
        axis.set_ylabel("cumulative squared error")
        axis.legend()

    def _plot_outputs(self, axis: pyplot.Axes.axes):
        for _j in range(self.output_dimension):
            axis.plot(self.time_axis, self.targets[_j], label="target {:d}".format(_j))
        for _i, method_name in enumerate(self.method_names):
            each_outputs = self.outputs[method_name]
            for _j in range(self.output_dimension):
                axis.plot(self.time_axis, each_outputs[_j], label="{:s} {:d}".format(method_name, _j), alpha=.5)
        axis.set_ylabel("output values")
        axis.legend()

    def plot(self):
        # self.time_axis.append(datetime.datetime.fromtimestamp(time_stamp, tz=tzutc()))
        is_datetime = self._convert_time()

        fig, (ax1, ax2) = pyplot.subplots(2, sharex="all")
        fig.suptitle("Comparative Evaluation")

        self._plot_outputs(ax1)
        self._plot_errors(ax2)

        if is_datetime:
            ax1.xaxis_date()
            ax2.xaxis_date()
        pyplot.show()

    def save(self, file_path: str):
        raise NotImplementedError


class QualitativeEvaluationSingleSequence(ComparativeEvaluation):
    def __init__(self, output_dimension: int, method_names: Sequence[str]):
        super().__init__(output_dimension, method_names)
        self.states = []                                                                    # type: List[Tuple[int, ...]]
        self.model_structures = []                                                          # type: List[List[int]

        self.certainties = []                                                               # type: List[float]

    def log_semiotic_model(self, time: TIME,
                           target_value: Tuple[float, ...], output_value: Tuple[float, ...], certainty: float,
                           structure: Tuple[int, ...], state: Tuple[int, ...]):
        assert len(target_value) == len(output_value) == self.output_dimension

        self.log_predictors(time, [output_value], target_value)

        self.states.append(state)
        self.model_structures.append(list(structure))

        self.certainties.append(certainty)

    def save(self, file_path: str):
        raise NotImplementedError()

    @staticmethod
    def _get_segments(time_axis: Sequence[TIME], states: List[Tuple[int, ...]]) -> Tuple[Sequence[Tuple[int, TIME]], ...]:
        assert(len(time_axis) == len(states))
        max_level = max(len(_x) for _x in states)
        levels = tuple([] for _ in range(max_level))

        for _j, (each_time, each_context) in enumerate(zip(time_axis, states)):
            for _i, each_level in enumerate(levels):
                each_shape = each_context[_i] if _i < len(each_context) else -1

                if 0 < len(each_level):
                    _, last_shape = each_level[-1]
                else:
                    last_shape = -1

                if each_shape != last_shape:
                    data_point = each_time, each_shape
                    each_level.append(data_point)

            if Timer.time_passed(2000):
                print("Finished {:5.2f}% of segmenting...".format(100. * _j / len(time_axis)))

        return levels

    @staticmethod
    def _plot_h_stacked_bars(axis: pyplot.Axes.axes, segments: Sequence[Sequence[Tuple[TIME, float]]]):
        for _i, each_level in enumerate(segments):
            for _x in range(len(each_level) - 1):
                each_left, each_shape = each_level[_x]
                each_right, _ = each_level[_x + 1]
                each_width = each_right - each_left
                hsv = distribute_circular(each_shape), .2, 1.
                axis.barh(_i, each_width, height=1., align="edge", left=each_left, color=hsv_to_rgb(hsv))

                if Timer.time_passed(2000):
                    print("Finished {:5.2f}% of plotting level {:d}/{:d}...".format(100. * _x / (len(each_level) - 1), _i, len(segments)))

    def _convert_time(self):
        type_set = {type(_x) for _x in self.time_axis}
        assert len(type_set) == 1
        time_type, = type_set
        if time_type == datetime.datetime:
            is_datetime = True
            for _i, each_datetime in enumerate(self.time_axis):
                self.time_axis[_i] = date2num(each_datetime)
        else:
            is_datetime = False
        return is_datetime

    def plot(self, plot_segments: bool = False):
        # self.time_axis.append(datetime.datetime.fromtimestamp(time_stamp, tz=tzutc()))

        is_datetime = self._convert_time()

        fig, (ax1, ax2) = pyplot.subplots(2, sharex="all")
        ax11 = ax1.twinx()
        fig.suptitle("Qualitative Evaluation")

        if plot_segments:
            self._plot_segments(ax1)

        self._plot_certainty(ax1)
        self._plot_outputs(ax11)

        self._plot_structure(ax2)

        if is_datetime:
            ax1.xaxis_date()
            ax11.xaxis_date()
            ax2.xaxis_date()
        pyplot.show()

    def _plot_structure(self, ax2):
        len_last_structure = len(self.model_structures[-1])
        for each_structure in self.model_structures:
            while len(each_structure) < len_last_structure:
                each_structure.append(0)
        transposed = list(zip(*self.model_structures))
        ax2.set_ylabel("model size")
        ax2.stackplot(self.time_axis, *transposed)

    def _plot_certainty(self, ax1):
        s = 1000
        smooth_certainty = smoothing_generator(self.certainties, s)
        ax1.plot(self.time_axis, list(smooth_certainty), label="certainty (smooth {:d})".format(s), alpha=.3)
        ax1.set_ylabel("certainty")
        ax1.legend(loc="upper left")

    def _plot_segments(self, ax1):
        segments = QualitativeEvaluationSingleSequence._get_segments(self.time_axis, self.states)
        QualitativeEvaluationSingleSequence._plot_h_stacked_bars(ax1, segments)
        max_levels = max(len(_x) for _x in self.model_structures)

        ax1.set_ylim(0., max_levels)

        class UpdatingRect(Rectangle):
            def __call__(self, ax: pyplot.Axes.axes):
                try:
                    ax.set_ylim(0., max_levels)
                except RecursionError as e:
                    pass

        reset_y = UpdatingRect([0, 0], 0, 0, facecolor="None", edgecolor="black", linewidth=1.)
        ax1.callbacks.connect("ylim_changed", reset_y)
