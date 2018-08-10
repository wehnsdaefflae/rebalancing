import datetime
from typing import List, Tuple, Sequence

from matplotlib import pyplot
from matplotlib.colors import hsv_to_rgb
from matplotlib.dates import date2num
from matplotlib.patches import Rectangle

from source.experiments.semiotic_modelling.methods import RationalSemioticModel
from source.experiments.semiotic_modelling.modelling import BASIC_OUT, EXAMPLE, MODEL, STATE, TRACE, TIME
from source.tools.helper_functions import distribute_circular, normalize
from source.tools.timer import Timer


class ComparativeEvaluation:
    def __init__(self, method_names: Sequence[str]):
        self.method_names = method_names
        self.outputs = tuple([] for _ in method_names)
        self.target_list = []
        self.no_methods = len(method_names)
        self.time_axis = []
        self.errors = tuple([] for _ in method_names)

    def log(self, time: TIME, output_values: Sequence[Tuple[float, ...]], target_values: Tuple[float, ...]):
        assert len(output_values) == self.no_methods
        self.time_axis.append(time)
        self.target_list.append(target_values)
        for _i, each_output in enumerate(output_values):
            output_list = self.outputs[_i]
            output_list.append(each_output)
            error_list = self.errors[_i]
            error_list.append(abs(sum(each_output) - sum(target_values)))

    def plot(self):
        type_set = {type(_x) for _x in self.time_axis}
        assert len(type_set) == 1
        time_type, = type_set

        if time_type == datetime.datetime:
            is_datetime = True
            for _i, each_datetime in enumerate(self.time_axis):
                self.time_axis[_i] = date2num(each_datetime)
        else:
            is_datetime = False

        fig, (ax1, ax2) = pyplot.subplots(2, sharex="all")
        fig.suptitle("Comparative Evaluation")

        ax1.set_ylabel("output values")
        ax2.set_ylabel("cumulative squared error")

        for _i, method_name in enumerate(self.method_names):
            ax1.plot(self.time_axis, self.outputs[_i], label=method_name, alpha=.5)
            cumulative_error = []
            for each_error in self.errors[_i]:
                if len(cumulative_error) < 1:
                    cumulative_error.append(each_error ** 2.)
                else:
                    cumulative_error.append(each_error ** 2. + cumulative_error[-1])
            ax2.plot(self.time_axis, cumulative_error, label=method_name, alpha=.5)
        ax1.plot(self.time_axis, [sum(each_target) for each_target in self.target_list], label="target", alpha=.5)
        ax1.legend()
        ax2.legend()

        if is_datetime:
            ax1.xaxis_date()
            ax2.xaxis_date()
        pyplot.show()

    def save(self, file_path: str):
        raise NotImplementedError


class QualitativeEvaluation:
    def __init__(self, dim: int):
        self.target_values = tuple([] for _ in range(dim))              # type: Tuple[List[float], ...]
        self.output_values = tuple([] for _ in range(dim))              # type: Tuple[List[float], ...]

        self.states = tuple([] for _ in range(dim))                     # type: Tuple[List[Tuple[int, ...]]]
        self.model_structures = []                                      # type: List[List[int]

        self.probabilities = tuple([] for _ in range(dim))              # type: Tuple[List[float], ...]

        self.time_axis = []

    def log(self, time: TIME, input_values: Tuple[Tuple[float, ...], ...], target_values: Tuple[float, ...], model: RationalSemioticModel):

        self.time_axis.append(time)

        output_values = model.predict(input_values)
        certainty = model.get_certainty(input_values, target_values)
        structure = model.get_structure()
        states = model.get_states()

        for _i, (target_value, output_value) in enumerate(zip(target_values, output_values)):
            target_list = self.target_values[_i]                    # type: List[float]
            target_list.append(target_value)

            output_list = self.output_values[_i]                    # type: List[float]
            output_list.append(output_value)                        # type: List[float]

        for _i, each_probability in enumerate(certainty):
            probability_list = self.probabilities[_i]               # type: List[float]
            probability_list.append(each_probability)

        for _i, each_situation in enumerate(states):
            state_list = self.states[_i]                      # type: List[Tuple[int, ...]]
            state_list.append(tuple(each_situation))

        self.model_structures.append(list(structure))

    def save(self, model: MODEL, traces: Tuple[TRACE], situations: Tuple[STATE], file_path: str):
        pass
        # raise NotImplementedError()

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

    def plot(self, plot_segments: bool = False):
        type_set = {type(_x) for _x in self.time_axis}
        assert len(type_set) == 1
        time_type, = type_set
        if time_type == datetime.datetime:
            is_datetime = True
            for _i, each_datetime in enumerate(self.time_axis):
                self.time_axis[_i] = date2num(each_datetime)
        else:
            is_datetime = False

        fig, (ax1, ax2) = pyplot.subplots(2, sharex="all")
        fig.suptitle("Qualitative Evaluation")
        ax11 = ax1.twinx()

        for _i, each_state_list in enumerate(self.states):
            if plot_segments:
                segments = QualitativeEvaluation._get_segments(self.time_axis, each_state_list)
                QualitativeEvaluation._plot_h_stacked_bars(ax1, segments)
            ax1.plot(self.time_axis, normalize(self.probabilities[_i]), label="certainty {:d}".format(_i), alpha=.3)

        max_levels = max(len(_x) for _x in self.model_structures)
        if plot_segments:
            ax1.set_ylim(0., max_levels)
        ax1.set_ylabel("certainty")

        class UpdatingRect(Rectangle):
            def __call__(self, ax: pyplot.Axes.axes):
                try:
                    ax.set_ylim(0., max_levels)
                except RecursionError as e:
                    pass

        if plot_segments:
            reset_y = UpdatingRect([0, 0], 0, 0, facecolor="None", edgecolor="black", linewidth=1.)
            ax1.callbacks.connect("ylim_changed", reset_y)
        ax1.legend(loc="upper left")

        for _i, (each_target_list, each_output_list) in enumerate(zip(self.target_values, self.output_values)):
            ax11.plot(self.time_axis, each_output_list, label="output {:d}".format(_i), alpha=.75)
            ax11.plot(self.time_axis, each_target_list, label="target {:d}".format(_i), alpha=.75)
        ax11.set_ylabel("values")
        ax11.legend(loc="upper right")

        len_last_structure = len(self.model_structures[-1])
        for each_structure in self.model_structures:
            while len(each_structure) < len_last_structure:
                each_structure.append(0)
        transposed = list(zip(*self.model_structures))
        ax2.set_ylabel("model size")
        ax2.stackplot(self.time_axis, *transposed)

        if is_datetime:
            ax1.xaxis_date()
            ax11.xaxis_date()
            ax2.xaxis_date()
        pyplot.show()
