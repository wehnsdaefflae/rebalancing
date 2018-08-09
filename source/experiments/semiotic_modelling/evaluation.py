import datetime
from typing import List, Tuple, Sequence

from matplotlib import pyplot
from matplotlib.colors import hsv_to_rgb
from matplotlib.dates import date2num
from matplotlib.patches import Rectangle

from source.experiments.semiotic_modelling.modelling import BASIC_OUT, EXAMPLE, MODEL, STATE, TRACE, TIME
from source.tools.helper_functions import distribute_circular, normalize
from source.tools.timer import Timer


class SimulationStats:
    def __init__(self, dim: int):
        self.input_values = tuple([] for _ in range(dim))               # type: Tuple[List[float], ...]
        self.target_values = tuple([] for _ in range(dim))              # type: Tuple[List[float], ...]
        self.output_values = tuple([] for _ in range(dim))              # type: Tuple[List[float], ...]

        self.states = tuple([] for _ in range(dim))                     # type: Tuple[List[Tuple[int, ...]]]
        self.model_structures = []                                      # type: List[List[int, ...]]

        self.probabilities = tuple([] for _ in range(dim))              # type: Tuple[List[float], ...]

        self.errors = tuple([] for _ in range(dim))          # type: Tuple[List[float], ...]

        self.time_axis = []

    def log(self, time: TIME, examples: List[EXAMPLE], output_values: List[BASIC_OUT],
            probabilities: Tuple[float, ...], model: MODEL, situations: Tuple[STATE, ...]):

        self.time_axis.append(time)

        for _i, (each_example, output_value) in enumerate(zip(examples, output_values)):
            input_value, target_value = each_example                # type: float, float

            input_list = self.input_values[_i]                      # type: List[float]
            input_list.append(input_value)
            target_list = self.target_values[_i]                    # type: List[float]
            target_list.append(target_value)

            output_list = self.output_values[_i]                    # type: List[float]
            output_list.append(output_value)                        # type: List[float]

            error = target_value - output_value
            self.errors[_i].append(error)

        for _i, each_probability in enumerate(probabilities):
            probability_list = self.probabilities[_i]               # type: List[float]
            probability_list.append(each_probability)

        for _i, each_situation in enumerate(situations):
            state_list = self.states[_i]                      # type: List[Tuple[int, ...]]
            state_list.append(tuple(each_situation))

        self.model_structures.append([len(_x) for _x in model])

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

        fig, (ax1, ax2, ax3) = pyplot.subplots(3, sharex="all")

        ax11 = ax1.twinx()

        for _i, each_state_list in enumerate(self.states):
            # segments = SimulationStats._get_segments(self.time_axis, each_state_list)
            # SimulationStats._plot_h_stacked_bars(ax1, segments)
            ax1.plot(self.time_axis, normalize(self.probabilities[_i]), label="probability {:d}".format(_i), alpha=.3)

        max_levels = max(len(_x) for _x in self.model_structures)
        # ax1.set_ylim(0., max_levels)
        ax1.set_ylabel("certainty")

        class UpdatingRect(Rectangle):
            def __call__(self, ax: pyplot.Axes.axes):
                try:
                    ax.set_ylim(0., max_levels)
                except RecursionError as e:
                    pass

        reset_y = UpdatingRect([0, 0], 0, 0, facecolor="None", edgecolor="black", linewidth=1.)
        ax1.callbacks.connect("ylim_changed", reset_y)
        ax1.legend()

        for _i, (each_input_list, each_target_list, each_output_list) in enumerate(zip(self.input_values, self.target_values, self.output_values)):
            # ax11.plot(self.time_axis, each_input_list, label="input {:d}".format(_i), alpha=.75)
            ax11.plot(self.time_axis, normalize(each_output_list), label="output {:d}".format(_i), alpha=.75)
            ax11.plot(self.time_axis, normalize(each_target_list), label="target {:d}".format(_i), alpha=.75)
        ax11.set_ylabel("values")
        ax11.legend()

        len_last_structure = len(self.model_structures[-1])
        for each_structure in self.model_structures:
            while len(each_structure) < len_last_structure:
                each_structure.append(0)
        transposed = list(zip(*self.model_structures))
        ax2.set_ylabel("model size")
        ax2.stackplot(self.time_axis, *transposed)

        for _i, each_errors in enumerate(self.errors):
            cumulative_error = []
            for each_error in each_errors:
                squared_error = each_error ** 2.
                delta = squared_error if len(cumulative_error) < 1 else cumulative_error[-1] + squared_error
                cumulative_error.append(delta)
            # TODO: normalize error properly (normalized variance/deviance?)
            # https://de.wikipedia.org/wiki/Variationskoeffizient
            ax3.plot(self.time_axis, cumulative_error, label="cumulative error {:d}".format(_i))
        ax3.set_ylabel("error")
        ax3.legend()
        pyplot.tight_layout()

        if is_datetime:
            ax1.xaxis_date()
            ax11.xaxis_date()
            ax2.xaxis_date()
            ax3.xaxis_date()
        pyplot.show()
