import datetime
from typing import List, Tuple, Sequence

from matplotlib import pyplot
from matplotlib.colors import hsv_to_rgb
from matplotlib.dates import date2num
from matplotlib.patches import Rectangle

from source.experiments.semiotic_modelling.modelling import BASIC_OUT, EXAMPLE, MODEL, SITUATION, STATE, TIME
from source.tools.helper_functions import distribute_circular
from source.tools.timer import Timer


class SimulationStats:
    def __init__(self, dim: int):
        self.input_values = tuple([] for _ in range(dim))               # type: Tuple[List[float], ...]
        self.target_values = tuple([] for _ in range(dim))              # type: Tuple[List[float], ...]
        self.output_values = tuple([] for _ in range(dim))              # type: Tuple[List[float], ...]

        self.contexts = tuple([] for _ in range(dim))                   # type: Tuple[List[Tuple[int, ...]]]
        self.model_structures = []                                      # type: List[List[int, ...]]

        self.cumulative_errors = tuple([] for _ in range(dim))          # type: Tuple[List[float], ...]

        self.time_axis = []

    def log(self, time: TIME, examples: List[EXAMPLE], output_values: List[BASIC_OUT], model: MODEL, situations: Tuple[SITUATION, ...]):
        self.time_axis.append(time)

        for _i, (each_example, output_value) in enumerate(zip(examples, output_values)):
            input_value, target_value = each_example                # type: float, float

            input_list = self.input_values[_i]                      # type: List[float]
            input_list.append(input_value)
            target_list = self.target_values[_i]                    # type: List[float]
            target_list.append(target_value)

            output_list = self.output_values[_i]                    # type: List[float]
            output_list.append(output_value)                        # type: List[float]

            error = (output_value - target_value) ** 2.
            cumulative_error = error + (self.cumulative_errors[_i][-1] if 0 < len(self.cumulative_errors[_i]) else 0.)
            self.cumulative_errors[_i].append(cumulative_error)

        for _i, each_situation in enumerate(situations):
            situation_list = self.contexts[_i]                      # type: List[Tuple[int, ...]]
            situation_list.append(tuple(each_situation))

        self.model_structures.append([len(_x) for _x in model])

    def save(self, model: MODEL, states: Tuple[STATE], file_path: str):
        pass
        # raise NotImplementedError()

    @staticmethod
    def _get_segments(time_axis: Sequence[TIME], contexts: List[Tuple[int, ...]]) -> Tuple[Sequence[Tuple[int, TIME]], ...]:
        assert(len(time_axis) == len(contexts))
        max_level = max(len(_x) for _x in contexts)
        levels = tuple([] for _ in range(max_level))

        for _j, (each_time, each_context) in enumerate(zip(time_axis, contexts)):
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
    def _plot_h_stacked_bars(axis: pyplot.Axes.axes, segmentations: Sequence[Sequence[Tuple[TIME, float]]]):
        for _i, each_level in enumerate(segmentations):
            for _x in range(len(each_level) - 1):
                each_left, each_shape = each_level[_x]
                each_right, _ = each_level[_x + 1]
                each_width = each_right - each_left
                hsv = distribute_circular(each_shape), .2, 1.
                axis.barh(_i, each_width, height=1., align="edge", left=each_left, color=hsv_to_rgb(hsv))

                if Timer.time_passed(2000):
                    print("Finished {:5.2f}% of plotting level {:d}/{:d}...".format(100. * _x / (len(each_level) - 1), _i, len(segmentations)))

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

        for _i, each_context in enumerate(self.contexts):
            segments = SimulationStats._get_segments(self.time_axis, each_context)
            SimulationStats._plot_h_stacked_bars(ax1, segments)

        max_levels = max(len(_x) for _x in self.model_structures)
        ax1.set_ylim(0., max_levels)
        ax1.set_ylabel("representations")

        class UpdatingRect(Rectangle):
            def __call__(self, ax: pyplot.Axes.axes):
                try:
                    ax.set_ylim(0., max_levels)
                except RecursionError as e:
                    pass

        reset_y = UpdatingRect([0, 0], 0, 0, facecolor="None", edgecolor="black", linewidth=1.)
        ax1.callbacks.connect("ylim_changed", reset_y)

        for _i, (each_input_list, each_target_list, each_output_list) in enumerate(zip(self.input_values, self.target_values, self.output_values)):
            ax11.plot(self.time_axis, each_input_list, label="input {:d}".format(_i))
            ax11.plot(self.time_axis, each_target_list, label="target {:d}".format(_i))
            ax11.plot(self.time_axis, each_output_list, label="output {:d}".format(_i))
        ax11.set_ylabel("values")
        ax11.legend()

        len_last_structure = len(self.model_structures[-1])
        for each_structure in self.model_structures:
            while len(each_structure) < len_last_structure:
                each_structure.append(0)
        transposed = list(zip(*self.model_structures))
        ax2.set_ylabel("model size")
        ax2.stackplot(self.time_axis, *transposed)

        for _i, each_cumulative_error in enumerate(self.cumulative_errors):
            ax3.plot(self.time_axis, each_cumulative_error, label="cumulative error {:d}".format(_i))
        ax3.set_ylabel("error")
        ax3.legend()
        pyplot.tight_layout()

        if is_datetime:
            ax1.xaxis_date()
            ax11.xaxis_date()
            ax2.xaxis_date()
            ax3.xaxis_date()
        pyplot.show()
