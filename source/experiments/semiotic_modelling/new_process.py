import datetime
import json
from math import sin, cos
from typing import Union, TypeVar, List, Tuple, Iterable, Dict, Optional, Generator, Sequence, Type, Callable

from dateutil.tz import tzutc
from matplotlib import pyplot

from source.data.data_generation import series_generator
from source.experiments.semiotic_modelling.content import Content, SymbolicContent, RationalContent
from source.tools.timer import Timer


BASIC_IN = TypeVar("BASIC_SHAPE_IN")
BASIC_OUT = TypeVar("BASIC_SHAPE_OUT")
EXAMPLE = Tuple[BASIC_IN, BASIC_OUT]

ABSTRACT_SHAPE = int                                        # TODO: make generic hashable

APPEARANCE = Union[BASIC_IN, BASIC_OUT, ABSTRACT_SHAPE]
HISTORY = Union[List[APPEARANCE], Tuple[APPEARANCE, ...]]


LEVEL = Dict[APPEARANCE, Content]
MODEL = List[LEVEL]
SITUATION = List[APPEARANCE]
STATE = List[HISTORY]

TIME = TypeVar("TIME")


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

    def plot(self):
        fig, (ax1, ax2, ax3) = pyplot.subplots(3, sharex="all")

        for _i, (each_input_list, each_target_list, each_output_list) in enumerate(zip(self.input_values, self.target_values, self.output_values)):
            ax1.plot(self.time_axis, each_input_list, label="input {:d}".format(_i))
            ax1.plot(self.time_axis, each_target_list, label="target {:d}".format(_i))
            ax1.plot(self.time_axis, each_output_list, label="output {:d}".format(_i))
        ax1.legend()

        len_last_structure = len(self.model_structures[-1])
        for each_structure in self.model_structures:
            while len(each_structure) < len_last_structure:
                each_structure.append(0)
        transposed = list(zip(*self.model_structures))
        ax2.stackplot(self.time_axis, *transposed)

        for _i, each_cumulative_error in enumerate(self.cumulative_errors):
            ax3.plot(self.time_axis, each_cumulative_error, label="cumulative error {:d}".format(_i))
        ax3.legend()
        pyplot.show()


def predict(model: MODEL, situation: SITUATION, input_value: BASIC_IN) -> Optional[BASIC_OUT]:
    len_model, len_situation = len(model), len(situation)
    assert len_model >= len_situation
    if len_situation < 1:
        return None
    content_shape = situation[0]
    if len_model < 1:
        return None
    base_layer = model[0]
    content = base_layer.get(content_shape)
    assert content is not None
    return content.predict(input_value, default=None)


def get_content(model: MODEL, situation: SITUATION, level: int) -> Content:
    assert level < len(model)
    layer = model[level]        # type: LEVEL

    assert level < len(situation)
    shape = situation[level]    # type: APPEARANCE

    content = layer.get(shape)  # type: Content
    assert content is not None
    return content


def update_states(states: Tuple[STATE, ...], situations: Tuple[SITUATION, ...], history_length: int):
    assert len(states) == len(situations)

    for each_state, each_situation in zip(states, situations):
        len_state, len_situation = len(each_state), len(each_situation)
        assert len_state == len_situation

        for each_shape, each_layer in zip(each_situation, each_state):
            each_layer.append(each_shape)
            while history_length < len(each_layer):
                each_layer.pop(0)


def generate_content(model: MODEL, situations: Tuple[SITUATION, ...], base_content: Type[Content], alpha: float):
    len_model = len(model)
    len_set = set(len(_x) for _x in situations)
    assert len(len_set) == 1
    len_situation, = len_set
    assert len_situation == len_model or len_situation == len_model + 1
    for _i in range(len_situation):
        situations_with_new_content = [_j for _j, each_situation in enumerate(situations) if each_situation[_i] == -1]
        if 0 < len(situations_with_new_content):
            if _i == len_model:
                each_layer = dict()
                model.append(each_layer)
            else:
                each_layer = model[_i]
            new_shape = len(each_layer)
            each_layer[new_shape] = base_content(new_shape, alpha) if _i < 1 else SymbolicContent(new_shape, alpha)
            for each_index in situations_with_new_content:
                each_situation = situations[each_index]
                each_situation[_i] = new_shape


def adapt_content(model: MODEL, states: Tuple[STATE, ...], situations: Tuple[SITUATION, ...]):
    len_states, len_situations = len(states), len(situations)
    assert len_states == len_situations

    len_model = len(model)
    for each_state, each_situation in zip(states, situations):
        len_state, len_situation = len(each_state), len(each_situation)
        assert len_model == len_state == len_situation

        for _i in range(len_situation - 1):
            content = get_content(model, each_situation, _i + 1)
            shape_out = each_situation[_i]
            history = each_state[_i]
            # at level >= 2 add symbolic shape from situation level == 1 to condition
            content.adapt(tuple(history), shape_out)


def update_situation(situation: SITUATION, shape: BASIC_IN, target_value: BASIC_OUT, state: STATE, model: MODEL, sigma: Callable[[int, int], float]):
    len_model = len(model)
    level = 0                                                                                                   # type: int

    # for each_shape in situation:
    while level < len_model:
        s = sigma(level, len(model[level]))
        content = get_content(model, situation, level)                                                          # type: Content
        if content.probability(shape, target_value) >= s:
            break

        layer = model[level]
        abstract_shape = tuple(state[level])
        if level + 1 < len_model:
            context = get_content(model, situation, level + 1)                                                      # type: Content
            # symbolic content transitions at level >= 2 with conditions from history and symbolic content from level == 1
            abstract_target = context.predict(abstract_shape)                                                                      # type: APPEARANCE
            if abstract_target is not None:
                content = layer[abstract_target]                                                                              # type: Content
                if content.probability(shape, target_value) >= s:
                    situation[level] = abstract_target
                    target_value = abstract_target
                    shape = abstract_shape
                    level += 1
                    continue

        content = max(layer.values(), key=lambda _x: _x.probability(shape, target_value))               # type: Content
        abstract_target = hash(content)                                                                           # type: APPEARANCE
        if content.probability(shape, target_value) >= s:
            situation[level] = abstract_target
            target_value = abstract_target
            shape = abstract_shape                                                                           # type: HISTORY
            level += 1
            continue

        situation[level] = -1                                                                           # type: APPEARANCE
        level += 1


def generate_layer(model: MODEL, situations: Tuple[SITUATION, ...]):
    len_set = {len(each_situation) for each_situation in situations}
    assert len(len_set) == 1
    len_situation, = len_set
    assert len_situation == len(model)
    
    if -1 in {each_situation[-1] for each_situation in situations} and len(model[-1]) == 2:
        for each_situation in situations:
            each_situation.append(-1)


def debug_series() -> Generator[Tuple[TIME, Sequence[EXAMPLE]], None, None]:
    with open("../../../configs/time_series.json", mode="r") as file:
        config = json.load(file)

    base_symbol = "ETH"                                         # type: str
    asset_symbols = ("EOS", "SNT", "QTUM", "BNT")[:1]           # type: Tuple[str, ...]

    source_paths = (config["data_dir"] + "{:s}{:s}.csv".format(_x, base_symbol) for _x in asset_symbols)

    start = datetime.datetime.fromtimestamp(1501113780, tz=tzutc())  # maybe divide by 1000?
    end = datetime.datetime.fromtimestamp(1503712000, tz=tzutc())
    # end = datetime.datetime.fromtimestamp(1529712000, tz=tzutc())
    start_str, end_str = str(start), str(end)

    generators = [series_generator(_x, interval_minutes=1, start_time=start_str, end_time=end_str) for _x in source_paths]
    last_values = [0. for _ in asset_symbols]
    while True:
        time_stamps = set()
        examples = []
        for _i, each_generator in enumerate(generators):
            try:
                t, f = next(each_generator)
            except StopIteration as e:
                raise e
            time_stamps.add(t)
            each_example = last_values[_i], f
            examples.append(each_example)
            last_values[_i] = f

        assert len(time_stamps) == 1
        t, = time_stamps

        yield t, examples


def debug_trig() -> Generator[Tuple[TIME, Sequence[EXAMPLE]], None, None]:
    for t in range(100000):
        examples = [(sin(t / 10.), cos(t / 10.))]
        yield t, examples


def simulation():
    sigma = lambda _x, _y: .02 if _x < 1 else .03                                     # type: Callable[[int], float]
    # sigma = lambda _level, _size: 1. - min(_size, 20.) / 20.                          # type: Callable[[int], float]
    # sigma = lambda _level, _size: max(1. - min(_size, 20.) / 20., 1. - min(_level, 5.) / 5.)                      # type: Callable[[int], float]
    # sigma = lambda _level, _size: float(_level < 5 and _size < 20)                      # type: Callable[[int], float]

    alpha = 100.                                                                        # type: float
    history_length = 1                                                                  # type: int
    no_senses = 1                                                                       # type: int
    sl = SimulationStats(no_senses)                                                     # type: SimulationStats

    # source = debug_series()                                                           # type: Iterable[List[EXAMPLE]]
    source = debug_trig()                                                               # type: Iterable[List[EXAMPLE]]

    model = [{0: RationalContent(0, alpha)}]                                            # type: MODEL
    states = tuple([[0 for _ in range(history_length)]] for _ in range(no_senses))      # type: Tuple[STATE, ...]
    situations = tuple([0] for _ in range(no_senses))                                   # type: Tuple[SITUATION, ...]

    for t, examples in source:
        assert len(examples) == no_senses

        # test
        output_values = []                                                              # type: List[BASIC_OUT]
        for _i, (input_value, target_value) in enumerate(examples):
            each_situation = situations[_i]
            base_content = get_content(model, each_situation, 0)                        # type: Content
            output_value = base_content.predict(input_value)                            # type: BASIC_OUT
            output_values.append(output_value)

            update_situation(each_situation, input_value, target_value, states[_i], model, sigma)

        # train
        generate_layer(model, situations)
        generate_content(model, situations, RationalContent, alpha)
        len_model = len(model)
        for each_state in states:
            len_state = len(each_state)
            if len_state == len_model - 1:
                each_state.append([0 for _ in range(history_length)])
            elif len_state == len_model:
                pass
            else:
                assert False

        adapt_content(model, states, situations)

        for _i, (input_value, target_value) in enumerate(examples):
            base_content = get_content(model, situations[_i], 0)                        # type: Content
            base_content.adapt(input_value, target_value)

        update_states(states, situations, history_length)

        sl.log(t, examples, output_values, model, situations)
        if Timer.time_passed(2000):
            print("At time stamp {:s}: {:s}".format(str(t), str(sl.model_structures[-1])))

    print(sl.model_structures[-1])
    # sl.save(model, states, "")
    sl.plot()


if __name__ == "__main__":
    simulation()
    # plot_trig()

    # predict from several inputs one target each
    # to predict one target from several inputs: multiple linear regression or symbolic "history"
