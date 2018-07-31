import datetime
import json
from typing import Union, TypeVar, List, Tuple, Iterable, Dict, Optional, Generator, Sequence

from source.data.data_generation import series_generator
from source.experiments.semiotic_modelling.content import Content, SymbolicContent

TIME = TypeVar("TIME")

BASIC_SHAPE_IN = TypeVar("BASIC_SHAPE_IN")
BASIC_SHAPE_OUT = TypeVar("BASIC_SHAPE_OUT")
EXAMPLE = Tuple[BASIC_SHAPE_IN, BASIC_SHAPE_OUT]

ABSTRACT_SHAPE = int                                        # TODO: make it generic hashable

APPEARANCE = Union[BASIC_SHAPE_IN, BASIC_SHAPE_OUT, ABSTRACT_SHAPE]
HISTORY = Union[List[APPEARANCE], Tuple[APPEARANCE, ...]]


LEVEL = Dict[APPEARANCE, Content]
MODEL = List[LEVEL]
SITUATION = List[APPEARANCE]
STATE = List[HISTORY]


class SimulationStats:
    def __init__(self, dim: int):
        self.input_values = tuple([] for _ in range(dim))
        self.target_values = tuple([] for _ in range(dim))
        self.output_values = tuple([] for _ in range(dim))           # type:

        self.contexts = tuple([] for _ in range(dim))                # type: List[List[Tuple[int, ...]]]
        self.model_structures = []                                   # type: List[Tuple[int, ...]]

        self.cumulative_errors = tuple([] for _ in range(dim))

    def log(self, examples: List[Tuple[BASIC_SHAPE_IN, BASIC_SHAPE_OUT]], output_values: List[BASIC_SHAPE_OUT], model: MODEL, states: List[STATE]):
        for _i, ((input_value, target_value), output_value, situation) in enumerate(zip(examples, output_values, states)):
            self.input_values[_i].append(input_value)
            self.target_values[_i].append(target_value)
            self.output_values[_i].append(output_value)

            context = tuple(hash(x_) for x_ in situation)       # type: Tuple[int, ...]
            self.contexts[_i].append(context)

            error = (output_value - target_value) ** 2
            cumulative_error = error + (self.cumulative_errors[_i][-1] if 0 < len(self.cumulative_errors[_i]) else 0.)
            self.cumulative_errors[_i].append(cumulative_error)

        model_structure = tuple(len(_x) for _x in model)        # type: Tuple[int, ...]
        self.model_structures.append(model_structure)

    def save(self, model: MODEL, states: List[STATE], file_path: str):
        raise NotImplementedError()

    def plot(self):
        raise NotImplementedError()


def predict(model: MODEL, situation: SITUATION, input_value: BASIC_SHAPE_IN) -> Optional[BASIC_SHAPE_OUT]:
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


def update_state(state: STATE, situation: SITUATION, history_length: int):
    len_state, len_situation = len(state), len(situation)
    assert len_state >= len_situation

    for each_shape, each_state in zip(situation, state):
        each_state.append(each_shape)
        while len(each_state) >= history_length:
            each_state.pop(0)


def adapt_content(model: MODEL, states: List[STATE], situations: List[SITUATION]):
    len_states, len_situations = len(states), len(situations)
    assert len_states == len_situations

    len_model = len(model)
    for each_state, each_situation in zip(states, situations):
        len_state, len_situation = len(each_state), len(each_situation)
        assert len_situation < len_state == len_model

        for _i in range(len_situation - 1):
            content = get_content(model, each_situation, _i + 1)
            shape_out = each_situation[_i]
            history = each_state[_i]
            content.adapt(tuple(history), shape_out)


def generate_content(model: MODEL, situations: List[SITUATION], alpha: float):
    len_model = len(model)
    for _i, each_layer in enumerate(model):
        new_shape = len(each_layer)                                             # type: ABSTRACT_SHAPE
        content_created = False                                                 # type: bool
        for each_situation in situations:
            len_situation = len(each_situation)
            assert len_model >= len_situation
            if _i >= len(each_situation):
                continue
            if each_situation[_i] == -1:
                each_situation[_i] = new_shape                                  # type: ABSTRACT_SHAPE
                if content_created:
                    continue
                each_layer[new_shape] = SymbolicContent(new_shape, alpha)       # type: Content
                content_created = True                                          # type: bool


def get_content(model: MODEL, situation: SITUATION, level: int) -> Content:
    assert level < len(model)
    layer = model[level]        # type: LEVEL

    assert level < len(situation)
    shape = situation[level]    # type: APPEARANCE

    content = layer.get(shape)  # type: Content
    assert content is not None
    return content


def update_situation(situation: SITUATION, shape: BASIC_SHAPE_IN, target_value: BASIC_SHAPE_OUT, state: STATE, model: MODEL, sigma: float):
    level = 0                                                                                                   # type: int
    content_shape = situation[level]                                                                            # type: Content
    layer = model[level]                                                                                        # type: LEVEL
    content = layer[content_shape]                                                                              # type: Content

    while content.probability(shape, target_value) < sigma and level + 1 < len(situation):
        context_shape = situation[level + 1]                                                                    # type: APPEARANCE
        upper_layer = model[level + 1]                                                                          # type: LEVEL
        context = upper_layer[context_shape]                                                                    # type: Content
        history = tuple(state[level])                                                                           # type: HISTORY

        condition = history, shape
        shape = context.predict(condition)                                                                      # type: APPEARANCE
        if shape is not None:
            content = layer[shape]                                                                              # type: Content
            if content.probability(shape, target_value) < sigma:
                content = max(layer.values(), key=lambda _x: _x.probability(shape, target_value))               # type: Content
                shape = hash(content)                                                                           # type: APPEARANCE
                if content.probability(shape, target_value) < sigma:
                    shape = -1                                                                                  # type: APPEARANCE

        level += 1                                                                                              # type: int

        situation[level] = shape                                                                                # type: APPEARANCE
        layer = upper_layer                                                                                     # type: LEVEL
        content = context                                                                                       # type: Content

    for _ in range(level, len(situation)):
        situation.pop()


def generate_layer(model: MODEL, situations: List[SITUATION]):
    len_model = len(model)
    for each_situation in situations:
        if len(each_situation) == len_model and each_situation[-1] == -1 and len(model[-1]) == 1:
            each_situation.append(-1)
            model.append(dict())
            return


def debug_series() -> Generator[Tuple[TIME, Sequence[EXAMPLE]]]:
    with open("../../../configs/time_series.json", mode="r") as file:
        config = json.load(file)

    base_symbol = "ETH"
    asset_symbols = "EOS", "SNT", "QTUM", "BNT"

    source_paths = (config["data_dir"] + "{:s}{:s}.csv".format(_x, base_symbol) for _x in asset_symbols)

    start = datetime.datetime.utcfromtimestamp(1501113780000)  # maybe divide by 1000?
    end = datetime.datetime.utcfromtimestamp(1529712000000)
    start_str, end_str = str(start), str(end)

    generators = [series_generator(_x, interval_minutes=1, start_time=start_str, end_time=end_str) for _x in source_paths]
    time_stamps = set()
    examples = []
    last_values = [0. for _ in asset_symbols]
    for _i, each_generator in enumerate(generators):
        try:
            t, f = next(each_generator)
        except StopIteration as e:
            raise e
        time_stamps.add(t)
        each_example = last_values[_i], f
        examples.append(each_example)

    assert len(time_stamps) == 1
    t, = time_stamps

    yield t, examples


def simulation():
    sigma = .1                                                                          # type: float
    alpha = 10.                                                                         # type: float
    history_length = 1                                                                  # type: int
    no_senses = 1                                                                       # type: int
    sl = SimulationStats(no_senses)                                                     # type: SimulationStats

    source = debug_series()                                                             # type: Iterable[List[EXAMPLE]]
    # source = sine_series()                                                            # type: Iterable[List[EXAMPLES]]

    model = []                                                                          # type: MODEL
    states = tuple([] for _ in range(no_senses))                                        # type: List[STATE]
    situations = tuple([] for _ in range(no_senses))                                    # type: List[SITUATION]

    for t, examples in enumerate(source):
        assert len(examples) == no_senses

        # test
        output_values = []                                                              # type: List[BASIC_SHAPE_OUT]
        for _i, (input_value, target_value) in enumerate(examples):
            base_content = get_content(model, situations[_i], 0)                        # type: Content
            output_value = base_content.predict(input_value)                            # type: BASIC_SHAPE_OUT
            output_values.append(output_value)

            update_situation(situations[_i], input_value, target_value, states[_i], model, sigma)

        # train
        generate_layer(model, situations)
        generate_content(model, situations, alpha)                                      # create new content if shape returns none
        adapt_content(model, states, situations)
        for _i, (input_value, target_value) in enumerate(examples):
            base_content = get_content(model, situations[_i], 0)                        # type: Content
            base_content.adapt(input_value, target_value)
            update_state(states[_i], situations[_i], history_length)

        sl.log(examples, output_values, model, states)

    sl.save(model, states, "")
    sl.plot()
