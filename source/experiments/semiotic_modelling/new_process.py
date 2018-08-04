import datetime
import json
from typing import Union, TypeVar, List, Tuple, Iterable, Dict, Optional, Generator, Sequence, Type, Callable

from dateutil.tz import tzutc

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
        self.input_values = tuple([] for _ in range(dim))
        self.target_values = tuple([] for _ in range(dim))
        self.output_values = tuple([] for _ in range(dim))           # type:

        self.contexts = tuple([] for _ in range(dim))                # type: List[List[Tuple[int, ...]]]
        self.model_structures = []                                   # type: List[Tuple[int, ...]]

        self.cumulative_errors = tuple([] for _ in range(dim))

    def log(self, model: MODEL):
        model_structure = tuple(len(_x) for _x in model)        # type: Tuple[int, ...]
        self.model_structures.append(model_structure)

    def _log(self, examples: List[Tuple[BASIC_IN, BASIC_OUT]], output_values: List[BASIC_OUT], model: MODEL, states: Tuple[STATE]):
        for _i, ((input_value, target_value), output_value, situation) in enumerate(zip(examples, output_values, states)):
            self.input_values[_i].append(input_value)
            self.target_values[_i].append(target_value)
            self.output_values[_i].append(output_value)

            context = tuple(hash(x_) for x_ in situation)       # type: Tuple[int, ...]
            self.contexts[_i].append(context)

            error = (output_value - target_value) ** 2
            cumulative_error = error + (self.cumulative_errors[_i][-1] if 0 < len(self.cumulative_errors[_i]) else 0.)
            self.cumulative_errors[_i].append(cumulative_error)

        self.log(model)

    def save(self, model: MODEL, states: Tuple[STATE], file_path: str):
        pass
        # raise NotImplementedError()

    def plot(self):
        pass
        # raise NotImplementedError()


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
            # at level >= 2 add symbolic shape from level == 1 to condition
            content.adapt(tuple(history), shape_out)


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

    base_symbol = "ETH"
    asset_symbols = "EOS",  #  "SNT", "QTUM", "BNT"

    source_paths = (config["data_dir"] + "{:s}{:s}.csv".format(_x, base_symbol) for _x in asset_symbols)

    start = datetime.datetime.fromtimestamp(1501113780, tz=tzutc())  # maybe divide by 1000?
    end = datetime.datetime.fromtimestamp(1529712000, tz=tzutc())
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


def simulation():
    sigma = lambda _x, _y: .01 if _x < 1 else .5                                        # type: Callable[[int], float]
    alpha = 10.                                                                         # type: float
    history_length = 1                                                                  # type: int
    no_senses = 1                                                                       # type: int
    sl = SimulationStats(no_senses)                                                     # type: SimulationStats

    source = debug_series()                                                             # type: Iterable[List[EXAMPLE]]

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

        # sl.log(examples, output_values, model, states)
        sl.log(model)
        if Timer.time_passed(2000):
            print("At time stamp {:s}: {:s}".format(str(t), str(sl.model_structures[-1])))

    print(sl.model_structures[-1])
    # sl.save(model, states, "")
    # sl.plot()


if __name__ == "__main__":
    simulation()
    # predict from several inputs one target each
    # to predict one target from several inputs: multiple linear regression or symbolic "history"
