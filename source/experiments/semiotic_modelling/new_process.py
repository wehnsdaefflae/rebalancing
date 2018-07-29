from typing import Union, TypeVar, List, Tuple, Iterable, Dict, Optional, Hashable

from source.experiments.semiotic_modelling.content import Content, SymbolicContent

BASIC_SHAPE_IN = TypeVar("BASIC_SHAPE_IN")
BASIC_SHAPE_OUT = TypeVar("BASIC_SHAPE_OUT")

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
        self.model_structures = []                              # type: List[Tuple[int, ...]]

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
    if not len_model >= len_situation:
        raise ValueError("not true: len(model) = {:d} >= len(situation) = {:d}".format(len_model, len_situation))
    if len_situation < 1:
        return None
    content_shape = situation[0]
    if len_model < 1:
        return None
    base_layer = model[0]
    content = base_layer.get(content_shape)
    if content is None:
        raise ValueError("content_shape {:s} not in base_layer".format(str(content_shape)))
    return content.predict(input_value, default=None)


def update_state(state: STATE, situation: SITUATION, history_length: int):
    len_state, len_situation = len(state), len(situation)
    if not len_state >= len_situation:
        raise ValueError("not true: len(state) = {:d} >= len(situation) = {:d}".format(len_state, len_situation))

    for each_shape, each_state in zip(situation, state):
        each_state.append(each_shape)
        while len(each_state) >= history_length:
            each_state.pop(0)


def adapt_content(model: MODEL, states: List[STATE], situations: List[SITUATION]):
    len_states, len_situations = len(states), len(situations)
    if not len_states == len_situations:
        raise ValueError("not true: len(states) = {:d} == len(situations) = {:d}".format(len_states, len_situations))

    len_model = len(model)
    for each_state, each_situation in zip(states, situations):
        len_state, len_situation = len(each_state), len(each_situation)
        if not(len_situation < len_state == len_model):
            msg = "not true: len(situation) = {:d} < len(state) = {:d} == len(model) = {:d}"
            raise ValueError(msg.format(len_situation, len_state, len_model))

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
            if not len_model >= len_situation:   # TODO: thats wrong. use oversized situation for layer introduction, remove _i dependency
                raise ValueError("not true: len(model) = {:d} >= len(situation) = {:d}".format(len_model, len_situation))
            if _i >= len(each_situation):
                continue
            if each_situation[_i] == -1:
                each_situation[_i] = new_shape                                  # type: ABSTRACT_SHAPE
                if content_created:
                    continue
                each_layer[new_shape] = SymbolicContent(new_shape, alpha)       # type: Content
                content_created = True                                          # type: bool


def get_content(model: MODEL, situation: SITUATION, level: int) -> Content:
    if not level < len(model):
        raise ValueError("not true: level = {:d} < len(model) = {:d}".format(level, len(model)))
    layer = model[level]        # type: LEVEL

    if not level < len(situation):
        raise ValueError("not true: level = {:d} < len(situation) = {:d}".format(level, len(situation)))
    shape = situation[level]    # type: APPEARANCE

    content = layer.get(shape)  # type: Content
    if content is None:
        raise ValueError("no content with shape {:s} at level {:d}".format(str(shape), level))
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

    # TODO: is that okay?
    if situation[-1] == -1 and len(model[-1]) == 1:
        situation.append(-1)
        model.append(dict())
        level += 1

    for _ in range(level, len(situation)):
        situation.pop()


def simulation():
    sigma = .1                                                                          # type: float
    alpha = 10.                                                                         # type: float
    history_length = 1                                                                  # type: int
    no_senses = 3                                                                       # type: int
    sl = SimulationStats(no_senses)                                                     # type: SimulationStats

    source = []                                                                         # type: Iterable[List[Tuple[BASIC_SHAPE_IN, BASIC_SHAPE_OUT]]]

    model = []                                                                          # type: MODEL
    states = tuple([] for _ in range(no_senses))                                        # type: List[STATE]
    situations = tuple([] for _ in range(no_senses))                                    # type: List[SITUATION]

    for t, examples in enumerate(source):
        if not len(examples) == no_senses:
            raise ValueError("not true: len(examples) = {:d} == no_senses = {:d}".format(len(examples), no_senses))

        # test
        output_values = []                                                              # type: List[BASIC_SHAPE_OUT]
        for _i, (input_value, target_value) in enumerate(examples):
            base_content = get_content(model, situations[_i], 0)                        # type: Content
            output_value = base_content.predict(input_value)                            # type: BASIC_SHAPE_OUT
            output_values.append(output_value)

            update_situation(situations[_i], input_value, target_value, states[_i], model, sigma)

        # train
        generate_content(model, situations, alpha)                                      # create new content if shape returns none
        adapt_content(model, states, situations)
        for _i, (input_value, target_value) in enumerate(examples):
            base_content = get_content(model, situations[_i], 0)
            base_content.adapt(input_value, target_value)
            update_state(states[_i], situations[_i], history_length)

        sl.log(examples, output_values, model, states)

    sl.save(model, states, "")
    sl.plot()
