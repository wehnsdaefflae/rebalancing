from typing import Union, TypeVar, List, Tuple, Iterable, Dict, Optional, Hashable

from source.experiments.semiotic_modelling.content import Content, SymbolicContent

BASIC_SHAPE_IN = TypeVar("BASIC_SHAPE_IN")
BASIC_SHAPE_OUT = TypeVar("BASIC_SHAPE_OUT")
ABSTRACT_SHAPE = int                                        # TODO: make it generic hashable

APPEARANCE = Union[BASIC_SHAPE_IN, BASIC_SHAPE_OUT, ABSTRACT_SHAPE]
HISTORY = Union[List[APPEARANCE], Tuple[APPEARANCE, ...]]

MODEL = List[Dict[APPEARANCE, Content]]
SITUATION = List[APPEARANCE]
STATE = List[HISTORY]


class SimulationStats:
    def __init__(self, dim: int):
        self.input_values = [[] for _ in range(dim)]
        self.target_values = [[] for _ in range(dim)]
        self.output_values = [[] for _ in range(dim)]           # type:

        self.contexts = [[] for _ in range(dim)]                # type: List[List[Tuple[int, ...]]]
        self.model_structures = []                              # type: List[Tuple[int, ...]]

        self.cumulative_errors = [[] for _ in range(dim)]

    def log(self, examples: List[Tuple[BASIC_SHAPE_IN, BASIC_SHAPE_OUT]], output_values: List[BASIC_SHAPE_OUT], model: MODEL, states: List[STATE]):
        for _i, ((input_value, target_value), output_value, situation) in zip(examples, output_values, states):
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


def _update_state(state: STATE, situation: SITUATION, history_length: int):
    len_state, len_situation = len(state), len(situation)
    if not len_situation < len_state:
        raise ValueError("not true: len(situation) = {:d} < len(state) = {:d}".format(len_situation, len_state))

    for _i, each_shape in enumerate(situation):
        each_state = state[_i]
        each_state.append(each_shape)
        while len(each_state) >= history_length:
            each_state.pop(0)


def update_states(states: List[STATE], situations: List[SITUATION], history_length: int):
    len_states, len_situations = len(states), len(situations)
    if not len_states == len_situations:
        raise ValueError("not true: len(states) = {:d} == len(situations) = {:d}".format(len_states, len_situations))

    for each_state, each_situation in zip(states, situations):
        _update_state(each_state, each_situation, history_length)


def adapt_contents(model: MODEL, states: List[STATE], situations: List[SITUATION]):
    len_states, len_situations = len(states), len(situations)
    if not len_states == len_situations:
        raise ValueError("not true: len(states) = {:d} == len(situations) = {:d}".format(len_states, len_situations))

    len_model = len(model)
    for each_state, each_situation in zip(states, situations):
        len_state, len_situation = len(each_state), len(each_situation)
        if not(len_situation < len_state == len_model + 1):
            msg = "not true: len(situation) = {:d} < len(state) = {:d} == len(model) + 1 = {:d}"
            raise ValueError(msg.format(len_situation, len_state, len_model + 1))

        for _i in range(len_situation - 1):
            history = each_state[_i]
            layer = model[_i]
            content_shape = each_situation[_i + 1]
            content = layer.get(content_shape)
            if content is None:
                raise ValueError("content_shape {:s} not in model layer {:d}".format(str(content_shape), _i))
            shape_out = each_situation[_i]
            content.adapt(tuple(history), shape_out)


def generate_contents(model: MODEL, situations: List[SITUATION], alpha: float):
    len_model = len(model)
    for _i, each_layer in enumerate(model):
        new_shape = len(each_layer)                                             # type: ABSTRACT_SHAPE
        content_created = False                                                 # type: bool
        for each_situation in situations:
            len_situation = len(each_situation)
            if not len_model >= len_situation:
                raise ValueError("not true: len(model) = {:d} >= len(situation) = {:d}".format(len_model, len_situation))
            if _i >= len(each_situation):
                continue
            if each_situation[_i] == -1:
                each_situation[_i] = new_shape                                  # type: ABSTRACT_SHAPE
                if content_created:
                    continue
                each_layer[new_shape] = SymbolicContent(new_shape, alpha)       # type: Content
                content_created = True                                          # type: bool


def _generate_model(level: int, model: MODEL, state: STATE, situation: SITUATION) -> APPEARANCE:
    raise NotImplementedError()


def update_situation(model: MODEL, state: STATE, situation: SITUATION, input_value: BASIC_SHAPE_IN, target_value: BASIC_SHAPE_OUT,
                     sigma: float) -> SITUATION:
    # TODO: change len_situation checks to _not_ include base level, model does _not_, only state _does_

    level = 0
    content_shape = situation[level]  # type: Content
    layer = model[level]
    content = layer[content_shape]
    while content.probability(input_value, target_value) < sigma and level + 1 < len(model):
        if level < len(situation):
            context_shape = situation[level + 1]
            upper_layer = model[level + 1]
            context = upper_layer[context_shape]  # type: Content
            history = tuple(state[level])
            condition = history, input_value
            content_shape = context.predict(condition)
            if content_shape is not None:
                content = layer[content_shape]
                if content.probability(input_value, target_value) < sigma:
                    content = max(layer.values(), key=lambda _x: _x.probability(input_value, target_value))
                    content_shape = hash(content)
                    if content.probability(input_value, target_value) < sigma:
                        content_shape = -1



    # add level as parameter and return new shape for this level. don't return anything if no change
    # return [input_value] + new abstract shapes
    # return missing content shapes as -1
    # this is where core functionality is
    raise NotImplementedError()


def simulation():
    sigma = .1
    alpha = 10.
    history_length = 1                                                                   # type: int
    no_senses = 3                                                                        # type: int
    sl = SimulationStats(no_senses)                                                      # type: SimulationStats

    source = []                                                                          # type: Iterable[List[Tuple[BASIC_SHAPE_IN, BASIC_SHAPE_OUT]]]

    model = []                                                                           # type: MODEL
    current_states = [[] for _ in range(no_senses)]                                      # type: List[STATE]
    situations = [[] for _ in range(no_senses)]                                          # type: List[SITUATION]

    for t, examples in enumerate(source):
        output_values = []                                                               # type: List[BASIC_SHAPE_OUT]

        for _i, (input_value, target_value) in enumerate(examples):
            this_state = current_states[_i]                                              # type: STATE
            output_value = predict(model, situations[_i], input_value)                   # type: BASIC_SHAPE_OUT
            output_values.append(output_value)

            update_situation(model, this_state, situations[_i], input_value, target_value, sigma)
            # including shapes of new contents as -1 but NO base shape (abstract target?)

        generate_contents(model, situations, alpha)                                      # create new content if shape returns none
        adapt_contents(model, current_states, situations)                                # adapt contents
        update_states(current_states, situations, history_length)

        sl.log(examples, output_values, model, current_states)

    sl.save(model, current_states, "")
    sl.plot()

