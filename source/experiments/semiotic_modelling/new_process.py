from typing import Union, TypeVar, List, Tuple, Iterable, Dict, Optional

from source.experiments.semiotic_modelling.content import Content, SymbolicContent

ABSTRACT_SHAPE = int                                        # or Hashable
BASIC_SHAPE_IN = TypeVar("BASIC_SHAPE_IN")
BASIC_SHAPE_OUT = TypeVar("BASIC_SHAPE_OUT")
EXAMPLE = Tuple[BASIC_SHAPE_IN, BASIC_SHAPE_OUT]

APPEARANCE = Union[BASIC_SHAPE_IN, ABSTRACT_SHAPE]
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

    def log(self, examples: List[EXAMPLE], output_values: List[BASIC_SHAPE_OUT], model: MODEL, states: List[STATE]):
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
    if len_model != len_situation - 1:
        raise ValueError("not identical: len(model) + 1 = {:d}, len(situation) = {:d}".format(len_model + 1, len_situation))
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
    if len_state != len_situation:
        raise ValueError("not identical: len(state) = {:d}, len(situation) = {:d}".format(len_state, len_situation))

    for each_state, each_shape in zip(state, situation):
        each_state.append(each_shape)
        while len(each_state) >= history_length:
            each_state.pop(0)


def adapt_contents(model: MODEL, state: STATE, situation: SITUATION):
    len_model, len_state, len_situation = len(model), len(state), len(situation)
    if not(len_model + 1 == len_state == len_situation):
        msg = "not identical: len(state) = {:d}, len(situation) = {:d}, len(model) + 1 = {:d}"
        raise ValueError(msg.format(len_state, len_situation, len_model + 1))

    for _i in range(len_situation - 1):
        history = state[_i]
        layer = model[_i]
        content_shape = situation[_i + 1]
        content = layer.get(content_shape)
        if content is None:
            raise ValueError("content_shape {:s} not in model layer {:d}".format(str(content_shape), _i))
        shape_out = situation[_i]
        content.adapt(tuple(history), shape_out)


def generate_contents(model: MODEL, situations: List[SITUATION], alpha: float):
    for _i, each_layer in enumerate(model):
        new_shape = len(each_layer)                                             # type: ABSTRACT_SHAPE
        content_created = False                                                 # type: bool
        for each_situation in situations:
            if each_situation[_i] == -1:
                each_situation[_i] = new_shape                                  # type: ABSTRACT_SHAPE
                if content_created:
                    continue
                each_layer[new_shape] = SymbolicContent(new_shape, alpha)       # type: Content
                content_created = True                                          # type: bool


def get_new_situation(model: MODEL, state: STATE, input_value: BASIC_SHAPE_IN, target_value: BASIC_SHAPE_OUT) -> SITUATION:
    # add level as parameter and return new shape for this level. don't return anything if no change
    # return [input_value] + new abstract shapes
    # return missing content shapes as -1
    # this is where core functionality is
    raise NotImplementedError()


def simulation():
    alpha = 10.
    history_length = 1                                                                   # type: int
    no_senses = 3                                                                        # type: int
    sl = SimulationStats(no_senses)                                                      # type: SimulationStats

    source = []                                                                          # type: Iterable[List[Tuple[BASIC_SHAPE_IN, BASIC_SHAPE_OUT]]]

    model = []                                                                           # type: MODEL
    current_states = []                                                                  # type: List[STATE]

    for t, examples in enumerate(source):
        output_values = []                                                               # type: List[BASIC_SHAPE_OUT]
        situations = []                                                                  # type: List[SITUATION]

        for _i, (input_value, target_value) in enumerate(examples):
            this_state = current_states[_i]                                              # type: STATE

            output_value = predict(model, situations[_i], input_value)                   # type: BASIC_SHAPE_OUT
            output_values.append(output_value)

            # ===

            situation = get_new_situation(model, this_state, input_value, target_value)  # type: SITUATION   # including shapes of new contentsas -1
                                                                                                            #  AND base shape (abstract target?)
            situations.append(situation)

        generate_contents(model, situations, alpha)                                      # create new content if shape returns none
        adapt_contents(model, current_states, situations)                                # adapt contents

        for each_state, each_situation in zip(current_states, situations):
            update_state(each_state, each_situation, history_length)

        sl.log(examples, output_values, model, current_states)

    sl.save(model, current_states, "")
    sl.plot()

