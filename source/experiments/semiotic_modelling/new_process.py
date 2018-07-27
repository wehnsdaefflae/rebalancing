from typing import Hashable, Union, TypeVar, Any, List, Tuple, Iterable, Dict

from source.experiments.semiotic_modelling.content import Content

INPUT = TypeVar("INPUT")
TARGET = TypeVar("TARGET")

SHAPE = Union[INPUT, Hashable]
HISTORY = Union[List[SHAPE], Tuple[SHAPE, ...]]

MODEL = List[Dict[Hashable, Content]]
SITUATION = List[SHAPE]
STATE = List[HISTORY]


class SimulationLogger:
    def __init__(self, dim: int):
        self.input_values = [[] for _ in range(dim)]
        self.target_values = [[] for _ in range(dim)]
        self.output_values = [[] for _ in range(dim)]           # type:

        self.contexts = [[] for _ in range(dim)]                # type: List[List[Tuple[int, ...]]]
        self.model_structures = []                              # type: List[Tuple[int, ...]]

        self.cumulative_errors = [[] for _ in range(dim)]

    def log(self, examples: List[Tuple[INPUT, TARGET]], output_values: List[TARGET], model: MODEL, states: List[STATE]):
        for _i, ((input_value, target_value), output_value, situation) in zip(examples, output_values, states):
            self.input_values[_i].append(input_value)
            self.target_values[_i].append(target_value)
            self.output_values[_i].append(output_value)

            context = tuple(hash(x_) for x_ in situation)       # type: Tuple[int, ...]
            self.contexts[_i].append(context)

            error = (output_value - target_value) ** 2
            cumulative_error = error + (self.cumulative_errors[_i][-1] if 0 < len(self.cumulative_errors[_i]) else 0.)
            self.cumulative_errors[_i].append(cumulative_error)

        model_structure = tuple(len(_x) for _x in model)    # type: Tuple[int, ...]
        self.model_structures.append(model_structure)

    def save(self, model: MODEL, states: List[STATE], file_path: str):
        raise NotImplementedError()

    def plot(self):
        raise NotImplementedError()


def predict(model: MODEL, state: STATE, input_value: INPUT) -> TARGET:
    raise NotImplementedError()


def get_situation(model: MODEL, state: STATE, input_value: INPUT, target_value: TARGET) -> SITUATION:
    raise NotImplementedError()


def get_state(model: MODEL, state: STATE, situation: SITUATION) -> STATE:
    raise NotImplementedError()


def adapt_model(model: MODEL, state: STATE, situation: SITUATION):
    raise NotImplementedError()


def adapt_state(prev_state: STATE, situation: SITUATION):
    raise NotImplementedError()


def simulation():
    no_senses = 3
    sl = SimulationLogger(no_senses)                                                # type: SimulationLogger

    source = []                                                                     # type: Iterable[List[Tuple[INPUT, TARGET]]]

    model = []                                                                      # type: MODEL
    current_states = []                                                             # type: List[STATE]

    for t, examples in enumerate(source):
        output_values = []
        situations = []

        for _i, (input_value, target_value) in enumerate(examples):
            this_state = current_states[_i]                                             #                   # abstract input?

            output_value = predict(model, this_state, input_value)                      # type: TARGET
            output_values.append(output_value)

            # ===

            situation = get_situation(model, this_state, input_value, target_value)     # type: SITUATION   # including shapes of new contents                                                                                                             # AND base shape (abstract target?)
            situations.append(situation)

            generate_contents(model, situation)                                         # create new content if shape returns none

        adapt_model(model, current_states, situations)  # 1st: generate new contents, 2nd: adapt contents. 1st can be done in above loop

        for each_state, each_situation in zip(current_states, situations):
            adapt_state(each_state, each_situation)

        sl.log(examples, output_values, model, current_states)

    sl.save(model, current_states, "")
    sl.plot()
