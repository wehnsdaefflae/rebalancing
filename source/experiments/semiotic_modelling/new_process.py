from typing import Hashable, Union, TypeVar, Any, List, Tuple, Iterable, Dict

from source.experiments.semiotic_modelling.content import Content

INPUT = TypeVar("INPUT")
TARGET = TypeVar("TARGET")

SHAPE = Union[INPUT, Hashable]
HISTORY = Union[List[SHAPE], Tuple[SHAPE, ...]]

MODEL = List[Dict[Hashable, Content]]
SITUATION = List[SHAPE]
STATE = List[HISTORY]


def get_situation(model: MODEL, prev_state: STATE, input_value: INPUT, target_value: TARGET) -> SITUATION:
    raise NotImplementedError()


def get_state(model: MODEL, prev_state: STATE, situation: SITUATION) -> STATE:
    raise NotImplementedError()


def adapt_model(model: MODEL, prev_state: STATE, this_state: STATE):
    raise NotImplementedError()


def adapt_state(prev_state: STATE, situation: SITUATION):
    raise NotImplementedError()


def new_process():
    source = []                                                                     # type: Iterable[Tuple[INPUT, TARGET]]

    model = []                                                                      # type: MODEL
    prev_state = []                                                                 # type: STATE

    for input_value, target_value in source:
        situation = get_situation(model, prev_state, input_value, target_value)     # type: SITUATION
        this_state = get_state(model, prev_state, situation)                        # type: STATE

        adapt_model(model, prev_state, this_state)
        adapt_state(prev_state, situation)

