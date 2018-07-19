import json
from typing import Any, Tuple, Dict, Hashable, Optional, List, Union, Generic

from source.tools.timer import Timer


# https://blog.yuo.be/2016/05/08/python-3-5-getting-to-grips-with-type-hints/

# BASE = float
BASE = str

SHAPE_A = Union[BASE, int]
SHAPE_B = int

HISTORY = List[SHAPE_A]
CONDITION = Tuple[Tuple[SHAPE_A, ...], "ACTION"]
ACTION = Union[SHAPE_B, CONDITION]
CONSEQUENCE = SHAPE_B


class Context(Hashable, Dict[CONDITION, Dict[CONSEQUENCE, int]]):
    def __init__(self, shape: int, **kwargs):
        super().__init__(**kwargs)
        self.__shape = shape              # type: int

    def __repr__(self) -> str:
        return str(self.__shape)

    def __str__(self) -> str:
        return self.__repr__()

    def __hash__(self) -> int:
        return hash(self.__shape)

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, self.__class__) and self.__shape == hash(other)

    def __lt__(self, other: Any) -> bool:
        return self.__shape < other.__shape


CONTENT = Union[SHAPE_A, Context]
STATE = List[HISTORY]               # implemented as Dict[int, HISTORY]!
LEVEL = Dict[SHAPE_A, CONTENT]
MODEL = List[LEVEL]                 # implemented as Dict[int, LEVEL]!


def probability(content: Context, condition: CONDITION, consequence: CONSEQUENCE, default: float = 1., pseudo_count: float=1.) -> float:
    sub_dict = content.get(condition)                           # type: Dict[CONSEQUENCE, int]
    if sub_dict is None:
        return default

    total_frequency = pseudo_count                              # type: int
    for each_consequence, each_frequency in sub_dict.items():
        total_frequency += each_frequency + pseudo_count

    frequency = sub_dict.get(consequence, 0.) + pseudo_count    # type: float
    return frequency / total_frequency


def adapt(content: Context, condition: CONDITION, consequence: CONSEQUENCE):
    sub_dict = content.get(condition)                           # type: Dict[CONSEQUENCE, int]
    if sub_dict is None:
        sub_dict = {consequence: 1}                             # type: Dict[CONSEQUENCE, int]
        content[condition] = sub_dict
    else:
        sub_dict[consequence] = sub_dict.get(consequence, 0) + 1


def _predict(content: Context, condition: CONDITION, default: Optional[CONSEQUENCE] = None) -> CONSEQUENCE:
    sub_dict = content.get(condition)                           # type: Dict[CONSEQUENCE, int]
    if sub_dict is None:
        return default
    consequence, _ = max(sub_dict.items(), key=lambda _x: _x[1])  # type: CONSEQUENCE, int
    return consequence


def predict(model: MODEL, state: STATE, action: Optional[SHAPE_B]) -> Optional[SHAPE_A]:
    if len(model) < 1:
        return None
    base_history = state[0]                 # type: HISTORY
    upper_history = state[1]
    content_shape = upper_history[-1]
    layer = model[0]
    content = layer.get(content_shape)                  # type: Optional[Context]
    condition = tuple(base_history), action     # type: CONDITION
    if content is None:
        return None
    return _predict(content, condition)


def generate_model(level: int, model, state: STATE, action: Optional[ACTION], consequence: SHAPE_A, sig: float = .1, alp: float = 1., h: int = 1):
    if level < len(state):
        history = state[level]                  # type: HISTORY
        condition = tuple(history), action      # type: CONDITION

        if level + 1 < len(state):
            upper_history = state[level + 1]            # type: HISTORY
            upper_shape = upper_history[-1]             # type: SHAPE_A
            upper_layer = model[level]                  # type: LEVEL
            upper_content = upper_layer[upper_shape]    # type: Context

            if probability(upper_content, condition, consequence, pseudo_count=alp) < sig:
                if level + 2 < len(state):
                    uppest_layer = model[level + 1]                                                     # type: LEVEL
                    uppest_history = state[level + 2]                                                   # type: HISTORY
                    uppest_shape = uppest_history[-1]                                                   # type: SHAPE_A
                    uppest_content = uppest_layer[uppest_shape]                                         # type: Context
                    abstract_condition = tuple(upper_history), condition                                # type: CONDITION
                    upper_shape = _predict(uppest_content, abstract_condition, default=upper_shape)     # type: SHAPE_A
                    upper_content = upper_layer[upper_shape]                                            # type: Context

                    if upper_content is None or probability(upper_content, condition, consequence, pseudo_count=alp) < sig:
                        upper_content = max(upper_layer.values(), key=lambda _x: probability(_x, condition, consequence, pseudo_count=alp))  # type: Context
                        upper_shape = hash(upper_content)

                        if probability(upper_content, condition, consequence, pseudo_count=alp) < sig:
                            upper_shape = len(upper_layer)                                # type: SHAPE_A
                            upper_content = Context(upper_shape)                          # type: Context
                            upper_layer[upper_shape] = upper_content

                else:
                    upper_shape = len(upper_layer)                                        # type: SHAPE_A
                    upper_content = Context(upper_shape)                                  # type: Context
                    upper_layer[upper_shape] = upper_content

                generate_model(level + 1, model, state, condition, upper_shape)

        else:
            upper_shape = 0                             # type: SHAPE_A
            upper_content = Context(upper_shape)        # type: Context
            upper_history = [upper_shape]               # type: HISTORY
            state.append(upper_history)
            upper_layer = {upper_shape: upper_content}  # type: LEVEL
            model.append(upper_layer)

        # TODO: externalise to enable parallelisation. change this name to "change state"
        # and perform adaptation afterwards from copy of old state + action to new state
        adapt(upper_content, condition, consequence)

    elif level == 0:
        history = []               # type: HISTORY
        state.append(history)

    else:
        raise ValueError("Level too high.")

    history = state[level]                              # type: HISTORY
    history.append(consequence)
    while h < len(history):
        history.pop(0)


def main():
    from source.data.data_generation import series_generator

    with open("../../../configs/time_series.json", mode="r") as file:
        config = json.load(file)

    # start_time = "2017-07-27 00:03:00 UTC"
    # end_time = "2018-06-22 23:52:00 UTC"
    start_time = "2017-08-01 00:00:00 UTC"
    end_time = "2017-08-02 00:00:00 UTC"
    interval_minutes = 1

    asset_symbol, base_symbol = "QTUM", "ETH"

    source_path = config["data_dir"] + "{:s}{:s}.csv".format(asset_symbol, base_symbol)
    series_generator = series_generator(source_path, start_time=start_time, end_time=end_time, interval_minutes=interval_minutes)
    # text = "peter piper picked a peck of pickled peppers a peck of pickled peppers peter piper picked if peter piper picked a peck of pickled " \
    #        "peppers wheres the peck of pickled peppers peter piper picked "

    model = []
    state = []
    success = 0
    iterations = 0
    next_elem = None

    base_elements = set()

    for each_time, each_elem in series_generator:
        success += int(each_elem == next_elem)

        # either: voronoi tesselation
        #   either:   adapt current representation to each_elem
        #   or:       adapt last prediction to each_elem
        # or: regression in base content
        #   either:   adapt current representation to each_elem
        #   or:       adapt last prediction to each_elem
        generate_model(0, model, state, None, each_elem, sig=0., h=2)

        next_elem = predict(model, state, None)
        iterations += 1
        if Timer.time_passed(2000):
            print("{:d} iterations, {:.5f} success".format(iterations, success / iterations))

    print(iterations)
    print()
    print(len(model))
    print(len(state))
    print()
    print(success / iterations)


if __name__ == "__main__":
    main()
