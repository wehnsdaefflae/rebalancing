import json
from typing import Any, Tuple, Dict, Hashable, Optional, List, Set, Sequence

from source.tools.timer import Timer

SHAPE_A = Hashable      # make it TypeVar, Hashable for making Content generic
SHAPE_B = Hashable
HISTORY = List[Optional[SHAPE_A]]  # change model generation to avoid nones
CONDITION = Tuple[HISTORY, SHAPE_B]
CONSEQUENCE = SHAPE_B


class Content(Dict[CONDITION, Dict[CONSEQUENCE, int]]):
    def __init__(self, shape: SHAPE_A, **kwargs):
        super().__init__(**kwargs)
        self.shape = shape              # type: SHAPE_A

    def __repr__(self) -> str:
        return str(self.shape)

    def __str__(self) -> str:
        return self.__repr__()

    def __hash__(self) -> int:
        return hash(self.shape)

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, self.__class__) and self.shape == other.shape

    def __lt__(self, other: Any) -> bool:
        return self.shape < other.shape


def probability(content: Content, condition: CONDITION, consequence: CONSEQUENCE, default: float = 1., pseudo_count: float=1.) -> float:
    sub_dict = content.get(condition)                           # type: Dict[CONSEQUENCE, int]
    if sub_dict is None:
        return default

    total_frequency = pseudo_count                              # type: int
    for each_consequence, each_frequency in sub_dict.items():
        total_frequency += each_frequency + pseudo_count

    frequency = sub_dict.get(consequence, 0.) + pseudo_count    # type: float
    return frequency / total_frequency


def adapt(content: Content, condition: CONDITION, consequence: CONSEQUENCE):
    sub_dict = content.get(condition)                           # type: Dict[CONSEQUENCE, int]
    if sub_dict is None:
        sub_dict = {consequence: 1}                             # type: Dict[CONSEQUENCE, int]
        content[condition] = sub_dict
    else:
        sub_dict[consequence] = sub_dict.get(consequence, 0) + 1


STATE = List[HISTORY]           # implemented as Dict[int, HISTORY]!
LEVEL = Dict[SHAPE_A, Content]
MODEL = List[LEVEL]             # implemented as Dict[int, LEVEL]!


def _predict(content: Content, condition: CONDITION, default: Optional[CONSEQUENCE] = None) -> CONSEQUENCE:
    sub_dict = content.get(condition)                           # type: Dict[CONSEQUENCE, int]
    if sub_dict is None:
        return default
    consequence, _ = max(sub_dict.items(), key=lambda _x: _x[1])  # type: CONSEQUENCE, int
    return consequence


def predict(state: STATE, action: Optional[SHAPE_B]) -> Optional[SHAPE_A]:
    if 0 not in state:
        return None
    if 1 not in state:
        base_history = state[0]                 # type: HISTORY
        return base_history[-1]
    base_history = state[0]                     # type: HISTORY
    condition = tuple(base_history), action     # type: CONDITION
    history = state[1]                          # type: HISTORY
    return _predict(history[-1], condition)


def generate_model(level: int, model, state: STATE, action: Optional[SHAPE_B], consequence: Content, sig: float = .1, alp: float = 1., h: int = 1):
    # consequence should be shape, or consequence must be optional
    if level < len(state):
        layer = model[level]                        # type: LEVEL
        layer[consequence.shape] = consequence      # type: CONSEQUENCE
        condition = tuple(state[level]), action     # type: CONDITION

        if level + 1 < len(state):
            upper_layer = model[level + 1]          # type: LEVEL
            upper_shape = state[level + 1][-1]      # type: SHAPE_A
            content = upper_layer[upper_shape]      # type: Content

            if probability(content, condition, consequence, pseudo_count=alp) < sig:
                if level + 2 < len(state):
                    uppest_layer = model[level + 2]                             # type: LEVEL
                    uppest_shape = state[level + 2][-1]                         # type: SHAPE_A
                    context = uppest_layer[uppest_shape]                        # type: Content
                    abstract_condition = tuple(state[level + 1]), condition     # type: CONDITION
                    content = _predict(context, abstract_condition)              # type: Content

                    if content is None or probability(content, condition, consequence, pseudo_count=alp) < sig:
                        content = max(upper_layer, key=lambda x: probability(x, condition, consequence, pseudo_count=alp))      # type: Content

                        if probability(content, condition, consequence, pseudo_count=alp) < sig:
                            shape = len(model[level + 1])                       # type: SHAPE_A
                            content = Content(shape)                            # type: Content

                else:
                    shape = len(model[level + 1])                               # type: SHAPE_A
                    content = Content(shape)                                    # type: Content

                generate_model(level + 1, model, state, condition, content)

        else:
            content = Content(0)                                                            # type: Content
            state[level + 1] = [None if b_i < h - 1 else content for b_i in range(h)]       # type: HISTORY
            model[level + 1] = {content.shape: content}                                     # type: LEVEL

        # TODO: externalise to enable parallelisation. change this name to "change state" and perform adaptation afterwards from copy of old state + action to new state
        adapt(content, condition, consequence)

    elif level == 0:
        state[0] = [None] * h                                                               # type: HISTORY
        model[0] = {consequence.shape: consequence}                                         # type: LEVEL  #  consequence can also be SHAPE_A!

    history = state[level]                                              # type: HISTORY
    history.append(consequence)
    history.pop(0)



def main():
    from source.data.data_generation import series_generator

    with open("../../configs/time_series.json", mode="r") as file:
        config = json.load(file)

    # start_time = "2017-07-27 00:03:00 UTC"
    # end_time = "2018-06-22 23:52:00 UTC"
    start_time = "2017-08-01 00:00:00 UTC"
    end_time = "2017-08-02 00:00:00 UTC"
    interval_minutes = 1

    asset_symbol, base_symbol = "QTUM", "ETH"

    source_path = config["data_dir"] + "{:s}{:s}.csv".format(asset_symbol, base_symbol)
    series_generator = series_generator(source_path, start_time=start_time, end_time=end_time, interval_minutes=interval_minutes)

    model = []
    state = []
    success = 0
    iterations = 0
    next_elem = None
    for each_time, each_elem in series_generator:
        success += int(each_elem == next_elem)
        generate_model(0, model, state, None, each_elem)
        next_elem = predict(state, None)
        iterations += 1
        if Timer.time_passed(2000):
            print("{:d} iterations, {:.5f} success".format(iterations, success / iterations))




if __name__ == "__main__":
    main()
