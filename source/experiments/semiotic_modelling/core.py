from typing import Any, Tuple, Dict, Hashable, Optional, List, Set, Sequence

SHAPE_A = Hashable      # make it TypeVar, Hashable for making Content generic
SHAPE_B = Hashable
HISTORY = List[Optional[SHAPE_A]]
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


def certainty(content: Content, condition: CONDITION, consequence: CONSEQUENCE, default: float = 1., pseudo_count: float=1.) -> float:
    sub_dict = content.get(condition)                           # type: Dict[CONSEQUENCE, int]
    if sub_dict is None:
        return default

    total_frequency = pseudo_count                              # type: int
    for each_consequence, each_frequency in sub_dict.items():
        total_frequency += each_frequency + pseudo_count

    frequency = sub_dict.get(consequence, 0.) + pseudo_count    # type: float
    return frequency / total_frequency


def predict(content: Content, condition: CONDITION, default: Optional[CONSEQUENCE] = None) -> CONSEQUENCE:
    sub_dict = content.get(condition)                           # type: Dict[CONSEQUENCE, int]
    if sub_dict is None:
        return default
    consequence, _ = max(sub_dict.items(), key=lambda _x: _x[1])  # type: CONSEQUENCE, int
    return consequence


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


def generate_model(level: int, model, state: STATE, action: SHAPE_B, consequence: Content, SIGMA: float = .1, ALPHA: float = 1., H: int = 1):
    if level in state:
        layer = model[level]                        # type: LEVEL
        layer[consequence.shape] = consequence
        condition = tuple(state[level]), action     # type: CONDITION

        if level + 1 in state:
            upper_layer = model[level + 1]          # type: LEVEL
            upper_shape = state[level + 1][-1]      # type: SHAPE_A
            content = upper_layer[upper_shape]      # type: Content

            if certainty(content, condition, consequence, pseudo_count=ALPHA) < SIGMA:
                if level + 2 in state:
                    uppest_layer = model[level + 2]                             # type: LEVEL
                    uppest_shape = state[level + 2][-1]                         # type: SHAPE_A
                    context = uppest_layer[uppest_shape]                        # type: Content
                    abstract_condition = tuple(state[level + 1]), condition     # type: CONDITION
                    content = predict(context, abstract_condition)              # type: Content

                    if content is None or certainty(content, condition, consequence, pseudo_count=ALPHA) < SIGMA:
                        content = max(upper_layer, key=lambda x: certainty(x, condition, consequence, pseudo_count=ALPHA))      # type: Content

                        if certainty(content, condition, consequence, pseudo_count=ALPHA) < SIGMA:
                            shape = len(model[level + 1])                       # type: SHAPE_A
                            content = Content(shape)                            # type: Content

                else:
                    shape = len(model[level + 1])                               # type: SHAPE_A
                    content = Content(shape)                                    # type: Content

                generate_model(level + 1, model, state, condition, content)

        else:
            content = Content(0)                                                            # type: Content
            state[level + 1] = [None if b_i < H - 1 else content for b_i in range(H)]       # type: HISTORY
            model[level + 1] = {content.shape: content}                                     # type: LEVEL

        # TODO: externalise to enable parallelisation. change this name to "change state" and perform adaptation afterwards from copy of old state + action to new state
        adapt(content, condition, consequence)

    elif level == 0:
        state[0] = [None] * H                                                               # type: HISTORY
        model[0] = {consequence.shape: consequence}                                         # type: LEVEL  #  consequence can also be SHAPE_A!

    history = state[level]                                              # type: HISTORY
    history.append(consequence)
    history.pop(0)


def expectation(state: STATE, action: SHAPE_B) -> Optional[SHAPE_A]:
    if 0 not in state:
        return None
    if 1 not in state:
        base_history = state[0]                 # type: HISTORY
        return base_history[-1]
    base_history = state[0]                     # type: HISTORY
    condition = tuple(base_history), action     # type: CONDITION
    history = state[1]                          # type: HISTORY
    return predict(history[-1], condition)


def main():
    pass


if __name__ == "__main__":
    main()