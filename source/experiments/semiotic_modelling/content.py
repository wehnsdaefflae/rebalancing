from typing import Hashable, Any, Dict, Optional, Union, List, Tuple

from source.experiments.semiotic_modelling.symbolic_layer import BASIC_SHAPE

SYMBOL = Hashable
NUMBER = float

SHAPE_A = Union[BASIC_SHAPE, int]
SHAPE_B = BASIC_SHAPE
HISTORY = List[SHAPE_A]
ACTION = Union[SHAPE_B, "CONDITION"]

CONDITION = Tuple[Tuple[SHAPE_A, ...], ACTION]
CONSEQUENCE = SHAPE_B


class Content(Hashable):
    def __init__(self, shape: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__shape = shape              # type: int

    def __repr__(self) -> str:
        return str(self.__shape)

    def __str__(self) -> str:
        return self.__repr__()

    def __hash__(self) -> int:
        return hash(self.__shape)

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, self.__class__) and self.__shape == hash(other)

    #def __lt__(self, other: Any) -> bool:
    #    return self.__shape < other.__shape


CONTENT = Union[SHAPE_A, Content]
LEVEL = Dict[SHAPE_A, CONTENT]
MODEL = List[LEVEL]
STATE = List[HISTORY]


class SymbolicContent(Content, Dict[Hashable, Dict[Hashable, int]]):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


def symbolic_probability(content: SymbolicContent, condition: CONDITION, consequence: CONSEQUENCE, default: float = 1., alp: float=1.) -> float:
    sub_dict = content.get(condition)                           # type: Dict[CONSEQUENCE, int]
    if sub_dict is None:
        return default

    total_frequency = alp                              # type: int
    for each_consequence, each_frequency in sub_dict.items():
        total_frequency += each_frequency + alp

    frequency = sub_dict.get(consequence, 0.) + alp    # type: float
    return frequency / total_frequency


def symbolic_adapt(content: SymbolicContent, condition: CONDITION, consequence: CONSEQUENCE):
    sub_dict = content.get(condition)                           # type: Dict[CONSEQUENCE, int]
    if sub_dict is None:
        sub_dict = {consequence: 1}                             # type: Dict[CONSEQUENCE, int]
        content[condition] = sub_dict
    else:
        sub_dict[consequence] = sub_dict.get(consequence, 0) + 1


def symbolic_predict(content: SymbolicContent, condition: CONDITION, default: Optional[CONSEQUENCE] = None) -> CONSEQUENCE:
    sub_dict = content.get(condition)                           # type: Dict[CONSEQUENCE, int]
    if sub_dict is None:
        return default
    consequence, _ = max(sub_dict.items(), key=lambda _x: _x[1])  # type: CONSEQUENCE, int
    return consequence


class RationalContent(Content):
    def __init__(self, shape: int, drag: int, *args, **kwargs):
        super().__init__(shape, *args, **kwargs)
        self.drag = drag
        self.mean_x = 0.
        self.mean_y = 0.
        self.var_x = 0.
        self.cov_xy = 0.
        self.initial = True
        self.iterations = 0

    def adapt(self, x: float, y: float):
        dx = x - self.mean_x
        dy = y - self.mean_y

        self.var_x = (self.drag * self.var_x + dx ** 2) / (self.drag + 1)
        self.cov_xy = (self.drag * self.cov_xy + dx * dy) / (self.drag + 1)

        if self.initial:
            self.mean_x = x
            self.mean_y = y
            self.initial = False

        else:
            self.mean_x = (self.drag * self.mean_x + x) / (self.drag + 1)
            self.mean_y = (self.drag * self.mean_y + y) / (self.drag + 1)

        self.iterations += 1

    def _get_parameters(self):
        a = 0. if self.var_x == 0. else self.cov_xy / self.var_x
        t = self.mean_y - a * self.mean_x
        return a, t

    def predict(self, condition: float) -> float:
        a, t = self._get_parameters()
        return condition * a + t

    def get_probability(self, condition: float, consequence: float) -> float:
        a, t = self._get_parameters()
        output = condition * a + t
        true_probability = 1. / (1. + abs(consequence - output))
        true_factor = self.iterations / (self.drag + self.iterations)
        return true_probability * true_factor + (1. - true_factor)


def rational_probability(content: RationalContent, condition: float, consequence: float) -> float:
    return content.get_probability(condition, consequence)


def rational_adapt(content: RationalContent, condition: float, consequence: float):
    content.adapt(condition, consequence)


def rational_predict(content: RationalContent, condition: float) -> float:
    return content.predict(condition)


# rational base layer
#   either: voronoi tesselation
#       either:   adapt current representation to each_elem
#       or:       adapt last prediction to each_elem
#   or: regression in base content
#       either:   adapt current representation to each_elem
#       or:       adapt last prediction to each_elem
# multidimensional
#   make adapt external (from old state to new state)
#   generate_model returns new state with dummies for new representations
#   final adapt generates representations
#   concurrency?
