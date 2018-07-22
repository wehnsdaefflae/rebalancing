from typing import Hashable, Any, Dict, Union, List, Tuple, Generic, TypeVar, Optional

from source.tools.regression import Regressor

SHAPE_A = TypeVar("A")
SHAPE_B = TypeVar("B")

HISTORY = List[SHAPE_A]
ACTION = Union[SHAPE_B, "CONDITION"]
CONDITION = Tuple[Tuple[SHAPE_A, ...], ACTION]
CONSEQUENCE = SHAPE_B


class Content(Hashable, Generic[SHAPE_A, SHAPE_B]):
    def __init__(self, shape: int, alpha: float):
        super().__init__()
        self.__shape = shape              # type: int
        self.alpha = alpha

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

    def probability(self, condition: CONDITION, consequence: CONSEQUENCE, default: float = 1.) -> float:
        raise NotImplementedError

    def adapt(self, condition: CONDITION, consequence: CONSEQUENCE):
        raise NotImplementedError

    def predict(self, condition: CONDITION, default: Optional[CONSEQUENCE] = None) -> CONSEQUENCE:
        raise NotImplementedError


CONTENT = Union[SHAPE_A, Content]
LEVEL = Dict[SHAPE_A, CONTENT]
MODEL = List[LEVEL]
STATE = List[HISTORY]


class SymbolicContent(Content[Hashable, Hashable]):
    def __init__(self, shape: int, alpha: float):
        super().__init__(shape, alpha)
        self.table = dict()                         # type: Dict[Hashable, Dict[Hashable, int]]

    def probability(self, condition: CONDITION, consequence: Hashable, default: float = 1.) -> float:
        sub_dict = self.table.get(condition)                           # type: Dict[CONSEQUENCE, int]
        if sub_dict is None:
            return default

        total_frequency = self.alpha                        # type: int
        for each_consequence, each_frequency in sub_dict.items():
            total_frequency += each_frequency + self.alpha

        frequency = sub_dict.get(consequence, 0.) + self.alpha     # type: float
        return frequency / total_frequency

    def adapt(self, condition: CONDITION, consequence: Hashable):
        sub_dict = self.table.get(condition)                           # type: Dict[CONSEQUENCE, int]
        if sub_dict is None:
            sub_dict = {consequence: 1}                             # type: Dict[CONSEQUENCE, int]
            self.table[condition] = sub_dict
        else:
            sub_dict[consequence] = sub_dict.get(consequence, 0) + 1

    def predict(self, condition: CONDITION, default: Optional[Hashable] = None) -> CONSEQUENCE:
        sub_dict = self.table.get(condition)                           # type: Dict[CONSEQUENCE, int]
        if sub_dict is None:
            return default
        consequence, _ = max(sub_dict.items(), key=lambda _x: _x[1])  # type: CONSEQUENCE, int
        return consequence


class RationalContent(Content[float, float]):
    def __init__(self, shape: int, alpha: float):
        super().__init__(shape, alpha)
        self.regressor = Regressor(20)
        self.iterations = 0

    def adapt(self, condition: CONDITION, consequence: CONSEQUENCE):
        self.regressor.fit(condition[0][0], consequence)
        self.iterations += 1

    def predict(self, condition: CONDITION, default: Optional[CONSEQUENCE] = None) -> float:
        return self.regressor.output(condition[0][0])

    def probability(self, condition: CONDITION, consequence: CONSEQUENCE, default: float = 1.) -> float:
        fx = self.regressor.output(condition[0][0])
        y = consequence
        true_probability = fx / y if fx < y else y / fx
        true_factor = self.iterations / (self.alpha + self.iterations)
        return true_probability * true_factor + (1. - true_factor)


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
