from typing import Hashable, Any, Dict, Union, List, Tuple, Generic, TypeVar, Optional, Type

SHAPE_A = TypeVar("A")
SHAPE_B = TypeVar("B")

HISTORY = List[SHAPE_A]
ACTION = Union[SHAPE_B, "CONDITION"]
CONDITION = Tuple[Tuple[SHAPE_A, ...], ACTION]
CONSEQUENCE = SHAPE_B


class Content(Hashable, Generic[SHAPE_A, SHAPE_B]):
    # TODO:  remove wildcards
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
    # pull out online regressor
    def __init__(self, shape: int, alpha: float):
        super().__init__(shape, alpha)
        self.mean_x = 0.
        self.mean_y = 0.
        self.var_x = 0.
        self.cov_xy = 0.
        self.initial = True
        self.iterations = 0

    def adapt(self, x: CONDITION, y: CONSEQUENCE):
        self._adapt(x[0][0], y)

    def _adapt(self, x: float, y: float):
        dx = x - self.mean_x
        dy = y - self.mean_y

        self.var_x = (self.alpha * self.var_x + dx ** 2) / (self.alpha + 1)
        self.cov_xy = (self.alpha * self.cov_xy + dx * dy) / (self.alpha + 1)

        if self.initial:
            self.mean_x = x
            self.mean_y = y
            self.initial = False

        else:
            self.mean_x = (self.alpha * self.mean_x + x) / (self.alpha + 1)
            self.mean_y = (self.alpha * self.mean_y + y) / (self.alpha + 1)

        self.iterations += 1

    def _get_parameters(self) -> Tuple[float, float]:
        a = 0. if self.var_x == 0. else self.cov_xy / self.var_x
        t = self.mean_y - a * self.mean_x
        return a, t

    def predict(self, condition: CONDITION, default: Optional[CONSEQUENCE] = None) -> float:
        return self._predict(condition[0][0])

    def _predict(self, x: float) -> float:
        a, t = self._get_parameters()
        return x * a + t

    def probability(self, condition: CONDITION, consequence: CONSEQUENCE, default: float = 1.) -> float:
        return self._probability(condition[0][0], consequence)

    def _probability(self, x: float, y: float) -> float:
        a, t = self._get_parameters()
        fx = x * a + t
        true_probability = 1. / (1. + abs(y - fx))
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
