from typing import Hashable, Any, Dict, Tuple, Generic, TypeVar, Optional, Callable

from source.tools.regression import Regressor, MultiRegressor

CONDITION = TypeVar("CONDITION")
CONSEQUENCE = TypeVar("CONSEQUENCE")


class Content(Hashable, Generic[CONDITION, CONSEQUENCE]):
    def __init__(self, shape: int, alpha: int):
        super().__init__()
        self.__shape = shape                # type: int
        self.alpha = alpha                  # type: int
        self.iterations = 0                 # type: int

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

    def _probability(self, condition: CONDITION, consequence: CONSEQUENCE, default: float = 1.) -> float:
        raise NotImplementedError()

    def probability(self, condition: CONDITION, consequence: CONSEQUENCE, default: float = 1.) -> float:
        factor = self.alpha / (self.alpha + self.iterations + 1.)
        p = self._probability(condition, consequence, default=default)
        return factor + (1. - factor) * p

    def adapt(self, condition: CONDITION, consequence: CONSEQUENCE):
        self._adapt(condition, consequence)
        self.iterations += 1

    def _adapt(self, condition: CONDITION, consequence: CONSEQUENCE):
        raise NotImplementedError

    def predict(self, condition: CONDITION, default: Optional[CONSEQUENCE] = None) -> CONSEQUENCE:
        raise NotImplementedError


class SymbolicContent(Content[Hashable, Hashable]):
    def __init__(self, shape: int, alpha: int):
        super().__init__(shape, alpha)
        self.table = dict()                                             # type: Dict[Hashable, Dict[Hashable, int]]

    def _probability(self, condition: CONDITION, consequence: Hashable, default: float = 1.) -> float:
        sub_dict = self.table.get(condition)                            # type: Dict[CONSEQUENCE, int]
        if sub_dict is None:
            return default
        total = sum(sub_dict.values())
        if 0 >= total:
            return default
        return sub_dict.get(consequence, 0) / total

    def _adapt(self, condition: CONDITION, consequence: Hashable):
        sub_dict = self.table.get(condition)                            # type: Dict[CONSEQUENCE, int]
        if sub_dict is None:
            sub_dict = {consequence: 1}                                 # type: Dict[CONSEQUENCE, int]
            self.table[condition] = sub_dict                            # type: Dict[CONSEQUENCE, int]
        else:
            sub_dict[consequence] = sub_dict.get(consequence, 0) + 1    # type: int

    def predict(self, condition: CONDITION, default: Optional[Hashable] = None) -> CONSEQUENCE:
        sub_dict = self.table.get(condition)                           # type: Dict[CONSEQUENCE, int]
        if sub_dict is None:
            return default
        consequence, _ = max(sub_dict.items(), key=lambda _x: _x[1])  # type: CONSEQUENCE, int
        return consequence


class RationalContent(Content[Tuple[float, ...], Tuple[float, ...]]):
    def __init__(self, input_dimension: int, output_dimension: int, shape: int, drag: int, alpha: int):
        super().__init__(shape, alpha)
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.regressions = tuple(MultiRegressor(input_dimension, drag) for _ in range(output_dimension))

    def _adapt(self, condition: CONDITION, consequence: CONSEQUENCE):
        assert len(condition) == self.input_dimension
        assert len(consequence) == self.output_dimension
        for _i, each_consequence in enumerate(consequence):
            each_regression = self.regressions[_i]
            each_regression.fit(condition, each_consequence)

    def predict(self, condition: CONDITION, default: Optional[CONSEQUENCE] = None) -> Tuple[float, ...]:
        assert len(condition) == self.input_dimension
        return tuple(each_regression.output(condition) for each_regression in self.regressions)

    def _probability(self, condition: CONDITION, consequence: CONSEQUENCE, default: float = 1.) -> float:
        assert len(condition) == self.input_dimension
        assert len(consequence) == self.output_dimension
        sim_sum = 0.
        for _i, each_consequence in enumerate(consequence):
            each_regression = self.regressions[_i]
            sim_sum += each_regression.sim(condition, each_consequence)
        return sim_sum / self.output_dimension


class ContentFactory:
    def __init__(self, input_dimension: int, output_dimensions: int, drag: int, alpha: int):
        self.input_dimension = input_dimension
        self.output_dimensions = output_dimensions
        self.drag = drag
        self.alpha = alpha

    def rational(self, shape: int):
        return RationalContent(self.input_dimension, self.output_dimensions, shape, self.drag, self.alpha)

    def symbolic(self, shape: int):
        return SymbolicContent(shape, self.alpha)
