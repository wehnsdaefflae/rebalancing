from typing import Hashable, Any, Dict, Tuple, Generic, TypeVar, Optional

from source.tools.regression import Regressor

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
        factor = self.alpha / (self.alpha + self.iterations)
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


class RationalContent(Content[float, float]):
    def __init__(self, shape: int, alpha: int):
        super().__init__(shape, alpha)
        self.regressor = Regressor(100)

    def _adapt(self, condition: CONDITION, consequence: CONSEQUENCE):
        self.regressor.fit(condition, consequence)

    def predict(self, condition: CONDITION, default: Optional[CONSEQUENCE] = None) -> float:
        return self.regressor.output(condition)

    def _probability(self, condition: CONDITION, consequence: CONSEQUENCE, default: float = 1.) -> float:
        return self.regressor.sim(condition, consequence)


# TODO: integrate unidimensional RationalContent into this (BASIC_IN can be a Tuple)
class MLPRationalContent(Content[Tuple[float, ...], float]):
    def __init__(self, shape: int, alpha: int):
        super().__init__(shape, alpha)
        raise NotImplementedError()

    def _probability(self, condition: CONDITION, consequence: CONSEQUENCE, default: float = 1.) -> float:
        raise NotImplementedError()

    def _adapt(self, condition: CONDITION, consequence: CONSEQUENCE):
        raise NotImplementedError()

    def predict(self, condition: CONDITION, default: Optional[CONSEQUENCE] = None) -> CONSEQUENCE:
        raise NotImplementedError()

