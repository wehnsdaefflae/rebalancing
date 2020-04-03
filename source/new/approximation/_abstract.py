from __future__ import annotations
from typing import Generic, Dict, Any, Sequence, TypeVar

from source.new.tools.serialization import JsonSerializable

OUTPUT = TypeVar("OUTPUT", Sequence[float], float)


class Approximation(JsonSerializable, Generic[OUTPUT]):
    @staticmethod
    def from_dict(d: Dict[str, Any]) -> Approximation:
        raise NotImplementedError()

    def to_dict(self) -> Dict[str, Any]:
        raise NotImplementedError()

    def output(self, in_value: Sequence[float]) -> OUTPUT:
        raise NotImplementedError()

    def fit(self, in_value: Sequence[float], target_value: OUTPUT, drag: int):
        raise NotImplementedError()

    def __str__(self) -> str:
        return self.__class__.__name__

    def get_parameters(self) -> Sequence[float]:
        raise NotImplementedError()
