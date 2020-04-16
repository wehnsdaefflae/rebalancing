from __future__ import annotations
from typing import Generic, Dict, Any, Sequence, TypeVar

from source.tools.serialization import JsonSerializable

OUTPUT_VALUE = TypeVar("OUTPUT_VALUE", Sequence[float], float, int)
INPUT_VALUE = TypeVar("INPUT_VALUE", Sequence[float], float, int)


class Approximation(JsonSerializable, Generic[INPUT_VALUE, OUTPUT_VALUE]):
    @staticmethod
    def from_dict(d: Dict[str, Any]) -> Approximation:
        raise NotImplementedError()

    def to_dict(self) -> Dict[str, Any]:
        raise NotImplementedError()

    def output(self, in_value: INPUT_VALUE) -> OUTPUT_VALUE:
        raise NotImplementedError()

    def fit(self, in_value: INPUT_VALUE, target_value: OUTPUT_VALUE, drag: int):
        raise NotImplementedError()

    def __str__(self) -> str:
        return self.__class__.__name__

    def get_parameters(self) -> Sequence[float]:
        raise NotImplementedError()

    def get_state(self) -> Any:
        return None
