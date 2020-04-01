from typing import Sequence


class Application:
    def __str__(self) -> str:
        raise NotImplementedError()

    def cycle(self, input_value: Sequence[float], target_value: Sequence[float]) -> Sequence[float]:
        raise NotImplementedError()


