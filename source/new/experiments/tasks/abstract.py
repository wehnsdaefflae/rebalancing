from typing import Sequence, Generator, Tuple, Any


class Application:
    def __str__(self) -> str:
        raise NotImplementedError()

    def cycle(self, input_value: Sequence[float], target_value: Sequence[float]) -> Sequence[float]:
        raise NotImplementedError()


class Experiment:
    def __int__(self, applications: Sequence[Application]):
        self.applications = applications
        self.iteration = 0

    def _examples(self) -> Generator[Tuple[Any, Sequence[float], Sequence[float]], None, None]:
        raise NotImplementedError()

    def _apply(self, key: Any, input_value: Sequence[float], target_value: Sequence[float], results: Sequence[Sequence[float]]):
        pass

    def start(self):
        generator_examples = self._examples()
        for key, input_value, target_value in generator_examples:
            results = tuple(
                each_approximation.cycle(input_value, target_value)
                for each_approximation in self.applications
            )
            self._apply(key, input_value, target_value, results)

            self.iteration += 1