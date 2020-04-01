from typing import Sequence, Any, Tuple, Generator

from source.new.experiments.applications.applications import Application


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


