from typing import Sequence

from source.data.abstract import EXAMPLE, STREAM_EXAMPLES


class Application:
    def __str__(self) -> str:
        raise NotImplementedError()

    def cycle(self, example: EXAMPLE) -> Sequence[float]:
        raise NotImplementedError()


class Experiment:
    def __int__(self, applications: Sequence[Application]):
        self.applications = applications
        self.iteration = 0

    def _examples(self) -> STREAM_EXAMPLES:
        raise NotImplementedError()

    def _apply(self, example: EXAMPLE, results: Sequence[Sequence[float]]):
        pass

    def start(self):
        generator_examples = self._examples()
        for example in generator_examples:
            results = tuple(
                each_application.cycle(example)
                for each_application in self.applications
            )
            self._apply(example, results)

            self.iteration += 1
