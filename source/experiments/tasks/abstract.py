from typing import Sequence

from source.data.abstract import STREAM_SNAPSHOTS, SNAPSHOT, EXAMPLE


class Application:
    def __str__(self) -> str:
        raise NotImplementedError()

    def _make_example(self, snapshot: SNAPSHOT) -> EXAMPLE:
        raise NotImplementedError()

    def _cycle(self, example: EXAMPLE) -> Sequence[float]:
        raise NotImplementedError()

    def cycle(self, snapshot: SNAPSHOT) -> Sequence[float]:
        example = self._make_example(snapshot)
        return self._cycle(example)


class Experiment:
    def __int__(self, applications: Sequence[Application]):
        self.applications = applications
        self.iteration = 0

    def _snapshots(self) -> STREAM_SNAPSHOTS:
        raise NotImplementedError()

    def _apply(self, snapshot: SNAPSHOT, results: Sequence[Sequence[float]]):
        pass

    def start(self):
        generator_snapshots = self._snapshots()
        for snapshot in generator_snapshots:
            results = tuple(
                each_application.cycle(snapshot)
                for each_application in self.applications
            )
            self._apply(snapshot, results)

            self.iteration += 1
