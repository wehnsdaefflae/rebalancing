from typing import Sequence

from source.data.abstract import STREAM_SNAPSHOTS, SNAPSHOT, EXAMPLE


class Application:
    def __str__(self) -> str:
        raise NotImplementedError()

    @staticmethod
    def get_timestamp(snapshot: SNAPSHOT) -> int:
        return snapshot["close_time"]

    @staticmethod
    def get_rates(snapshot: SNAPSHOT) -> Sequence[float]:
        rates = tuple(
            snapshot[x]
            for x in sorted(snapshot.keys())
            if x.startswith("rate_")
        )
        return rates

    def _make_example(self, snapshot: SNAPSHOT) -> EXAMPLE:
        raise NotImplementedError()

    def _cycle(self, example: EXAMPLE, act: bool) -> Sequence[float]:
        raise NotImplementedError()

    def cycle(self, snapshot: SNAPSHOT, act: bool = True) -> Sequence[float]:
        example = self._make_example(snapshot)
        return self._cycle(example, act)


class Experiment:
    def __int__(self, applications: Sequence[Application], delay: int):
        self.applications = applications
        self.delay = delay
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
