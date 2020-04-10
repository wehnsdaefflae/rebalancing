from typing import Sequence, Dict, Any

from source.data.abstract import STREAM_SNAPSHOTS, SNAPSHOT

RESULT = Dict[str, Any]


class Application:
    def __init__(self, name: str):
        self.name = name

    def __str__(self) -> str:
        return self.name

    @staticmethod
    def is_valid_snapshot(snapshot: SNAPSHOT) -> bool:
        raise NotImplementedError()

    @staticmethod
    def is_valid_result(result: RESULT) -> bool:
        raise NotImplementedError()

    def cycle(self, snapshot: SNAPSHOT, act: bool = True) -> RESULT:
        raise NotImplementedError()


class Experiment:
    def __init__(self, applications: Sequence[Application], delay: int):
        self.investors = applications
        self.delay = delay
        self.iteration = 0

    def _snapshots(self) -> STREAM_SNAPSHOTS:
        raise NotImplementedError()

    def _postprocess_results(self, snapshot: SNAPSHOT, results: Sequence[RESULT]):
        pass

    def start(self):
        generator_snapshots = self._snapshots()
        for snapshot in generator_snapshots:
            results = tuple(
                each_application.cycle(snapshot, act=self.iteration >= self.delay)
                for each_application in self.investors
            )
            self._postprocess_results(snapshot, results)

            self.iteration += 1
