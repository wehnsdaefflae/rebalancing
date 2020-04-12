from typing import Sequence

from source.data.abstract import STREAM_SNAPSHOTS, SNAPSHOT, TARGET_VALUE
from source.experiments.tasks.speculation import Investor

ACTION = Sequence[float]


class Application:
    def __init__(self, name: str):
        self.name = name

    def __str__(self) -> str:
        return self.name

    def learn(self, snapshot: SNAPSHOT):
        raise NotImplementedError()

    def act(self) -> ACTION:
        raise NotImplementedError()


class Experiment:
    def __init__(self, applications: Sequence[Application], delay: int):
        self.applications = applications
        self.delay = delay
        self.iterations = 0

    def _snapshots(self) -> STREAM_SNAPSHOTS:
        raise NotImplementedError()

    def _process_results(self, snapshot: SNAPSHOT, results: Sequence[RESULT]):
        pass

    def _perform(self, index_application: int, action: ACTION):
        raise NotImplementedError()

    def start(self):
        generator_snapshots = self._snapshots()
        for snapshot in generator_snapshots:
            results = tuple(
                each_application.learn(snapshot)
                for each_application in self.applications
            )

            if self.iterations >= self.delay:
                each_application: Investor
                for i, each_application in enumerate(self.applications):
                    action = each_application.act()
                    self._perform(i, action)

            self._process_results(snapshot, results)
            self.iterations += 1
