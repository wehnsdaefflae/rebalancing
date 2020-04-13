from typing import Sequence

from source.data.abstract import STREAM_SNAPSHOTS, SNAPSHOT, TARGET_VALUE, INPUT_VALUE, EXAMPLE


class Application:
    def __init__(self, name: str):
        self.name = name

    def __str__(self) -> str:
        return self.name

    def learn(self, input_value: INPUT_VALUE, target_value: TARGET_VALUE):
        raise NotImplementedError()

    def act(self, input_value: INPUT_VALUE) -> TARGET_VALUE:
        raise NotImplementedError()


class Experiment:
    def __init__(self, applications: Sequence[Application], delay: int):
        self.applications = applications
        self.delay = delay
        self.iteration = 0

    def _snapshots(self) -> STREAM_SNAPSHOTS:
        raise NotImplementedError()

    def _pre_process(self, snapshot: SNAPSHOT):
        pass

    def _get_example(self, snapshot: SNAPSHOT) -> EXAMPLE:
        raise NotImplementedError()

    def _perform(self, index_application: int, action: TARGET_VALUE):
        raise NotImplementedError()

    def _post_process(self, snapshot: SNAPSHOT):
        pass

    def start(self):
        input_value_last = None

        generator_snapshots = self._snapshots()
        for snapshot in generator_snapshots:
            self._pre_process(snapshot)

            input_value, target_value = self._get_example(snapshot)

            if self.iteration >= self.delay:
                for index_application, each_application in enumerate(self.applications):
                    if 0 < self.iteration and not (input_value_last is None or target_value is None):
                        each_application.learn(input_value_last, target_value)

                    if input_value is not None:
                        action = each_application.act(input_value)
                        self._perform(index_application, action)

            input_value_last = input_value

            self._post_process(snapshot)
            self.iteration += 1
