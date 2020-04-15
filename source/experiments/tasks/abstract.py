import time
from typing import Sequence, List, Optional

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
    def __init__(self, applications: Sequence[Application]):
        self.applications = applications
        self.iterations = 0
        self.time_sample = -1

        self.input_value_last: Optional[INPUT_VALUE]
        self.input_value_last = None

        self.target_value_last: Optional[TARGET_VALUE]
        self.target_value_last = None

        self.output_values_last: List[Optional[TARGET_VALUE]]
        self.output_values_last = [None for _ in applications]

    def _snapshots(self) -> STREAM_SNAPSHOTS:
        raise NotImplementedError()

    def _skip(self) -> bool:
        return False

    def _pre_process(self, snapshot: SNAPSHOT):
        pass

    def _get_offset_example(self, snapshot: SNAPSHOT) -> EXAMPLE:
        raise NotImplementedError()

    def _perform(self, index_application: int, action: TARGET_VALUE):
        # changes experiment state for postprocessing (eg. plot)
        raise NotImplementedError()

    def _information_sample(self) -> str:
        return ""

    def _post_process(self):
        pass

    def start(self):
        generator_snapshots = self._snapshots()
        for snapshot in generator_snapshots:
            self._pre_process(snapshot)

            # initialize input, target, output
            self.target_value_last, input_value_this = self._get_offset_example(snapshot)
            output_values_this: List[Optional[TARGET_VALUE]]
            output_values_this = [None for _ in self.applications]

            for index_application, each_application in enumerate(self.applications):
                if self.input_value_last is not None:
                    each_application.learn(self.input_value_last, self.target_value_last)

                if input_value_this is not None:
                    output_value = each_application.act(input_value_this)
                    self._perform(index_application, output_value)
                    output_values_this[index_application] = output_value

            self._post_process()

            # remember input, output
            self.input_value_last = input_value_this
            for i, output_value in enumerate(output_values_this):
                self.output_values_last[i] = output_value

            time_now = round(time.time() * 1000)
            if self.time_sample < 0 or 1000 < time_now - self.time_sample:
                print(self._information_sample() + "\n")
                self.time_sample = time_now

            self.iterations += 1

        print("done!")
