import time
from typing import Sequence, List, Optional

from source.data.abstract import OUTPUT_VALUE, INPUT_VALUE, OFFSET_EXAMPLES


class Application:
    def __init__(self, name: str):
        self.name = name

    def __str__(self) -> str:
        return self.name

    def learn(self, input_value: INPUT_VALUE, target_value: OUTPUT_VALUE):
        raise NotImplementedError()

    def act(self, input_value: INPUT_VALUE) -> OUTPUT_VALUE:
        raise NotImplementedError()


class Experiment:
    def __init__(self, applications: Sequence[Application]):
        self.applications = applications
        self.timestamp = -1
        self.time_sample = -1

        self.input_value_this: Optional[INPUT_VALUE]
        self.input_value_this = None

        self.input_value_last: Optional[INPUT_VALUE]
        self.input_value_last = None

        self.target_value_last: Optional[OUTPUT_VALUE]
        self.target_value_last = None

        self.output_values_last: List[Optional[OUTPUT_VALUE]]
        self.output_values_last = [None for _ in applications]

    def _offset_examples(self) -> OFFSET_EXAMPLES:
        raise NotImplementedError()

    def _pre_process(self):
        pass

    def _perform(self, index_application: int, action: OUTPUT_VALUE):
        # changes experiment state for postprocessing (eg. plot)
        raise NotImplementedError()

    def _information_sample(self) -> str:
        return ""

    def _post_process(self):
        pass

    def start(self):
        generator_snapshots = self._offset_examples()
        for self.timestamp, self.target_value_last, self.input_value_this in generator_snapshots:
            self._pre_process()

            # initialize input, target, output
            output_values_this: List[Optional[OUTPUT_VALUE]]
            output_values_this = [None for _ in self.applications]

            for index_application, each_application in enumerate(self.applications):
                if self.input_value_last is not None:
                    each_application.learn(self.input_value_last, self.target_value_last)

                if self.input_value_this is not None:
                    output_value = each_application.act(self.input_value_this)
                    self._perform(index_application, output_value)
                    output_values_this[index_application] = output_value

            self._post_process()

            # remember input, output
            self.input_value_last = self.input_value_this
            for i, output_value in enumerate(output_values_this):
                self.output_values_last[i] = output_value

            time_now = round(time.time() * 1000)
            if self.time_sample < 0 or 1000 < time_now - self.time_sample:
                info = self._information_sample()
                if 0 < len(info):
                    print(info + "\n")
                self.time_sample = time_now

        print("done!")
