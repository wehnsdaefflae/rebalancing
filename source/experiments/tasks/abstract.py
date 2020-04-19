from typing import Sequence, List, Optional

from source.data.abstract import OUTPUT_VALUE, INPUT_VALUE, GENERATOR_STATES, STATE, EXAMPLE


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

        self.state_experiment = dict()

        self.input_last = None
        self.target_last = None
        self.outputs_last = None

        self.timestamp = -1
        self.input_this = None
        self.target_this = None

    def _update_experiment(self, state_environment: STATE):
        raise NotImplementedError()

    def _get_offset_example(self) -> EXAMPLE:
        raise NotImplementedError()

    def _states(self) -> GENERATOR_STATES:
        raise NotImplementedError()

    def _initialize_state(self) -> STATE:
        raise NotImplementedError()

    def _pre_process(self):
        pass

    def _perform(self, index_application: int, action: OUTPUT_VALUE):
        # changes experiment state for postprocessing (eg. plot)
        raise NotImplementedError()

    def _post_process(self):
        pass

    def start(self):
        self._initialize_state()

        generator_states = self._states()
        for state_environment in generator_states:
            self._update_experiment(state_environment)

            self.input_last = self.input_this
            self.timestamp, self.target_last, self.input_this = self._get_offset_example()

            self._pre_process()

            output_values_this: List[Optional[OUTPUT_VALUE]]
            output_values_this = [None for _ in self.applications]

            for index_application, each_application in enumerate(self.applications):
                if self.input_last is not None:
                    each_application.learn(self.input_last, self.target_last)

                output_value = each_application.act(self.input_this)
                self._perform(index_application, output_value)
                output_values_this[index_application] = output_value

            self._post_process()

            self.input_last = self.input_this
            self.outputs_last = tuple(output_values_this)

        print("done!")
