from typing import Generic, TypeVar, Tuple

from source.experiments.semiotic_modelling.content import ContentFactory
from source.experiments.semiotic_modelling.modelling import MODEL, TRACE, STATE, generate_state_layer, generate_content, adapt_abstract_content, \
    update_traces, update_state, get_outputs, generate_trace_layer, adapt_base_contents
from source.tools.regression import MultiRegressor

INPUT_TYPE = TypeVar("INPUT_TYPE")
OUTPUT_TYPE = TypeVar("OUTPUT_TYPE")


class Predictor(Generic[INPUT_TYPE, OUTPUT_TYPE]):
    def __init__(self, no_examples: int):
        self.no_examples = no_examples

    def _fit(self, input_values: Tuple[INPUT_TYPE, ...], target_values: Tuple[OUTPUT_TYPE, ...]):
        raise NotImplementedError

    def fit(self, input_values: Tuple[INPUT_TYPE, ...], target_values: Tuple[OUTPUT_TYPE, ...]):
        assert len(input_values) == len(target_values) == self.no_examples
        self._fit(input_values, target_values)

    def save(self, file_path):
        raise NotImplementedError

    def _predict(self, input_values: Tuple[INPUT_TYPE, ...]) -> Tuple[OUTPUT_TYPE, ...]:
        raise NotImplementedError

    def predict(self, input_values: Tuple[INPUT_TYPE, ...]) -> Tuple[OUTPUT_TYPE, ...]:
        assert len(input_values) == self.no_examples
        output_values = self._predict(input_values)
        assert self.no_examples == len(output_values)
        return output_values

    def get_structure(self) -> Tuple[int, ...]:
        raise NotImplementedError


RATIONAL_VECTOR = Tuple[float, ...]
RATIONAL_SCALAR = float


class MovingAverage(Predictor[RATIONAL_VECTOR, RATIONAL_SCALAR]):
    def __init__(self, no_examples: int, drag: int):
        super().__init__(no_examples)
        self.drag = drag
        self.average = [0. for _ in range(self.no_examples)]
        self.initial = True

    def _fit(self, input_values: Tuple[RATIONAL_VECTOR, ...], target_values: Tuple[RATIONAL_SCALAR, ...]):
        if self.initial:
            for _i, each_target in enumerate(target_values):
                self.average[_i] = each_target
            self.initial = False

        else:
            for _i, each_target in enumerate(target_values):
                self.average[_i] = (self.average[_i] * self.drag + each_target) / (self.drag + 1)

    def _predict(self, input_values: Tuple[RATIONAL_VECTOR, ...]) -> Tuple[RATIONAL_SCALAR, ...]:
        return tuple(self.average)

    def save(self, file_path):
        raise NotImplementedError

    def get_structure(self) -> Tuple[int, ...]:
        raise NotImplementedError


class Regression(Predictor[RATIONAL_VECTOR, RATIONAL_SCALAR]):
    def __init__(self, no_examples: int, drag: int, input_dimension: int):
        super().__init__(no_examples)
        self.input_dimension = input_dimension
        self.drag = drag
        self.regressions = tuple(MultiRegressor(input_dimension, self.drag) for _ in range(self.no_examples))

    def _fit(self, input_values: Tuple[RATIONAL_VECTOR, ...], target_values: Tuple[RATIONAL_SCALAR, ...]):
        for each_regression, each_input, each_target in zip(self.regressions, input_values, target_values):
            each_regression.fit(each_input, each_target)

    def _predict(self, input_values: Tuple[RATIONAL_VECTOR, ...]) -> Tuple[RATIONAL_SCALAR, ...]:
        return tuple(each_regression.output(each_input) for each_regression, each_input in zip(self.regressions, input_values))

    def save(self, file_path):
        raise NotImplementedError

    def get_structure(self) -> Tuple[int, ...]:
        raise NotImplementedError


class RationalSemioticModel(Predictor[RATIONAL_VECTOR, RATIONAL_SCALAR]):
    @staticmethod
    def __sigma(level: int, size: int) -> float:
        return .9

    @staticmethod
    def __alpha(level: int, size: int) -> int:
        return 0

    @staticmethod
    def __fix_level_at_size(_level: int) -> int:
        # sizes = [100, 50, 20, 10, 1, 0]
        sizes = [10, 5, 1, 0]
        # sizes = [1, 0]
        if _level < len(sizes):
            return sizes[_level]
        return -1

    def get_certainty(self, input_values: Tuple[INPUT_TYPE, ...], target_values: Tuple[OUTPUT_TYPE, ...]) -> Tuple[float, ...]:
        base_shapes = tuple(each_state[0] for each_state in self.states)
        base_layer = self.model[0]
        base_contents = tuple(base_layer[each_shape] for each_shape in base_shapes)
        return tuple(content.probability(_input, _target) for (content, _input, _target) in zip(base_contents, input_values, target_values))

    @staticmethod
    def __update_states(input_values: Tuple[INPUT_TYPE, ...], target_values: Tuple[OUTPUT_TYPE, ...],
                        model: MODEL, traces: Tuple[STATE, ...], states: Tuple[STATE, ...]):

        for _i, (input_value, target_value) in enumerate(zip(input_values, target_values)):
            update_state(input_value, target_value,
                         model, traces[_i], states[_i],
                         RationalSemioticModel.__sigma, RationalSemioticModel.__fix_level_at_size)

    def __init__(self, no_examples: int, drag: int, input_dimensions: int, trace_length: int):
        super().__init__(no_examples)
        self.base_content_factory = ContentFactory(input_dimensions, drag, RationalSemioticModel.__alpha)
        self.trace_length = trace_length

        self.model = [{0: self.base_content_factory.rational(0, 0, 0)}]                                     # type: MODEL
        self.traces = tuple([[0 for _ in range(trace_length)]] for _ in range(no_examples))                 # type: Tuple[TRACE, ...]
        self.states = tuple([0] for _ in range(no_examples))                                                # type: Tuple[STATE, ...]

    def _fit(self, input_values: Tuple[INPUT_TYPE, ...], target_values: Tuple[OUTPUT_TYPE, ...]):
        examples = zip(input_values, target_values)

        RationalSemioticModel.__update_states(input_values, target_values, self.model, self.traces, self.states)
        generate_state_layer(self.model, self.states)

        generate_content(self.model, self.states, self.base_content_factory)

        generate_trace_layer(self.trace_length, self.model, self.traces)

        adapt_abstract_content(self.model, self.traces, self.states)
        adapt_base_contents(examples, self.model, self.states)

        update_traces(self.traces, self.states, self.trace_length)

    def _predict(self, input_values: Tuple[INPUT_TYPE, ...]) -> Tuple[OUTPUT_TYPE, ...]:
        output_list = get_outputs(input_values, self.model, self.states)
        return tuple(output_list)

    def save(self, file_path: str):
        raise NotImplementedError

    def get_structure(self) -> Tuple[int, ...]:
        return tuple(len(_x) for _x in self.model)

    def get_states(self) -> Tuple[Tuple[int, ...], ...]:
        return tuple(tuple(each_state) for each_state in self.states)
