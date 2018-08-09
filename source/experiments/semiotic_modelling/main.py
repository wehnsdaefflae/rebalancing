from typing import List, Tuple, Iterable, Callable, Sequence

from source.experiments.semiotic_modelling.content import Content, RationalContent
from source.experiments.semiotic_modelling.data_generators import debug_trig, ExchangeRateGeneratorFactory
from source.experiments.semiotic_modelling.evaluation import SimulationStats
from source.experiments.semiotic_modelling.modelling import EXAMPLE, get_content, update_traces, generate_content, adapt_abstract_content, \
    update_situation, generate_state_layer, MODEL, TRACE, STATE, BASIC_OUT, BASIC_IN
from source.tools.timer import Timer


sigma = lambda _level, _size: .9                                                          # type: Callable[[int, int], float]
# sigma = lambda _level, _size: .8 if _level < 1 else .1                                      # type: Callable[[int, int], float]
# sigma = lambda _level, _size: 1. - min(_size, 20.) / 20.                                    # type: Callable[[[int, int], float]
# sigma = lambda _level, _size: max(1. - min(_size, 20.) / 20., 1. - min(_level, 5.) / 5.)    # type: Callable[[[int, int], float]
# sigma = lambda _level, _size: float(_level < 5 and _size < 20)                              # type: Callable[[[int, int], float]
# sigma = lambda _level, _size: .7                                                            # type: Callable[[[int, int], float]

alpha = lambda _level, _size: 0                                                           # type: Callable[[int, int], int]


def fix_level_at_size(_level: int) -> int:
    sizes = [100, 50, 20, 10, 1, 0]
    # sizes = [1, 0]
    if _level < len(sizes):
        return sizes[_level]
    return -1


def get_outputs(inputs: Iterable[BASIC_IN], model: MODEL, states: Tuple[STATE, ...]) -> List[BASIC_OUT]:
    output_values = []                                                                          # type: List[BASIC_OUT]
    for _i, input_value in enumerate(inputs):
        each_situation = states[_i]                                                         # type: STATE
        base_content = get_content(model, each_situation, 0)                                    # type: Content
        output_value = base_content.predict(input_value)                                        # type: BASIC_OUT
        output_values.append(output_value)
    return output_values


def update_states(examples: Iterable[EXAMPLE], model: MODEL, traces: Tuple[TRACE, ...], states: Tuple[STATE, ...]):
    for _i, (input_value, target_value) in enumerate(examples):
        update_situation(input_value, target_value, model, traces[_i], states[_i], sigma, fix_level_at_size)


def adapt_base_contents(examples: Iterable[EXAMPLE], model: MODEL, states: Tuple[STATE, ...]):
    for _i, (input_value, target_value) in enumerate(examples):
        base_content = get_content(model, states[_i], 0)  # type: Content
        base_content.adapt(input_value, target_value)


def get_probabilities(examples: Iterable[EXAMPLE], model: MODEL, states: Tuple[STATE, ...]) -> Tuple[float, ...]:
    base_shapes = tuple(each_state[0] for each_state in states)
    base_layer = model[0]
    base_contents = tuple(base_layer[each_shape] for each_shape in base_shapes)
    return tuple(content.probability(*example) for (content, example) in zip(base_contents, examples))


def continuous_erratic_sequence_prediction():
    # TODO: generate examples from object, retrieve no_senses, no_dimensions, and BaseContentClass from object
    symbols = "EOS", "SNT", "QTUM", "BNT"                                                       # type: Tuple[str, ...]
    factory = ExchangeRateGeneratorFactory(symbols[:1], symbols[:1])

    no_senses = len(factory.output_definition)                                                  # type: int
    no_dimensions = len(factory.input_definition)                                               # type: int

    history_length = 1                                                                          # type: int
    sl = SimulationStats(no_senses)                                                             # type: SimulationStats

    source = factory.get_generator()                                                            # type: Iterable[List[EXAMPLE]]
    #source = debug_trig()                                                                      # type: Iterable[List[EXAMPLE]]

    class DimContent(RationalContent):
        def __init__(self, shape: int, _alpha: int):
            super().__init__(no_dimensions, shape, _alpha)

    model = [{0: DimContent(0, alpha(0, 0))}]                                                   # type: MODEL
    traces = tuple([[0 for _ in range(history_length)]] for _ in range(no_senses))              # type: Tuple[TRACE, ...]
    states = tuple([0] for _ in range(no_senses))                                               # type: Tuple[STATE, ...]

    for time_point, examples in source:
        assert len(examples) == no_senses

        # test
        input_values = [input_value for input_value, _ in examples]
        output_values = get_outputs(input_values, model, states)

        probabilities = get_probabilities(examples, model, states)                              # type: Tuple[float, ...]

        # train
        update_states(examples, model, traces, states)
        generate_state_layer(model, states)

        generate_content(model, states, DimContent, alpha)

        generate_trace_layer(history_length, model, traces)

        adapt_abstract_content(model, traces, states)
        adapt_base_contents(examples, model, states)

        update_traces(traces, states, history_length)

        sl.log(time_point, examples, output_values, probabilities, model, states)
        if Timer.time_passed(2000):
            print("At time stamp {:s}: {:s}".format(str(time_point), str(sl.model_structures[-1])))

    print(sl.model_structures[-1])
    # sl.save(model, traces, "")
    sl.plot()


def generate_trace_layer(history_length: int, model: MODEL, traces: Tuple[TRACE]):
    no_model_layers = len(model)  # type: int
    for each_trace in traces:
        no_trace_layers = len(each_trace)  # type: int
        if no_trace_layers == no_model_layers - 1:
            each_trace.append([0 for _ in range(history_length)])
        elif no_trace_layers == no_model_layers:
            pass
        else:
            assert False


def main():
    continuous_erratic_sequence_prediction()


if __name__ == "__main__":
    main()

    # predict from several inputs one target each
    # to predict one target from several inputs: multiple linear regression or symbolic "history"

