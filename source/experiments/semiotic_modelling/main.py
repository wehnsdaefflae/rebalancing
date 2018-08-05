from typing import List, Tuple, Iterable, Callable, Sequence

from source.experiments.semiotic_modelling.content import Content, RationalContent
from source.experiments.semiotic_modelling.data_generators import debug_trig, debug_series
from source.experiments.semiotic_modelling.evaluation import SimulationStats
from source.experiments.semiotic_modelling.modelling import EXAMPLE, get_content, update_traces, generate_content, adapt_abstract_content, \
    update_situation, generate_situation_layer, MODEL, TRACE, STATE, BASIC_OUT, BASIC_IN
from source.tools.timer import Timer


# sigma = lambda _level, _size: .01 if _level < 1 else .03                                    # type: Callable[[int, int], float]
# sigma = lambda _level, _size: 1. - min(_size, 20.) / 20.                                    # type: Callable[[[int, int], float]
# sigma = lambda _level, _size: max(1. - min(_size, 20.) / 20., 1. - min(_level, 5.) / 5.)    # type: Callable[[[int, int], float]
# sigma = lambda _level, _size: float(_level < 5 and _size < 20)                              # type: Callable[[[int, int], float]
sigma = lambda _level, _size: .01                                                            # type: Callable[[[int, int], float]

# alpha = lambda _level, _size: 100. if _level < 1 else 10.                                   # type: Callable[[[int, int], float]
alpha = lambda _level, _size: 50.                                                           # type: Callable[[[int, int], float]


def fix_level_at_size(_level: int) -> int:
    return 1000
    if _level == 0:
        return 2
    if _level == 1:
        return 1
    return 0


def get_outputs(inputs: Iterable[BASIC_IN], model: MODEL, states: Tuple[STATE, ...]) -> List[BASIC_OUT]:
    output_values = []                                                                          # type: List[BASIC_OUT]
    for _i, input_value in enumerate(inputs):
        each_situation = states[_i]                                                         # type: STATE
        base_content = get_content(model, each_situation, 0)                                    # type: Content
        output_value = base_content.predict(input_value)                                        # type: BASIC_OUT
        output_values.append(output_value)
    return output_values


def update_situations(examples: Iterable[EXAMPLE], model: MODEL, traces: Tuple[TRACE, ...], states: Tuple[STATE, ...]):
    for _i, (input_value, target_value) in enumerate(examples):
        update_situation(input_value, target_value, model, traces[_i], states[_i], sigma, fix_level_at_size)


def adapt_base_contents(examples: Iterable[EXAMPLE], model: MODEL, states: Tuple[STATE, ...]):
    for _i, (input_value, target_value) in enumerate(examples):
        base_content = get_content(model, states[_i], 0)  # type: Content
        base_content.adapt(input_value, target_value)


def continuous_erratic_sequence_prediction():
    history_length = 1                                                                          # type: int
    no_senses = 1                                                                               # type: int
    sl = SimulationStats(no_senses)                                                             # type: SimulationStats

    # source = debug_series()                                                                     # type: Iterable[List[EXAMPLE]]
    source = debug_trig()                                                                       # type: Iterable[List[EXAMPLE]]

    model = [{0: RationalContent(0, alpha(0, 0))}]                                              # type: MODEL
    traces = tuple([[0 for _ in range(history_length)]] for _ in range(no_senses))              # type: Tuple[TRACE, ...]
    situations = tuple([0] for _ in range(no_senses))                                           # type: Tuple[STATE, ...]

    for t, examples in source:
        assert len(examples) == no_senses

        # test
        input_values = [input_value for input_value, _ in examples]
        output_values = get_outputs(input_values, model, situations)

        # train
        update_situations(examples, model, traces, situations)
        generate_situation_layer(model, situations)

        generate_content(model, situations, RationalContent, alpha)

        generate_trace_layer(history_length, model, traces)

        adapt_abstract_content(model, traces, situations)
        adapt_base_contents(examples, model, situations)

        update_traces(traces, situations, history_length)

        sl.log(t, examples, output_values, model, situations)
        if Timer.time_passed(2000):
            print("At time stamp {:s}: {:s}".format(str(t), str(sl.model_structures[-1])))

    print(sl.model_structures[-1])
    # sl.save(model, traces, "")
    sl.plot()


def generate_trace_layer(history_length: int, model: MODEL, states: Tuple[STATE]):
    len_model = len(model)  # type: int
    for each_state in states:
        len_state = len(each_state)  # type: int
        if len_state == len_model - 1:
            each_state.append([0 for _ in range(history_length)])
        elif len_state == len_model:
            pass
        else:
            assert False


def main():
    continuous_erratic_sequence_prediction()


if __name__ == "__main__":
    main()

    # predict from several inputs one target each
    # to predict one target from several inputs: multiple linear regression or symbolic "history"
