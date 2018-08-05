from typing import List, Tuple, Iterable, Callable

from source.experiments.semiotic_modelling.content import Content, RationalContent
from source.experiments.semiotic_modelling.data_generators import debug_trig, debug_series
from source.experiments.semiotic_modelling.evaluation import SimulationStats
from source.experiments.semiotic_modelling.modelling import EXAMPLE, get_content, update_states, generate_content, adapt_content, \
    update_situation, generate_layer, MODEL, STATE, SITUATION, BASIC_OUT
from source.tools.timer import Timer


def simulation():
    sigma = lambda _x, _y: .02 if _x < 1 else .03                                     # type: Callable[[int], float]
    # sigma = lambda _level, _size: 1. - min(_size, 20.) / 20.                          # type: Callable[[int], float]
    # sigma = lambda _level, _size: max(1. - min(_size, 20.) / 20., 1. - min(_level, 5.) / 5.)                      # type: Callable[[int], float]
    # sigma = lambda _level, _size: float(_level < 5 and _size < 20)                      # type: Callable[[int], float]

    alpha = 10.                                                                        # type: float
    history_length = 1                                                                  # type: int
    no_senses = 1                                                                       # type: int
    sl = SimulationStats(no_senses)                                                     # type: SimulationStats

    source = debug_series()                                                           # type: Iterable[List[EXAMPLE]]
    # source = debug_trig()                                                               # type: Iterable[List[EXAMPLE]]

    model = [{0: RationalContent(0, alpha)}]                                            # type: MODEL
    states = tuple([[0 for _ in range(history_length)]] for _ in range(no_senses))      # type: Tuple[STATE, ...]
    situations = tuple([0] for _ in range(no_senses))                                   # type: Tuple[SITUATION, ...]

    for t, examples in source:
        assert len(examples) == no_senses

        # test
        output_values = []                                                              # type: List[BASIC_OUT]
        for _i, (input_value, target_value) in enumerate(examples):
            each_situation = situations[_i]
            base_content = get_content(model, each_situation, 0)                        # type: Content
            output_value = base_content.predict(input_value)                            # type: BASIC_OUT
            output_values.append(output_value)

            update_situation(each_situation, input_value, target_value, states[_i], model, sigma)

        # train
        generate_layer(model, situations)
        generate_content(model, situations, RationalContent, alpha)
        len_model = len(model)
        for each_state in states:
            len_state = len(each_state)
            if len_state == len_model - 1:
                each_state.append([0 for _ in range(history_length)])
            elif len_state == len_model:
                pass
            else:
                assert False

        adapt_content(model, states, situations)

        for _i, (input_value, target_value) in enumerate(examples):
            base_content = get_content(model, situations[_i], 0)                        # type: Content
            base_content.adapt(input_value, target_value)

        update_states(states, situations, history_length)

        sl.log(t, examples, output_values, model, situations)
        if Timer.time_passed(2000):
            print("At time stamp {:s}: {:s}".format(str(t), str(sl.model_structures[-1])))

    print(sl.model_structures[-1])
    # sl.save(model, states, "")
    sl.plot()


def main():
    simulation()
    date2num

if __name__ == "__main__":
    main()

    # predict from several inputs one target each
    # to predict one target from several inputs: multiple linear regression or symbolic "history"
