import json
from typing import Optional, Type

from matplotlib import pyplot

from source.experiments.semiotic_modelling.content import LEVEL, Content, HISTORY, MODEL, STATE, ACTION, SHAPE_A, SymbolicContent, CONDITION, \
    RationalContent
from source.tools.timer import Timer


# https://blog.yuo.be/2016/05/08/python-3-5-getting-to-grips-with-type-hints/


def generate_model(level: int, model: MODEL, state: STATE, action: Optional[ACTION], consequence: SHAPE_A, content_class: Type[Content],
                   sig: float = .1, alp: float = 1., h: int = 1):
    if level < len(state):
        history = state[level]                  # type: HISTORY
        condition = tuple(history), action      # type: CONDITION

        if level + 1 < len(state):
            upper_history = state[level + 1]            # type: HISTORY
            upper_shape = upper_history[-1]             # type: SHAPE_A
            upper_layer = model[level]                  # type: LEVEL
            upper_content = upper_layer[upper_shape]    # type: Content

            if upper_content.probability(condition, consequence) < sig:
                if level + 2 < len(state):
                    uppest_layer = model[level + 1]                                                     # type: LEVEL
                    uppest_history = state[level + 2]                                                   # type: HISTORY
                    uppest_shape = uppest_history[-1]                                                   # type: SHAPE_A
                    uppest_content = uppest_layer[uppest_shape]                                         # type: Content
                    abstract_condition = tuple(upper_history), condition                                # type: CONDITION
                    upper_shape = uppest_content.predict(abstract_condition, default=upper_shape)       # type: SHAPE_A
                    upper_content = upper_layer[upper_shape]                                            # type: Content

                    if upper_content is None or upper_content.probability(condition, consequence) < sig:
                        upper_content = max(upper_layer.values(), key=lambda _x: _x.probability(condition, consequence))  # type:
                        # SymbolicContent
                        upper_shape = hash(upper_content)

                        if upper_content.probability(condition, consequence) < sig:
                            upper_shape = len(upper_layer)                                # type: SHAPE_A
                            upper_content = content_class(upper_shape, alp)             # type: Content
                            upper_layer[upper_shape] = upper_content

                else:
                    upper_shape = len(upper_layer)                                        # type: SHAPE_A
                    upper_content = content_class(upper_shape, alp)                     # type: Content
                    upper_layer[upper_shape] = upper_content

                generate_model(level + 1, model, state, condition, upper_shape, SymbolicContent)

        else:
            upper_shape = 0                                     # type: SHAPE_A
            upper_content = content_class(upper_shape, alp)     # type: Content
            upper_history = [upper_shape]                       # type: HISTORY
            state.append(upper_history)
            upper_layer = {upper_shape: upper_content}          # type: LEVEL
            model.append(upper_layer)

        # TODO: externalise to enable parallelisation. change this name to "change state"
        # and perform adaptation afterwards from copy of old state + action to new state
        upper_content.adapt(condition, consequence)

    elif level == 0:
        history = []               # type: HISTORY
        state.append(history)

    else:
        raise ValueError("Level too high.")

    history = state[level]                              # type: HISTORY
    history.append(consequence)
    while h < len(history):
        history.pop(0)


def main():
    from source.data.data_generation import series_generator

    with open("../../../configs/time_series.json", mode="r") as file:
        config = json.load(file)

    start_time = "2017-08-01 00:00:00 UTC"
    end_time = "2017-08-02 00:00:00 UTC"
    interval_minutes = 1

    asset_symbol, base_symbol = "QTUM", "ETH"

    source_path = config["data_dir"] + "{:s}{:s}.csv".format(asset_symbol, base_symbol)
    series_generator = series_generator(source_path, start_time=start_time, end_time=end_time, interval_minutes=interval_minutes)

    model = []
    state = []
    error = 0
    iterations = 0
    predictions = 0
    last_elem, next_elem = None, None

    input_series = []
    output_series = []

    for each_time, each_elem in series_generator:
        if next_elem is not None:
            output_series.append((each_time, next_elem))
            predictions += 1
            error += abs(each_elem - next_elem)

        generate_model(0, model, state, None, each_elem, RationalContent, sig=.99, h=1)

        if len(state) >= 2:
            context_shape = state[1][-1]
            layer = model[0]                                                        # type: LEVEL
            context = layer[context_shape]                                          # type: Content
            history = state[0]                                                      # type: HISTORY

            next_elem = context.predict((tuple(history), None), default=each_elem)
        else:
            next_elem = None

        if each_elem is not None:
            input_series.append((each_time, each_elem))

        iterations += 1
        if Timer.time_passed(2000):
            print("{:d} iterations, {:.5f} avrg error".format(iterations, error / iterations))

    print(iterations)
    print()
    print([len(_x) for _x in model])
    print(len(state))
    print()
    print(error / iterations)
    print(predictions)

    pyplot.plot(*zip(*input_series), label="in")
    pyplot.plot(*zip(*output_series), label="out")
    pyplot.legend()
    pyplot.show()


if __name__ == "__main__":
    main()
