import json
from typing import Optional

from source.experiments.semiotic_modelling.content import SymbolicContent, symbolic_predict, SHAPE_A, SHAPE_B, MODEL, STATE, Content
from source.experiments.semiotic_modelling.symbolic_layer import symbolic_layer
from source.tools.timer import Timer


# https://blog.yuo.be/2016/05/08/python-3-5-getting-to-grips-with-type-hints/


def base_predict(model: MODEL, state: STATE, action: Optional[SHAPE_B]) -> Optional[SHAPE_A]:
    if len(model) < 1:
        return None
    base_history = state[0]                 # type: HISTORY
    upper_history = state[1]
    content_shape = upper_history[-1]
    layer = model[0]
    content = layer.get(content_shape)                  # type: Optional[SymbolicContent]
    condition = tuple(base_history), action     # type: CONDITION
    if content is None:
        return None
    return symbolic_predict(content, condition)


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
    success = 0
    iterations = 0
    last_elem, next_elem = None, None

    for each_time, each_elem in series_generator:
        success += int(each_elem == next_elem)

        symbolic_layer(0, model, state, None, each_elem, sig=0.1, h=2)

        if len(state) >= 2:
            context_shape = state[1][-1]
            context = model[0][context_shape]                       # type: SymbolicContent
            history = state[0]
            next_elem = symbolic_predict(context, (tuple(history), None), default=each_elem)
        else:
            next_elem = None

        iterations += 1
        if Timer.time_passed(2000):
            print("{:d} iterations, {:.5f} success".format(iterations, success / iterations))

    print(iterations)
    print()
    print(len(model))
    print(len(state))
    print()
    print(success / iterations)


if __name__ == "__main__":
    main()
