import json
from math import sin
from typing import Optional, Type, Callable, List

from matplotlib import pyplot

from source.data.data_generation import series_generator
from source.experiments.semiotic_modelling.content import LEVEL, Content, HISTORY, MODEL, STATE, ACTION, SHAPE_A, SymbolicContent, CONDITION, \
    RationalContent


# https://blog.yuo.be/2016/05/08/python-3-5-getting-to-grips-with-type-hints/
from source.tools.timer import Timer


def generate_model(level: int, model: MODEL, state: STATE, action: Optional[ACTION], consequence: SHAPE_A, content_class: Type[Content],
                   sigma: float = .1, alpha: float = 1., h: int = 1):
    if level < len(state):
        history = state[level]                  # type: HISTORY
        condition = tuple(history), action      # type: CONDITION

        if level + 1 < len(state):
            upper_history = state[level + 1]            # type: HISTORY
            upper_shape = upper_history[-1]             # type: SHAPE_A
            upper_layer = model[level]                  # type: LEVEL
            upper_content = upper_layer[upper_shape]    # type: Content

            if upper_content.probability(condition, consequence) < sigma:
                if level + 2 < len(state):
                    uppest_layer = model[level + 1]                                                     # type: LEVEL
                    uppest_history = state[level + 2]                                                   # type: HISTORY
                    uppest_shape = uppest_history[-1]                                                   # type: SHAPE_A
                    uppest_content = uppest_layer[uppest_shape]                                         # type: Content
                    abstract_condition = tuple(upper_history), condition                                # type: CONDITION
                    upper_shape = uppest_content.predict(abstract_condition, default=upper_shape)       # type: SHAPE_A
                    upper_content = upper_layer[upper_shape]                                            # type: Content

                    if upper_content is None or upper_content.probability(condition, consequence) < sigma:
                        upper_content = max(upper_layer.values(), key=lambda _x: _x.probability(condition, consequence))  # type:
                        # SymbolicContent
                        upper_shape = hash(upper_content)

                        if upper_content.probability(condition, consequence) < sigma:
                            upper_shape = len(upper_layer)                                # type: SHAPE_A
                            upper_content = content_class(upper_shape, alpha)             # type: Content
                            upper_layer[upper_shape] = upper_content

                else:
                    upper_shape = len(upper_layer)                                        # type: SHAPE_A
                    upper_content = content_class(upper_shape, alpha)                     # type: Content
                    upper_layer[upper_shape] = upper_content

                generate_model(level + 1, model, state, condition, upper_shape, SymbolicContent)

        else:
            upper_shape = 0                                     # type: SHAPE_A
            upper_content = content_class(upper_shape, alpha)     # type: Content
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


def debug_series():
        with open("../../../configs/time_series.json", mode="r") as file:
            config = json.load(file)

        start_time = "2017-07-27 00:02:00+00:00"
        end_time = "2018-06-23 00:00:00+00:00"
        interval_minutes = 1

        asset_symbol, base_symbol = "EOS", "ETH"

        source_path = config["data_dir"] + "{:s}{:s}.csv".format(asset_symbol, base_symbol)
        return series_generator(source_path, start_time=start_time, end_time=end_time, interval_minutes=interval_minutes)


def sine_series():
    def sine_generator():
        _i = 0
        while True:
            yield _i, sin(_i / 100)
            _i += 1
    return sine_generator()


class TimeSeriesEvaluation:
    def __init__(self, abort_at=-1):
        # self.series = debug_series()
        self.series = sine_series()

        self.error = 0.
        self.baseline_error = 0.

        self.time_axis = []
        self.value_axis = []
        self.prediction_axis = []
        self.model_development = dict()
        self.error_axis = []
        self.baseline_error_axis = []
        self.abort_at = abort_at

    @staticmethod
    def _predict(model: MODEL, state: STATE, default: float = 0.):
        if len(state) >= 2:
            context_shape = state[1][-1]
            layer = model[0]                                                        # type: LEVEL
            context = layer[context_shape]                                          # type: Content
            history = state[0]                                                      # type: HISTORY

            return context.predict((tuple(history), None), default=default)

        return default

    def _log(self, model, state, time, delta, baseline_delta):
        for _i, _l in enumerate(model):
            each_level_dev = self.model_development.get(_i)    # type: List[int]
            if each_level_dev is None:
                self.model_development[_i] = [len(model[_i])]
            else:
                each_level_dev.append(len(model[_i]))

        self.error_axis.append(delta)
        self.baseline_error_axis.append(baseline_delta)
        self.time_axis.append(time)

    def plot(self):
        fig, (ax1, ax2, ax3) = pyplot.subplots(3, sharex="all")
        # max_l = max(len(x_) for x_ in self.model_development.values())
        max_l = len(self.time_axis)
        for each_i, each_dev in self.model_development.items():
            self.model_development[each_i] = [0] * (max_l - len(each_dev)) + each_dev

        if 0 < len(self.model_development):
            ax1.stackplot(self.time_axis, *[self.model_development[_i] for _i in sorted(self.model_development.keys())])
        ax2.plot(self.time_axis, self.error_axis, label="error")
        ax2.plot(self.time_axis, self.baseline_error_axis, label="baseline error")
        ax2.legend()
        ax3.plot(self.time_axis, self.value_axis, label="time series")
        ax3.plot(self.time_axis, self.prediction_axis, label="prediction")
        ax3.legend()
        pyplot.show()

    def evaluate(self):
        model = []
        state = []

        iterations = 0

        last_element = 0.
        predicted_element = 0.

        for each_time, each_value in self.series:
            if -1 < self.abort_at <= iterations:
                break

            self.value_axis.append(each_value)
            self.prediction_axis.append(predicted_element)

            delta = abs(predicted_element - each_value)
            baseline_delta = abs(last_element - each_value)
            self.error += delta
            self.baseline_error += baseline_delta

            generate_model(0, model, state, None, each_value, RationalContent, sigma=.0, alpha=1., h=1)

            predicted_element = TimeSeriesEvaluation._predict(model, state, each_value)

            last_element = each_value

            self._log(model, state, each_time, delta, baseline_delta)
            iterations += 1
            if Timer.time_passed(2000):
                print("{:d} iterations, {:.5f} avrg error".format(iterations, self.error / iterations))

        print(iterations)
        print(self.error)
        print(self.baseline_error)
        print([len(_x) for _x in model])


def main():
    tse = TimeSeriesEvaluation(abort_at=10000)
    tse.evaluate()
    tse.plot()


if __name__ == "__main__":
    main()
