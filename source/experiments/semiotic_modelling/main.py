import json
from math import sin, cos
from typing import Optional, Type, Callable, List

from matplotlib import pyplot

from source.data.data_generation import series_generator
from source.experiments.semiotic_modelling.content import LEVEL, Content, HISTORY, MODEL, STATE, ACTION, SHAPE_A, SymbolicContent, CONDITION, \
    RationalContent, SHAPE_B

# https://blog.yuo.be/2016/05/08/python-3-5-getting-to-grips-with-type-hints/
from source.tools.regression import Regressor
from source.tools.timer import Timer


def generate_model(level: int, model: MODEL, state: STATE, action: Optional[ACTION], consequence: SHAPE_A, content_class: Type[Content],
                   sigma: Callable[[int], float], alpha: float = 1., h: int = 1):
    if level < len(state):
        history = state[level]                                                                                              # type: HISTORY
        condition = tuple(history), action                                                                                  # type: CONDITION

        if level + 1 < len(state):
            upper_history = state[level + 1]                                                                                # type: HISTORY
            upper_shape = upper_history[-1]                                                                                 # type: SHAPE_A
            upper_layer = model[level]                                                                                      # type: LEVEL
            upper_content = upper_layer[upper_shape]                                                                        # type: Content

            if upper_content.probability(condition, consequence) < sigma(level):
                if level + 2 < len(state):
                    uppest_layer = model[level + 1]                                                                         # type: LEVEL
                    uppest_history = state[level + 2]                                                                       # type: HISTORY
                    uppest_shape = uppest_history[-1]                                                                       # type: SHAPE_A
                    uppest_content = uppest_layer[uppest_shape]                                                             # type: Content
                    abstract_condition = tuple(upper_history), condition                                                    # type: CONDITION
                    upper_shape = uppest_content.predict(abstract_condition, default=upper_shape)                           # type: SHAPE_A
                    upper_content = upper_layer[upper_shape]                                                                # type: Content

                    if upper_content is None or upper_content.probability(condition, consequence) < sigma(level):
                        upper_content = max(upper_layer.values(), key=lambda _x: _x.probability(condition, consequence))    # type: SymbolicContent
                        upper_shape = hash(upper_content)                                                                   # type: SHAPE_A

                        if upper_content.probability(condition, consequence) < sigma(level):
                            upper_shape = len(upper_layer)                                                                  # type: SHAPE_A
                            upper_content = content_class(upper_shape, alpha)                                               # type: Content
                            upper_layer[upper_shape] = upper_content                                                        # type: SymbolicContent

                else:
                    upper_shape = len(upper_layer)                                                                          # type: SHAPE_A
                    upper_content = content_class(upper_shape, alpha)                                                       # type: Content
                    upper_layer[upper_shape] = upper_content                                                                # type: SymbolicContent

                # generate_model(level + 1, model, state, condition, upper_shape, SymbolicContent, sigma, alpha=alpha, h=h)
                generate_model(level + 1, model, state, None, upper_shape, SymbolicContent, sigma, alpha=alpha, h=h)

        else:
            upper_shape = 0                                                                                                 # type: SHAPE_A
            upper_content = content_class(upper_shape, alpha)                                                               # type: SymbolicContent
            upper_history = [upper_shape]                                                                                   # type: HISTORY
            upper_layer = {upper_shape: upper_content}                                                                      # type: LEVEL
            state.append(upper_history)
            model.append(upper_layer)

        # TODO: externalise to enable parallelisation. change this name to "change state"
        # and perform adaptation afterwards from copy of old state + action to new state
        upper_content.adapt(condition, consequence)

    elif level == 0:
        history = []                                                                                                        # type: HISTORY
        state.append(history)

    else:
        raise ValueError("Level too high.")

    history = state[level]                                                                                                  # type: HISTORY
    history.append(consequence)
    while h < len(history):
        history.pop(0)


def adapt_model_to_state(old_state: STATE, base_action: SHAPE_B, new_state: STATE, model: MODEL, content_class: Type[Content]):
    old_l = len(old_state)
    new_l = len(new_state)
    if old_l + 1 < new_l:
        raise ValueError("length of new state must be equal or 1 + length of old state")

    action = base_action
    for _i in range(len(new_state) - 1):
        new_history = new_state[_i]
        consequence = new_history[-1]

        old_history = old_state[_i]

        upper_history = new_state[_i + 1]
        shape = upper_history[-1]
        if _i >= len(model):
            content = content_class(shape)
            layer = {shape: content}
            model.append(layer)
        else:
            layer = model[_i]
            content = layer.get(shape)
            if content is None:
                content = content_class(shape)
                layer[shape] = content

        condition = tuple(old_history), action
        content.adapt(condition, consequence)
        action = condition


def debug_series():
    with open("../../../configs/time_series.json", mode="r") as file:
        config = json.load(file)

    interval_minutes = 1
    asset_symbol, base_symbol = "EOS", "ETH"

    source_path = config["data_dir"] + "{:s}{:s}.csv".format(asset_symbol, base_symbol)
    return series_generator(source_path, interval_minutes=interval_minutes)


def sine_series():
    def sine_generator():
        _i = 0
        while True:
            yield _i, sin(_i / 10) + 1.1 + cos(_i / 21) * 3 + 3.5
            _i += 1

    return sine_generator()


class TimeSeriesEvaluation:
    def __init__(self, abort_at=-1):
        # self.series = debug_series()
        self.series = sine_series()

        self.time_axis = []
        self.value_axis = []
        self.prediction_axis = []
        self.baseline_prediction_axis = []
        self.model_development = dict()
        self.error_axis = []
        self.baseline_error_axis = []
        self.abort_at = abort_at

    @staticmethod
    def _predict(model: MODEL, state: STATE, default: float = 0.):
        if len(state) >= 2:
            context_shape = state[1][-1]
            layer = model[0]  # type: LEVEL
            context = layer[context_shape]  # type: Content
            history = state[0]  # type: HISTORY

            return context.predict((tuple(history), None), default=default)

        return default

    def _log(self, model, time, target, output, baseline_output):
        for _i, _l in enumerate(model):
            each_level_dev = self.model_development.get(_i)  # type: List[int]
            if each_level_dev is None:
                self.model_development[_i] = [len(model[_i])]
            else:
                each_level_dev.append(len(model[_i]))

        self.value_axis.append(target)
        self.prediction_axis.append(output)
        self.baseline_prediction_axis.append(baseline_output)

        delta = (output - target) ** 2
        if len(self.error_axis) < 1:
            self.error_axis.append(delta)
        else:
            self.error_axis.append(self.error_axis[-1] + delta)

        baseline_delta = (baseline_output - target) ** 2
        if len(self.baseline_error_axis) < 1:
            self.baseline_error_axis.append(baseline_delta)
        else:
            self.baseline_error_axis.append(self.baseline_error_axis[-1] + baseline_delta)

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
        ax3.plot(self.time_axis, self.prediction_axis, label="prediction")
        ax3.plot(self.time_axis, self.baseline_prediction_axis, label="baseline prediction")
        ax3.plot(self.time_axis, self.value_axis, label="time series")
        ax3.legend()
        pyplot.show()

    def evaluate(self):
        model = []
        state = []

        baseline_method = Regressor(20)

        iterations = 0

        s = lambda _x: .95 if _x == 0 else .1
        last_value = 0.
        predicted_element = 0.
        baseline_prediction = 0.

        for each_time, each_value in self.series:
            if -1 < self.abort_at <= iterations:
                break

            self._log(model, each_time, each_value, predicted_element, baseline_prediction)

            # TODO: separate state from value such that input and output can be different. after pulling out state modifications?
            # TODO: similarity tolerance as a property of content, alternative: increasing tolerance with number of contents
            generate_model(0, model, state, None, each_value, RationalContent, sigma=s, alpha=100., h=1)
            predicted_element = TimeSeriesEvaluation._predict(model, state, each_value)

            if 0 < iterations:
                baseline_method.fit(last_value, each_value)
                baseline_prediction = baseline_method.output(each_value)
            else:
                baseline_prediction = each_value

            last_value = each_value
            iterations += 1
            if Timer.time_passed(2000):
                print("{:d} iterations, structure: {:s}".format(iterations, str([len(_x) for _x in model])))

        print(iterations)
        print("Accumulated error: {:.5f}".format(self.error_axis[-1]))
        print("Accumulated baseline error: {:.5f}".format(self.baseline_error_axis[-1]))
        print([len(_x) for _x in model])
        pass


def main():
    tse = TimeSeriesEvaluation(abort_at=100000)
    tse.evaluate()
    tse.plot()
    pass


if __name__ == "__main__":
    main()
