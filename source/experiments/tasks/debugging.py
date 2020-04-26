import datetime
import math
import random
import time
from typing import Sequence

from matplotlib import pyplot

from source.approximation.abstract import Approximation
from source.approximation.regression import RegressionMultiple
from source.data.abstract import STATE, EXAMPLE, GENERATOR_STATES
from source.experiments.tasks.abstract import Application, Experiment
from source.tools.functions import smear
from source.tools.moving_graph import MovingGraph


class ApplicationDebug(Application):
    def __init__(self, name: str):
        super().__init__(name)
        self.name = name
        self.iterations = 0

    def __str__(self) -> str:
        return self.name

    def _learn(self, input_value: Sequence[float], target_value: Sequence[float]):
        raise NotImplementedError()

    def learn(self, input_value: Sequence[float], target_value: Sequence[float]):
        self._learn(input_value, target_value)
        self.iterations += 1

    def _act(self, input_value: Sequence[float]) -> Sequence[float]:
        raise NotImplementedError()

    def act(self, input_value: Sequence[float]) -> Sequence[float]:
        return self._act(input_value)


class TransformRational(ApplicationDebug):
    def __init__(self, name: str, approximation: Approximation[Sequence[float], Sequence[float]]):
        super().__init__(name)
        self.approximation = approximation

    def __str__(self) -> str:
        return self.name

    def _learn(self, input_value: Sequence[float], target_value: Sequence[float]):
        self.approximation.fit(input_value, target_value, self.iterations)

    def _act(self, input_value: Sequence[float]) -> Sequence[float]:
        return self.approximation.output(input_value)


class TransformHistoric(TransformRational):
    def __init__(self, name: str, approximation: Approximation[Sequence[float], Sequence[float]], length_history: int):
        super().__init__(name, approximation)
        self.length_history = length_history
        self.history = [1. for _ in range(self.length_history)]

    def _learn(self, input_value: Sequence[float], target_value: Sequence[float]):
        self.history.extend(input_value)
        del(self.history[:-self.length_history])
        self.approximation.fit(self.history, target_value, self.iterations)

    def _act(self, input_value: Sequence[float]) -> Sequence[float]:
        history_new = self.history + list(input_value)
        del(history_new[:-self.length_history])
        return self.approximation.output(history_new)


class ExperimentStatic(Experiment):
    def _update_experiment(self, state_environment: STATE):
        self.state_experiment.update(state_environment)

    def _get_offset_example(self) -> EXAMPLE:
        return self.state_experiment["iterations"], self.state_experiment["target_last"], self.state_experiment["input_this"]

    def _initialize_state(self) -> STATE:
        pass

    def __init__(self, application: ApplicationDebug):
        super().__init__((application,))
        # self.function = lambda x: math.cos(.2 * x ** 2.)
        # self.function = lambda x: math.cos(5. * math.log(x + 1.))
        self.function = lambda x: 6.*x**0. + +4.2*x**1. + -2.7*x**2. + +.3*x**3.
        self.fig, self.subplot = pyplot.subplots(nrows=1, ncols=1)
        self.errors = [1. for _ in self.applications]
        self.max_x = 10.

        self.samples = []

    def _states(self) -> GENERATOR_STATES:
        iterations = 0
        target_last = 0.
        while True:
            input_this = random.uniform(0., self.max_x)
            target_this = self.function(input_this)
            state_environment = {"iterations": iterations, "target_last": (target_last,), "input_this": (input_this,)}
            yield state_environment
            iterations += 1
            target_last = target_this

    def _perform(self, index_application: int, action: float):
        pass

    def _post_process(self):
        for i, each_output in enumerate(self.outputs_last):
            self.errors[i] = abs(1. if each_output is None else each_output[0] - self.target_last[0])
        print(f"errors: {str(self.errors):s}")

        if self.input_last is None:
            return

        sample_new = self.input_last[0], self.target_last[0]
        self.samples.append(sample_new)
        del(self.samples[:-100])

        self.subplot.clear()

        self.subplot.scatter(*zip(*self.samples), alpha=.5)
        input_values = tuple(self.max_x * x / 100. for x in range(100))
        target_values = tuple(self.function(_x) for _x in input_values)
        output_values = tuple(self.applications[0].act([_x])[0] for _x in input_values)

        self.subplot.set_xlim(xmin=0., xmax=self.max_x)
        target_min, target_max = min(target_values), max(target_values)
        target_span = (target_max - target_min) * .1
        self.subplot.set_ylim(ymin=target_min - target_span, ymax=target_max + target_span)

        self.subplot.plot(input_values, target_values)
        self.subplot.plot(input_values, output_values, alpha=.5)

        pyplot.tight_layout()
        pyplot.pause(.05)
        time.sleep(.1)


class ExperimentTimeseries(Experiment):
    def __init__(self, application: ApplicationDebug, examples: STATE):
        super().__init__([application])
        info_subplots = tuple(
            {
                "name_axis":        f"{str(each_application):s}",
                "name_plots":       ("input", "target", "output"),  # , "moving error"),
                "moving_average":   "None",
                "limits":           None,
                "types":            "step",
                "stacked":          "False",
            }
            for each_application in self.applications)

        self.graph = MovingGraph(
            info_subplots,
            50,
            interval_draw_ms=0,
        )

        self.now = datetime.datetime.now()
        self.examples = examples
        self.errors = [-1. for _ in self.applications]
        self.iterations = 0

    def _update_experiment(self, state_environment: STATE):
        self.state_experiment.update(state_environment)

    def _get_offset_example(self) -> EXAMPLE:
        return self.state_experiment["iteration"], self.state_experiment["target_last"], self.state_experiment["input_this"]

    def _initialize_state(self) -> STATE:
        pass

    def _perform(self, index_application: int, action: float):
        pass

    def _post_process(self):
        input_last = 0. if self.input_last is None else self.input_last[0]
        for i, (each_error, each_output) in enumerate(zip(self.errors, self.outputs_last)):
            each_error = 1. if each_output is None or self.iterations < 1 else abs(each_output[0] - self.target_last[0])
            self.errors[i] = smear(each_error, each_error, self.iterations)
        self.iterations += 1

        points = tuple(
            {
                "input": input_last,
                "target": each_target,
                "output": 0. if each_output is None else each_output[0],
                "moving error": each_error,
            }
            for each_output, each_target, each_error in zip(self.outputs_last, self.target_last, self.errors)
        )

        self.graph.add_snapshot(self.now + datetime.timedelta(seconds=self.timestamp), points)
        time.sleep(.1)

    @staticmethod
    def f_square() -> STATE:
        iteration = 0
        frequency = 10
        while True:
            target_last = float((iteration - 1) % frequency >= frequency // 2),
            input_this = float(iteration % frequency < frequency // 2),
            yield {"iteration": iteration, "target_last": target_last, "input_this": input_this}
            iteration += 1

    @staticmethod
    def nf_triangle() -> STATE:
        iteration = 0
        period = 10.
        while True:
            target_last = 2. * abs(iteration / period - math.floor(iteration / period + 1. / 2.)),
            input_this = 0.,
            yield {"iteration": iteration, "target_last": target_last, "input_this": input_this}
            iteration += 1

    @staticmethod
    def nf_square() -> STATE:
        iteration = 0
        frequency = 10
        while True:
            target_last = float((iteration - 1) % frequency >= frequency // 2),
            input_this = 0.,
            yield {"iteration": iteration, "target_last": target_last, "input_this": input_this}
            iteration += 1

    @staticmethod
    def nf_trigonometry() -> STATE:
        iteration = 0
        period = math.pi
        while True:
            # target_last = float((iteration - 1) % frequency >= frequency // 2),
            target_last = (math.cos(iteration / period) + 1.) / 2.,
            input_this = (math.sin((iteration + 1) / period) + 1.) / 2.,
            yield {"iteration": iteration, "target_last": target_last, "input_this": input_this}
            iteration += 1

    @staticmethod
    def nf_erratic() -> STATE:
        iteration = 0
        period = math.pi
        f0 = lambda x: (math.cos(x / period) + 1.) / 2. - 1.
        f1 = lambda x: 2.
        while True:
            if random.random() < .1:
                f0, f1 = f1, f0
            target_last = f0(iteration),
            input_this = (math.sin((iteration + 1) / period) + 1.) / 2.,
            yield {"iteration": iteration, "target_last": target_last, "input_this": input_this}
            iteration += 1

    def _states(self) -> STATE:
        return self.examples
