import datetime
import math
import random
import time
from typing import Sequence

from matplotlib import pyplot

from source.approximation.abstract import Approximation
from source.data.abstract import INPUT_VALUE, OUTPUT_VALUE, OFFSET_EXAMPLES
from source.experiments.tasks.abstract import Application, Experiment
from source.tools.functions import smear
from source.tools.moving_graph import MovingGraph


class TransformRational(Application):
    def __init__(self, name: str, approximation: Approximation[Sequence[float], Sequence[float]]):
        super().__init__(name)
        self.name = name
        self.approximation = approximation
        self.iterations = 0

    def __str__(self) -> str:
        return self.name

    def learn(self, input_value: INPUT_VALUE, target_value: OUTPUT_VALUE):
        self.approximation.fit(input_value, target_value, self.iterations)
        self.iterations += 1

    def act(self, input_value: INPUT_VALUE) -> OUTPUT_VALUE:
        return self.approximation.output(input_value)


class ExperimentStatic(Experiment):
    def __init__(self, application: Application):
        super().__init__((application,))
        # self.function = lambda x: math.cos(.2 * x ** 2.)
        self.function = lambda x: math.cos(10. * math.log(x + 1.))
        self.fig, self.subplot = pyplot.subplots(nrows=1, ncols=1)
        self.errors = [1. for _ in self.applications]
        self.max_x = 10.

        self.samples = []

    def _states(self) -> OFFSET_EXAMPLES:
        iterations = 0
        target_last = 0.
        while True:
            input_this = random.uniform(0., self.max_x)
            target_this = self.function(input_this)
            yield iterations, (target_last,), (input_this,)
            iterations += 1
            target_last = target_this

    def _perform(self, index_application: int, action: OUTPUT_VALUE):
        pass

    def _post_process(self):
        if None not in self.output_values_last:
            for i, each_output in enumerate(self.output_values_last):
                self.errors[i] = abs(each_output[0] - self.target_value_last[0])

        if self.input_value_last is None:
            return

        sample_new = self.input_value_last[0], self.target_value_last[0]
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
    def __init__(self, applications: Sequence[Application], examples: OFFSET_EXAMPLES):
        super().__init__(applications)
        info_subplots = tuple(
            (
                f"{str(each_application):s}",
                ("input", "target", "output", "average error"),
                "None",
                None,
                "regular"
            ) for each_application in applications)

        self.graph = MovingGraph(
            info_subplots,
            50,
            interval_draw_ms=0,
        )

        self.now = datetime.datetime.now()
        self.examples = examples
        self.errors = [-1. for _ in self.applications]
        self.iterations = 0

    def _perform(self, index_application: int, action: OUTPUT_VALUE):
        pass

    def _post_process(self):
        print(f"{self.applications[0].approximation.index_classifier_current:03d}: {str(self.applications[0].approximation.get_structure()):s}")

        input_last = 0. if self.input_value_last is None else self.input_value_last[0]
        for i, (output_value_last, each_error) in enumerate(zip(self.output_values_last, self.errors)):
            self.errors[i] = smear(each_error, 1. if output_value_last is None else abs(output_value_last[0] - self.target_value_last[0]), self.iterations)
        self.iterations += 1

        points = tuple(
            {
                "input": input_last,
                "target": self.target_value_last[0],
                "output": 0. if output_value_last is None else output_value_last[0],
                "average error": each_error,
            }
            for output_value_last, each_error in zip(self.output_values_last, self.errors)
        )

        self.graph.add_snapshot(self.now + datetime.timedelta(seconds=self.timestamp), points)
        time.sleep(.1)

    @staticmethod
    def f_square() -> OFFSET_EXAMPLES:
        iteration = 0
        frequency = 10
        while True:
            target_last = float((iteration - 1) % frequency >= frequency // 2),
            input_this = float(iteration % frequency < frequency // 2),
            yield iteration, target_last, input_this
            iteration += 1

    @staticmethod
    def nf_triangle() -> OFFSET_EXAMPLES:
        iteration = 0
        period = 10.
        while True:
            target_last = 2. * abs(iteration / period - math.floor(iteration / period + 1. / 2.)),
            input_this = 0.,
            yield iteration, target_last, input_this
            iteration += 1

    @staticmethod
    def nf_square() -> OFFSET_EXAMPLES:
        iteration = 0
        frequency = 10
        while True:
            target_last = float((iteration - 1) % frequency >= frequency // 2),
            input_this = 0.,
            yield iteration, target_last, input_this
            iteration += 1

    @staticmethod
    def nf_trigonometry() -> OFFSET_EXAMPLES:
        iteration = 0
        period = math.pi
        while True:
            # target_last = float((iteration - 1) % frequency >= frequency // 2),
            target_last = (math.cos(iteration / period) + 1.) / 2.,
            input_this = (math.sin((iteration + 1) / period) + 1.) / 2.,
            yield iteration, target_last, input_this
            iteration += 1

    def _states(self) -> OFFSET_EXAMPLES:
        return self.examples
