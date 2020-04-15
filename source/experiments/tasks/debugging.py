import datetime
import math
import random
import time
from typing import Sequence

from matplotlib import pyplot

from source.approximation.abstract import Approximation
from source.data.abstract import INPUT_VALUE, TARGET_VALUE, OFFSET_EXAMPLES
from source.experiments.tasks.abstract import Application, Experiment
from source.tools.moving_graph import MovingGraph


class TransformRational(Application):
    def __init__(self, name: str, approximation: Approximation[Sequence[float]]):
        super().__init__(name)
        self.name = name
        self.approximation = approximation
        self.iterations = 0
        self.d = dict()

    def __str__(self) -> str:
        return self.name

    def learn(self, input_value: INPUT_VALUE, target_value: TARGET_VALUE):
        # self.d[tuple(input_value)] = tuple(target_value)
        self.approximation.fit(input_value, target_value, self.iterations)

    def act(self, input_value: INPUT_VALUE) -> TARGET_VALUE:
        # return self.d.get(tuple(input_value), tuple(input_value))
        return self.approximation.output(input_value)


class ExperimentStatic(Experiment):
    def __init__(self, application: Application):
        super().__init__((application,))
        self.function = lambda x: math.sin(x)
        self.fig, self.subplot = pyplot.subplots(nrows=1, ncols=1)
        self.errors = [1. for _ in self.applications]
        self.max_x = 10.

        self.samples = []

    def _offset_examples(self) -> OFFSET_EXAMPLES:
        iterations = 0
        target_last = 0.
        while True:
            input_this = random.uniform(0., self.max_x)
            target_this = self.function(input_this)
            yield iterations, (target_last,), (input_this,)
            iterations += 1
            target_last = target_this

    def _perform(self, index_application: int, action: TARGET_VALUE):
        pass

    def _post_process(self):
        if None not in self.output_values_last:
            for i, each_output in enumerate(self.output_values_last):
                self.errors[i] = abs(each_output[0] - self.target_value_last[0])

        if self.input_value_last is None:
            return

        sample_new = self.input_value_last[0], self.target_value_last[0]
        self.samples.append(sample_new)

        self.subplot.clear()
        self.subplot.set_xlim(xmin=0., xmax=self.max_x)
        self.subplot.set_ylim(ymin=-2., ymax=2.)

        self.subplot.scatter(*list(zip(*self.samples)), alpha=.5)
        x = tuple(self.max_x * x / 100. for x in range(100))
        self.subplot.plot(x, tuple(self.function(_x) for _x in x))
        self.subplot.plot(x, tuple(self.applications[0].act([_x])[0] for _x in x), alpha=.5)

        pyplot.tight_layout()
        pyplot.pause(.05)
        time.sleep(.1)


class ExperimentTimeseries(Experiment):
    def __init__(self, applications: Sequence[Application]):
        super().__init__(applications)
        info_subplots = tuple(
            (
                f"{str(each_application):s}",
                ("input", "target", "output", "error"),
                None,
                None,
            ) for each_application in applications)

        self.graph = MovingGraph(
            info_subplots,
            20,
            interval_draw_ms=0,
        )

        self.now = datetime.datetime.now()

    def _offset_examples(self) -> OFFSET_EXAMPLES:
        iteration = 0
        frequency = 4
        while True:
            yield iteration, float((iteration - 1) % frequency >= frequency // 2), float(iteration % frequency < frequency // 2)
            iteration += 1

    def _perform(self, index_application: int, action: TARGET_VALUE):
        pass

    def _post_process(self):
        points = tuple(
            {
                "input": 0. if self.input_value_last is None else self.input_value_last[0],
                "target": self.target_value_last[0],
                "output": 0. if output_value_last is None else output_value_last[0],
                "error": 1. if output_value_last is None else abs(output_value_last[0] - self.target_value_last[0]),
            }
            for output_value_last in self.output_values_last
        )

        self.graph.add_snapshot(self.now + datetime.timedelta(seconds=self.timestamp), points)
        time.sleep(.1)

    def start(self):
        super().start()
        self.graph.show()
