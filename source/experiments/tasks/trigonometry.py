import datetime
import time
from typing import Sequence

from source.approximation.abstract import Approximation
from source.data.abstract import EXAMPLE, SNAPSHOT, STREAM_SNAPSHOTS, INPUT_VALUE, TARGET_VALUE
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
        self.d[tuple(input_value)] = tuple(target_value)
        # self.approximation.fit(input_value, target_value, self.iterations)

    def act(self, input_value: INPUT_VALUE) -> TARGET_VALUE:
        return self.d.get(tuple(input_value), tuple(input_value))
        # return self.approximation.output(input_value)


class ExperimentTrigonometry(Experiment):
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

    def _snapshots(self) -> STREAM_SNAPSHOTS:
        iteration = 0
        while True:
            yield {
                "iteration": iteration,
                # "input": 0.,
                "input": float(iteration % 10 < 5),
                # "target": math.sin(iteration / 2.),
                "target_last": float((iteration - 1) % 10 >= 5),
            }
            iteration += 1

    def _get_offset_example(self, snapshot: SNAPSHOT) -> EXAMPLE:
        return (snapshot["target_last"], ), (snapshot["input"], )

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

        self.graph.add_timestep(self.now + datetime.timedelta(seconds=self.iterations), points)
        time.sleep(.1)

    def start(self):
        super().start()
        self.graph.show()
