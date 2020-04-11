import datetime
import math
import time
from typing import Sequence

from source.approximation.abstract import Approximation
from source.data.abstract import EXAMPLE, SNAPSHOT, STREAM_SNAPSHOTS
from source.experiments.tasks.abstract import Application, Experiment, RESULT
from source.tools.moving_graph import MovingGraph


class SineToCosine(Application):
    @staticmethod
    def is_valid_result(result: RESULT) -> bool:
        return "error" in result and "output" in result

    @staticmethod
    def is_valid_snapshot(snapshot: SNAPSHOT) -> bool:
        return all(x in snapshot for x in ("iteration", "input", "target"))

    def __init__(self, name: str, approximation: Approximation[Sequence[float]]):
        self.name = name
        self.approximation = approximation
        self.iteration = 0

    def __str__(self) -> str:
        return self.name

    @staticmethod
    def make_example(snapshot: SNAPSHOT) -> EXAMPLE:
        return snapshot["iteration"], (snapshot["input"], ), (snapshot["target"], )

    def _make_example(self, snapshot: SNAPSHOT) -> EXAMPLE:
        return SineToCosine.make_example(snapshot)

    def _cycle(self, example: EXAMPLE, act: bool) -> RESULT:
        iterations, input_value, target_value = example
        output_value = self.approximation.output(input_value)
        error = abs(output_value[0] - target_value[0])
        self.approximation.fit(input_value, target_value, self.iteration)
        return {"error": error, "output": output_value[0]}


class ExperimentTrigonometry(Experiment):
    def __init__(self, application: Application):
        super().__init__((application, ), 0)
        self.graph = MovingGraph(
            "values", ("input", "target", "predicted"),
            "error", ("error",),
            40, moving_average_secondary=None, interval_draw_ms=0
        )
        self.now = datetime.datetime.now()

    def _snapshots(self) -> STREAM_SNAPSHOTS:
        iteration = 0
        while True:
            stretched = iteration / 2
            yield {
                "iteration": iteration,
                "input": math.sin(stretched),
                "target": math.cos(stretched),
            }
            iteration += 1

    def _postprocess_results(self, snapshot: SNAPSHOT, results: Sequence[RESULT]):
        assert len(results) == 1
        result = results[0]
        index_time, input_values, target_values = SineToCosine.make_example(snapshot)
        error = result["error"]
        output_value = result["output"]
        self.graph.add_snapshot(self.now + datetime.timedelta(seconds=index_time), (input_values[0], target_values[0], output_value), (error, ))
        time.sleep(.1)
