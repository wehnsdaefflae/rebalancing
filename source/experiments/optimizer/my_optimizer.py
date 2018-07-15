#!/usr/bin/env python3
import itertools
import json
from math import sqrt
from typing import Callable, Sequence, Tuple, List, Optional

from matplotlib import pyplot

from source.data.data_generation import series_generator

RANGE = Tuple[float, float]
POINT = Tuple[float, ...]
SAMPLE = Tuple[POINT, float]
AREA = Tuple[POINT, POINT]
PRIORITY_ELEMENT = Tuple[float, POINT, AREA]


class MyOptimizer:
    def __init__(self, evaluation_function: Callable[..., float], ranges: Sequence[RANGE], limit: int = 1000):
        self.eval = evaluation_function                                         # type: Callable[..., float]
        self.dimensionality = len(ranges)                                       # type: int
        origin = tuple(min(_x) for _x in ranges)                                # type: POINT
        destination = tuple(max(_x) for _x in ranges)                           # type: POINT
        self.region = origin, destination                                       # type: AREA
        self.limit = limit                                                      # type: int

        self.best_parameters, self.best_value = self.__evaluate(self.region)    # type: POINT, float
        first_elem = self.best_value, self.best_parameters, self.region         # type: PRIORITY_ELEMENT
        self.major_priority_list = [first_elem]                                 # type: List[PRIORITY_ELEMENT]
        self.minor_priority_list = []                                           # type: List[PRIORITY_ELEMENT]

    def __evaluate(self, region: AREA) -> SAMPLE:
        center = self.__center(region)                                          # type: POINT
        return center, self.eval(*center)

    def __enqueue(self, priority_element: PRIORITY_ELEMENT):
        no_values = len(self.major_priority_list)                                     # type: int
        element_value = priority_element[0]                                     # type: float
        i = 0                                                                   # type: int
        while i < no_values and element_value < self.major_priority_list[i][0]:
            i += 1
        self.major_priority_list.insert(i, priority_element)

    def __check_edges(self, point_a: POINT, point_b: POINT):
        len_a, len_b = len(point_a), len(point_b)                               # type: int, int
        if not (len_a == len_b == self.dimensionality):
            raise ValueError("Not all edges have a dimensionality of {:d}.".format(self.dimensionality))

    def __diagonal(self, region: AREA) -> float:
        point_a, point_b = region                                               # type: POINT, POINT
        self.__check_edges(point_a, point_b)
        return sqrt(sum((point_a[_i] - point_b[_i]) ** 2. for _i in range(self.dimensionality)))

    def __center(self, region: AREA) -> POINT:
        point_a, point_b = region                                               # type: POINT, POINT
        self.__check_edges(point_a, point_b)
        return tuple((point_a[_i] + point_b[_i]) / 2. for _i in range(self.dimensionality))

    @staticmethod
    def _divide(borders: AREA, center: POINT) -> Tuple[AREA, ...]:
        return tuple((_x, center) for _x in itertools.product(*zip(*borders)))

    def _next_region(self, region: AREA) -> Optional[Tuple[POINT, float]]:
        center, value = self.__evaluate(region)                 # type: POINT, float
        if len(center) != self.dimensionality:
            msg = "Expected {:d} dimensions, received {:d}."
            raise ValueError(msg.format(self.dimensionality, len(center)))
        if value < 0.:
            raise ValueError("Evaluation cannot be negative.")

        diagonal = self.__diagonal(region)                      # type: float
        if 0. >= diagonal:
            return None
        element = value * diagonal, center, region              # type: PRIORITY_ELEMENT
        self.__enqueue(element)

        if self.best_value < value:
            self.best_value = value                             # type: float
            self.best_parameters = center                       # type: POINT
        return center, value

    def next(self) -> Tuple[SAMPLE, ...]:
        current_value, current_center, current_region = self.major_priority_list.pop(0)  # type: float, POINT, AREA
        sub_regions = MyOptimizer._divide(current_region, current_center)                # type: Tuple[AREA, ...]
        samples = tuple(self._next_region(each_region) for each_region in sub_regions)

        while 0 < self.limit < len(self.major_priority_list):
            self.major_priority_list.pop()

        return samples


def main():
    with open("../../../configs/time_series.json", mode="r") as file:
        config = json.load(file)

    time_series = series_generator(config["data_dir"] + "QTUMETH.csv",
                                   start_time=config["start_time"],
                                   end_time=config["end_time"],
                                   interval_minutes=config["interval_minutes"])
    y_values = [_x[1] for _x in time_series]
    length = len(y_values)
    x_values = list(range(length))
    f = lambda _x: y_values[round(_x)]

    #length = 1000
    #x_values = list(range(length))
    #f = lambda _x: sin(_x * .07) + cos(_x * .03) + 5.
    #y_values = [f(x) for x in x_values]

    max_value = max(y_values)
    parameter_ranges = (0., length),
    o = MyOptimizer(f, parameter_ranges)

    pyplot.plot(x_values, y_values)

    last_best = float("-inf")
    for _i in range(50):
        c = o.next()
        pyplot.axvline(x=c[0], alpha=.1)

        p = o.best_parameters
        v = o.best_value
        pyplot.plot(p, [v], "o")

        if last_best < v:
            print("{:05.2f}% of maximum after {:d} iterations".format(100. * v / max_value, _i))
            if v >= max_value:
                pyplot.axvline(x=p[0], alpha=1.)
                break
            last_best = v

    pyplot.show()


if __name__ == "__main__":
    main()

