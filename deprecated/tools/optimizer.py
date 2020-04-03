#!/usr/bin/env python3
import itertools
import json
from math import sqrt
from typing import Callable, Sequence, Tuple, List, Optional

from matplotlib import pyplot

from deprecated.data.data_processing import series_generator

RANGE = Tuple[float, float]
POINT = Tuple[float, ...]
SAMPLE = Tuple[POINT, float]
AREA = Tuple[POINT, POINT]
PRIORITY_ELEMENT = Tuple[float, POINT, AREA]


# TODO: make into generator with send
class StatefulOptimizer:
    def __init__(self, evaluation_function: Callable[..., float], ranges: Sequence[RANGE], limit: int = 1000):
        self.eval = evaluation_function                                         # type: Callable[[...], float]
        self.dimensionality = len(ranges)                                       # type: int
        self.limit = limit                                                      # type: int

        origin = tuple(min(_x) for _x in ranges)                                # type: POINT
        destination = tuple(max(_x) for _x in ranges)                           # type: POINT
        complete_region = origin, destination                                   # type: AREA
        genesis_element = 0., self.__center(complete_region), complete_region   # type: PRIORITY_ELEMENT
        self.priority_list = [genesis_element]                                  # type: List[PRIORITY_ELEMENT]
        self.cache_list = []                                                    # type: List[AREA]

        self.best_value = 0.                                                    # type: float
        self.best_parameters = None                                             # type: Optional[POINT]

    def __enlist(self, value: float, center: POINT, region: AREA):
        no_values = len(self.priority_list)                                     # type: int
        priority = self.__diagonal(region) * value                              # type: float
        i = 0                                                                   # type: int
        while i < no_values and priority < self.priority_list[i][0]:
            i += 1
        priority_element = priority, center, region                             # type: PRIORITY_ELEMENT
        self.priority_list.insert(i, priority_element)

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

    def next(self) -> SAMPLE:
        if len(self.cache_list) < 1:
            _, center, region = self.priority_list.pop(0)   # type: float, POINT, AREA
            sub_regions = self._divide(region, center)      # type: Tuple[AREA, ...]
            self.cache_list.extend(sub_regions)

        current_region = self.cache_list.pop()              # type: AREA
        current_center = self.__center(current_region)      # type: POINT
        current_value = self.eval(*current_center)          # type: float

        if self.best_value < current_value:
            self.best_value = current_value                 # type: float
            self.best_parameters = current_center           # type: POINT

        self.__enlist(current_value, current_center, current_region)

        while 0 < self.limit < len(self.priority_list):
            self.priority_list.pop()

        return current_center, current_value


def main():
    with open("../../configs/time_series.json", mode="r") as file:
        config = json.load(file)

    time_series = series_generator(config["data_dir"] + "QTUMETH.csv",
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
    o = StatefulOptimizer(f, parameter_ranges)

    pyplot.plot(x_values, y_values)

    last_best = float("-inf")
    for _i in range(20):
        c, v = o.next()
        pyplot.axvline(x=c[0], alpha=.1)

        if last_best < v:
            print("{:05.2f}% of maximum after {:d} iterations".format(100. * v / max_value, _i))
            p = o.best_parameters
            v = o.best_value
            pyplot.plot(p, [v], "o")
            last_best = v

    pyplot.show()


if __name__ == "__main__":
    main()

