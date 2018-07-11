#!/usr/bin/env python3
import itertools
import logging
import sys
from math import sin, cos, sqrt
from typing import Callable, Sequence, Tuple, Optional, List, Union

from matplotlib import pyplot

from source.data.data_generation import DEBUG_SERIES
from source.experiments.timer import Timer


class PriorityQueue(list):
    def __init__(self, evaluation_function=lambda x: x):
        super().__init__()
        self.evaluation_function = evaluation_function

    def pop(self, index=None):
        if index is None:
            return super(PriorityQueue, self).pop()[1]
        return super(PriorityQueue, self).pop(index)[1]

    def enqueue(self, element):
        this_v = self.evaluation_function(element)
        l = len(self)
        new_entry = (this_v, element)
        if l < 1:
            self.append(new_entry)
        else:
            i = 0
            while i < l and self[i][0] < this_v:
                i += 1
            if i == l:
                self.append(new_entry)
            else:
                self.insert(i, new_entry)


class EquiprobabilityOptimization:
    def __init__(self, parameter_range, evaluation_function, depth_limit=-1):
        self.external_evaluation = evaluation_function
        parameters = EquiprobabilitySampling.get_center(parameter_range)
        self.maximum_entry = (0, parameters, parameter_range, self.external_evaluation(parameters))
        self.priority_queue = PriorityQueue(evaluation_function=lambda x: x[3] / (x[0] + 1))
        self.priority_queue.enqueue(self.maximum_entry)
        self.depth_limit = depth_limit

    def next_parameters(self):
        if len(self.priority_queue) < 1:
            raise StopIteration

        depth, center, space, _ = self.priority_queue.pop()
        if 0 < self.depth_limit <= depth:
            logging.warning("Depth limit <{}> reached!".format(depth))
            return center

        subspaces = EquiprobabilitySampling.get_sub_spaces(space, center)
        for each_space in subspaces:
            parameters = EquiprobabilitySampling.get_center(each_space)
            evaluation = self.external_evaluation(parameters)
            entry = (depth + 1, parameters, each_space, evaluation)
            self.priority_queue.enqueue(entry)
            if self.maximum_entry[3] < evaluation:
                self.maximum_entry = entry
                msg = "Maximum updated from <{}: {}> to <{}: {}>."
                msg = msg.format(self.maximum_entry[1], self.maximum_entry[3], parameters, evaluation)
                logging.info(msg)

        return center

    def get_maximum_parameters(self):
        return self.maximum_entry[1]


class EquiprobabilitySampling:
    def __init__(self, parameter_range):
        self.queue = [parameter_range]
        self.dimensionality = len(parameter_range)

    @staticmethod
    def get_sub_spaces(parameter_range, pivot_point):
        indices = []
        for i in range(2 ** len(parameter_range)):
            format_string = "{:0" + str(len(parameter_range)) + "b}"
            binary_string = format_string.format(i)
            indices.append([int(x) for x in binary_string])

        sub_spaces = []
        for i, each_indices in enumerate(indices):
            each_range = [list(x) for x in parameter_range]
            for j, each_index in enumerate(each_indices):
                each_range[j][each_index] = pivot_point[j]
            each_range = tuple(each_range)
            sub_spaces.append(each_range)

        return sub_spaces

    @staticmethod
    def get_center(extension):
        return [lo + (hi - lo) / 2 for lo, hi in extension]

    def get_next_parameters(self):
        this_range = self.queue.pop()
        parameters = EquiprobabilitySampling.get_center(this_range)
        sub_spaces = EquiprobabilitySampling.get_sub_spaces(this_range, parameters)

        sub_spaces.extend(self.queue)
        self.queue = sub_spaces

        return parameters


RANGE = Tuple[float, float]
POINT = Tuple[float, ...]
AREA = Tuple[POINT, POINT]
PRIOELEM = Tuple[float, POINT, AREA]


class MyOptimizer:
    def __init__(self, evaluation_function: Callable[[POINT], float], ranges: Sequence[RANGE], limit: int = 1000):
        # convert from limits to points
        self.eval = evaluation_function
        origin = tuple(min(_x) for _x in ranges)                             # type: POINT
        destination = tuple(max(_x) for _x in ranges)                        # type: POINT
        self.region = (origin, destination)                                  # type: AREA
        self.full_diameter = MyOptimizer.__diagonal(self.region)             # type: float
        self.limit = limit

        self.best_value = float("-inf")                                      # type: float
        self.best_parameters = MyOptimizer._get_center(self.region)          # type: POINT
        first_entry = self.best_value, self.best_parameters, self.region     # type: PRIOELEM
        self.region_values = [first_entry]                                   # type: List[PRIOELEM]

    def __enqueue(self, prio_elem: PRIOELEM):
        i = 0
        while i < len(self.region_values) and prio_elem[0] < self.region_values[i][0]:
            i += 1
        self.region_values.insert(i, prio_elem)

    @staticmethod
    def __diagonal(region: AREA) -> float:
        point_a, point_b = region
        len_a, len_b = len(point_a), len(point_b)
        if len_a != len_b:
            raise ValueError("Border points not of equal dimension.")

        return sqrt(sum((point_a[_i] - point_b[_i]) ** 2. for _i in range(len_a)))

    def next(self) -> POINT:
        _, current_center, current_region = self.region_values.pop(0)
        for each_sub_region in MyOptimizer._subdivide(current_region, current_center):
            each_center = MyOptimizer._get_center(each_sub_region)                      # type: POINT
            each_value = self.eval(each_center)                                         # type: float
            priority = each_value + MyOptimizer.__diagonal(each_sub_region)             # type: float
            element = priority, each_center, each_sub_region                            # type: PRIOELEM
            self.__enqueue(element)

            if self.best_value < each_value:
                self.best_value = each_value
                self.best_parameters = each_center

        while 0 < self.limit < len(self.region_values):
            self.region_values.pop()

        return current_center

    @staticmethod
    def _get_center(borders: AREA) -> POINT:
        point_a, point_b = borders
        len_a, len_b = len(point_a), len(point_b)
        if len_a != len_b:
            raise ValueError("Border points not of equal dimension.")

        return tuple((point_a[_i] + point_b[_i]) / 2. for _i in range(len_a))

    @staticmethod
    def _subdivide(borders: AREA, center: POINT) -> Tuple[AREA, ...]:
        # check circular equidistant distribution
        return tuple((_x, center) for _x in itertools.product(*zip(*borders)))


def main():
    y_values = [_x[1] for _x in DEBUG_SERIES("BNB", config_path="../../../configs/config.json")]
    max_value = max(y_values)
    length = len(y_values)
    x_values = list(range(length))
    f = lambda _x: y_values[round(_x[0])]

    o = MyOptimizer(f, ((0., length), ))

    pyplot.plot(x_values, y_values)

    last_best = float("-inf")
    for _i in range(10000):
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

