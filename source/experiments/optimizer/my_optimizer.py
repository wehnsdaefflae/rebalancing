#!/usr/bin/env python3
import itertools
import logging
from math import sin, cos
from typing import Callable, Sequence, Tuple, Optional, List

from matplotlib import pyplot


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


POINT = Tuple[float, ...]
AREA = Tuple[POINT, POINT]


class MyOptimizer:
    def __init__(self, evaluation_function: Callable[[POINT], float], region: AREA, limit: int = 1000):
        # convert from limits to points
        self.eval = evaluation_function
        self.region = region
        self.limit = limit

        self.first_value = True                             # type: bool
        self.best_value = 0.                                # type: float
        self.region_values = [(self.best_value, region)]    # type: List[Tuple[float, AREA]]
        self.best_parameters = None                         # type: Optional[POINT]

    def __enqueue(self, value: float, region: AREA):
        i = 0
        while i < len(self.region_values) and value < self.region_values[i][0]:
            i += 1
        self.region_values.insert(i, (value, region))

    def next(self):
        _, current_region = self.region_values.pop(0)
        current_center = MyOptimizer._get_center(current_region)
        current_value = self.eval(current_center)
        for each_sub_region in MyOptimizer._subdivide(current_region, current_center):
            self.__enqueue(current_value, each_sub_region)

        if self.best_value < current_value or self.first_value:
            self.best_value = current_value
            self.best_parameters = current_center
            self.first_value = False

        while 0 < self.limit < len(self.region_values):
            self.region_values.pop()

    @staticmethod
    def _get_center(borders: AREA) -> POINT:
        point_a, point_b = borders
        len_a, len_b = len(point_a), len(point_b)
        if len_a != len_b:
            raise ValueError("Border points not of equal dimension.")

        return tuple((point_a[_i] + point_b[_i]) / 2. for _i in range(len_a))

    @staticmethod
    def _subdivide(borders: AREA, center: POINT) -> Tuple[AREA, ...]:
        return tuple((_x, center) for _x in itertools.product(*zip(*borders)))


def main():
    f = lambda _x: sin(_x[0] / 50.) + cos(_x[0] / 17.)
    x_values = range(1000)
    y_values = [f((_x, )) for _x in x_values]

    o = MyOptimizer(f, ((0.,), (1000.,)))   # give limits, not edges

    pyplot.plot(x_values, y_values)

    for _ in range(1000):
        o.next()
        p = o.best_parameters
        v = o.best_value
        pyplot.plot(p, [v], "o")

    pyplot.show()


if __name__ == "__main__":
    main()

