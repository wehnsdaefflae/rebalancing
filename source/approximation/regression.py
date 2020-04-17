from __future__ import annotations
import math
from typing import Dict, Any, Sequence, Callable

import numpy

from source.approximation.abstract import Approximation
from source.tools.functions import smear, product, accumulating_combinations_with_replacement


class RegressionMultiple(Approximation[Sequence[float], float]):
    @staticmethod
    def from_dict(d: Dict[str, Any]) -> RegressionMultiple:
        pass

    def to_dict(self) -> Dict[str, Any]:
        pass

    def __init__(self, addends: Sequence[Callable[[Sequence[float]], float]]):
        self.addends = addends
        self.var_matrix = tuple([0. for _ in addends] for _ in addends)
        self.cov_vector = [0. for _ in addends]

    def fit(self, in_values: Sequence[float], target_value: float, drag: int):
        assert drag >= 0
        components = tuple(f_a(in_values) for f_a in self.addends)

        for i, component_a in enumerate(components):
            var_row = self.var_matrix[i]
            for j in range(i + 1):
                component_b = components[j]
                value = smear(var_row[j], component_a * component_b, drag)
                var_row[j] = value

                if j == i:
                    continue
                var_row_other = self.var_matrix[j]
                var_row_other[i] = value

            self.cov_vector[i] = smear(self.cov_vector[i], target_value * component_a, drag)

    def get_parameters(self) -> Sequence[float]:
        try:
            # gaussian elimination
            parameters = numpy.linalg.solve(self.var_matrix, self.cov_vector)
            return tuple(parameters)

        except numpy.linalg.linalg.LinAlgError:
            parameters = numpy.linalg.lstsq(self.var_matrix, self.cov_vector, rcond=None)[0]
            return tuple(parameters)
            # return tuple(0. for _ in self.addends)

    def output(self, in_values: Sequence[float]) -> float:
        parameters = self.get_parameters()
        results_addends = tuple(p * f_a(in_values) for p, f_a in zip(parameters, self.addends))
        return sum(results_addends)


class RegressionMultiplePolynomial(RegressionMultiple):
    @staticmethod
    def polynomial_addends(no_arguments: int, degree: int) -> Sequence[Callable[[Sequence[float]], float]]:
        # todo: make functions persistable by generation them from a function that is stored, retrieved with the json trick in JSONSerializable
        def create_product(indices: Sequence[int]) -> Callable[[Sequence[float]], float]:
            def product_select(x: Sequence[float]) -> float:
                l_x = len(x)
                assert no_arguments == l_x
                factors = []
                for i in indices:
                    assert i < l_x
                    factors.append(x[i])
                return product(factors)

            return product_select

        addends = [lambda _: 1.]
        for j in accumulating_combinations_with_replacement(range(no_arguments), degree):
            addends.append(create_product(j))

        return addends

    def __init__(self, no_arguments: int, degree: int):
        super().__init__(RegressionMultiplePolynomial.polynomial_addends(no_arguments, degree))


class RegressionMultivariate(Approximation[Sequence[float], Sequence[float]]):
    @staticmethod
    def from_dict(d: Dict[str, Any]) -> Approximation:
        pass

    def to_dict(self) -> Dict[str, Any]:
        pass

    def __init__(self, no_outputs: int, addends: Sequence[Callable[[Sequence[float]], float]]):
        self.regressions = tuple(RegressionMultiple(addends) for _ in range(no_outputs))

    @staticmethod
    def error_distance(output: Sequence[float], target: Sequence[float]) -> float:
        len_target = len(target)
        assert len(output) == len_target
        return math.sqrt(sum((a - b) ** 2. for a, b in zip(output, target)))

    @staticmethod
    def length(vector: Sequence[float]) -> float:
        return math.sqrt(sum(x ** 2. for x in vector))

    @staticmethod
    def normalize(vector: Sequence[float]) -> Sequence[float]:
        length = RegressionMultivariate.length(vector)
        if length < 0.:
            raise ValueError("Length of vector cannot be negative.")
        elif length == 0.:
            return tuple(vector)
        return tuple(x / length for x in vector)

    @staticmethod
    def error_distance_normalized(output: Sequence[float], target: Sequence[float]) -> float:
        output_normalized = RegressionMultivariate.normalize(output)
        target_normalized = RegressionMultivariate.normalize(target)

        sum_output = sum(output_normalized)
        sum_target = sum(target_normalized)

        if sum_output == sum_target == 0.:
            return 1.

        if 0. == sum_output or 0. == sum_target:
            return 1.

        return RegressionMultivariate.error_distance(output_normalized, target_normalized) // 2.

    def output(self, in_value: Sequence[float]) -> Sequence[float]:
        output_value = tuple(each_regression.output(in_value) for each_regression in self.regressions)
        return output_value

    def fit(self, in_value: Sequence[float], target_value: Sequence[float], drag: int):
        for each_regression, each_target in zip(self.regressions, target_value):
            each_regression.fit(in_value, each_target, drag)

    def get_parameters(self) -> Sequence[float]:
        return tuple(x for each_regression in self.regressions for x in each_regression.get_parameters())


class RegressionMultivariatePolynomial(RegressionMultivariate):
    def __init__(self, no_arguments: int, degree: int, no_outputs: int):
        super().__init__(no_outputs, RegressionMultiplePolynomial.polynomial_addends(no_arguments, degree))


