from __future__ import annotations
import math
import random
from typing import Dict, Any, Sequence, Callable, List

import numpy

from source.new.learning.tools import smear, product, accumulating_combinations_with_replacement, z_score_normalized_generator
from source.new.learning.approximation import Approximation


class MultipleRegression(Approximation[float]):
    @staticmethod
    def from_dict(d: Dict[str, Any]) -> MultipleRegression:
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
            return tuple(numpy.linalg.solve(self.var_matrix, self.cov_vector))

        except numpy.linalg.linalg.LinAlgError:
            return tuple(0. for _ in self.addends)

    def output(self, in_values: Sequence[float]) -> float:
        parameters = self.get_parameters()
        return sum(p * f_a(in_values) for p, f_a in zip(parameters, self.addends))


class MultiplePolynomialRegression(MultipleRegression):
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
        super().__init__(MultiplePolynomialRegression.polynomial_addends(no_arguments, degree))


class MultivariateRegression(Approximation[Sequence[float]]):
    @staticmethod
    def from_dict(d: Dict[str, Any]) -> Approximation:
        pass

    def to_dict(self) -> Dict[str, Any]:
        pass

    def __init__(self, no_outputs: int, addends: Sequence[Callable[[Sequence[float]], float]]):
        self.regressions = tuple(MultipleRegression(addends) for _ in range(no_outputs))

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
        length = MultivariateRegression.length(vector)
        if length < 0.:
            raise ValueError("Length of vector cannot be negative.")
        elif length == 0.:
            return tuple(vector)
        return tuple(x / length for x in vector)

    @staticmethod
    def error_distance_normalized(output: Sequence[float], target: Sequence[float]) -> float:
        output_normalized = MultivariateRegression.normalize(output)
        target_normalized = MultivariateRegression.normalize(target)

        sum_output = sum(output_normalized)
        sum_target = sum(target_normalized)

        if sum_output == sum_target == 0.:
            return 1.

        if 0. == sum_output or 0. == sum_target:
            return 1.

        return MultivariateRegression.error_distance(output_normalized, target_normalized) // 2.

    def output(self, in_value: Sequence[float]) -> Sequence[float]:
        return tuple(each_regression.output(in_value) for each_regression in self.regressions)

    def fit(self, in_value: Sequence[float], target_value: Sequence[float], drag: int):
        for each_regression, each_target in zip(self.regressions, target_value):
            each_regression.fit(in_value, each_target, drag)

    def get_parameters(self) -> Sequence[float]:
        return tuple(x for each_regression in self.regressions for x in each_regression.get_parameters())


class MultivariatePolynomialRegression(MultivariateRegression):
    def __init__(self, no_arguments: int, degree: int, no_outputs: int):
        super().__init__(no_outputs, MultiplePolynomialRegression.polynomial_addends(no_arguments, degree))


class MultivariateRecurrentRegression(MultivariateRegression):
    def __init__(self,
                 no_outputs: int,
                 addends: Sequence[Callable[[Sequence[float]], float]], addends_memory: Sequence[Callable[[Sequence[float]], float]],
                 resolution_memory: int = 1,
                 error_memory: Callable[[Sequence[float], Sequence[float]], float] = MultivariateRegression.error_distance
                 ):
        super().__init__(no_outputs, addends)
        self.regressions_memory = tuple(MultipleRegression(addends_memory) for _ in range(resolution_memory))
        self.values_memory = [0. for _ in range(resolution_memory)]
        self.error_memory = error_memory
        self.last_input = None
        self.normalization = z_score_normalized_generator()
        next(self.normalization)

    def _get_memory(self, in_values: Sequence[float]) -> List[float]:
        return [each_memory.output(in_values) for each_memory in self.regressions_memory]

    def output(self, in_value: Sequence[float]) -> Sequence[float]:
        input_contextualized = tuple(in_value) + tuple(self.values_memory)
        self.values_memory = self._get_memory(input_contextualized)
        return super().output(input_contextualized)

    def fit(self, in_value: Sequence[float], target_value: Sequence[float], drag: int):
        output_value = super().output(tuple(in_value) + tuple(self.values_memory))

        e = self.error_memory(output_value, target_value)
        e_z = self.normalization.send(e)
        for i in range(len(self.values_memory)):
            if random.random() < e_z:
                self.values_memory[i] = random.random()

        if self.last_input is not None:
            for each_memory, each_value in zip(self.regressions_memory, self.values_memory):
                each_memory.fit(self.last_input, each_value, drag)  # smaller, fixed drag?

        input_contextualized = tuple(in_value) + tuple(self.values_memory)
        super().fit(input_contextualized, target_value, drag)

    def get_parameters(self) -> Sequence[float]:
        return tuple(super().get_parameters()) + tuple(x for each_memory in self.regressions_memory for x in each_memory.get_parameters())


class MultivariatePolynomialRecurrentRegression(MultivariateRecurrentRegression):
    def __init__(self,
                 no_arguments: int, degree: int, no_outputs: int,
                 resolution_memory: int = 1,
                 error_memory: Callable[[Sequence[float], Sequence[float]], float] = MultivariateRegression.error_distance
                 ):
        addends_basic = MultiplePolynomialRegression.polynomial_addends(no_arguments + resolution_memory, degree)
        addends_memory = MultiplePolynomialRegression.polynomial_addends(no_arguments + resolution_memory, degree)
        super().__init__(no_outputs, addends_basic, addends_memory, resolution_memory=resolution_memory, error_memory=error_memory)


class MultivariateFailureRegression(MultivariateRegression):
    def __init__(self,
                 no_arguments: int,
                 no_outputs: int, addends: Sequence[Callable[[Sequence[float]], float]],
                 resolution_context: int, addends_context: Sequence[Callable[[Sequence[float]], float]],
                 error_context: Callable[[Sequence[float], Sequence[float]], float] = MultivariateRegression.error_distance
                 ):
        super().__init__(no_outputs, addends)
        self.context = MultivariateRegression(resolution_context, addends_context)
        self.error_context = error_context
        self.context = tuple(0. for _ in range(no_arguments + no_outputs))

    def fit(self, in_value: Sequence[float], target_value: Sequence[float], drag: int):
        # todo: continue here
        super().fit()

    def output(self, in_value: Sequence[float]) -> Sequence[float]:
        pass