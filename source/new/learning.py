from __future__ import annotations

import itertools
import json
import math
import random
from typing import TypeVar, Sequence, Generic, Dict, Any, Callable, Iterable, Tuple, Generator, Union, Type

import numpy

OUTPUT = TypeVar("OUTPUT", Sequence[float], float)
T = TypeVar("T")


def smear(average: float, value: float, inertia: int) -> float:
    return (inertia * average + value) / (inertia + 1.)


def accumulating_combinations_with_replacement(elements: Iterable[T], repetitions: int) -> Generator[Tuple[T, ...], None, None]:
    yield from (c for _r in range(repetitions) for c in itertools.combinations_with_replacement(elements, _r + 1))


def product(values: Sequence[float]) -> float:
    output = 1.
    for _v in values:
        output *= _v
    return output


class JsonSerializable(Generic[T]):
    @staticmethod
    def from_dict(d: Dict[str, Any]) -> T:
        """
        name_class = d["name_class"]
        name_module = d["name_module"]
        this_module = importlib.import_module(name_module)
        this_class = getattr(this_module, name_class)
        """
        raise NotImplementedError()

    def to_dict(self) -> Dict[str, Any]:
        """
        this_class = self.__class__
        d = {
            "name_class": this_class.__name__,
            "name_module": this_class.__module__,
        }
        """
        raise NotImplementedError()

    @staticmethod
    def load_from(path_name: str) -> T:
        with open(path_name, mode="r") as file:
            d = json.load(file)
            return JsonSerializable.from_dict(d)

    def save_as(self, path: str):
        with open(path, mode="w") as file:
            d = self.to_dict()
            json.dump(d, file, indent=2, sort_keys=True)


class Approximation(JsonSerializable, Generic[OUTPUT]):
    @staticmethod
    def from_dict(d: Dict[str, Any]) -> Approximation:
        raise NotImplementedError()

    def to_dict(self) -> Dict[str, Any]:
        raise NotImplementedError()

    def output(self, in_value: Sequence[float]) -> OUTPUT:
        raise NotImplementedError()

    def fit(self, in_value: Sequence[float], target_value: OUTPUT, drag: int):
        raise NotImplementedError()

    def __str__(self) -> str:
        return str(self.get_parameters())

    def get_parameters(self) -> Sequence[float]:
        raise NotImplementedError()


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
                 error_memory: Callable[[Sequence[float], Sequence[float]], float] = MultivariateRegression.error_distance
                 ):
        super().__init__(no_outputs, addends)
        self.regression_memory = MultipleRegression(addends_memory)
        self.error_memory = error_memory
        self.memory = 0.
        self.last_input = None

    def _get_memory(self, in_values: Sequence[float]) -> float:
        return self.regression_memory.output(in_values)

    def output(self, in_value: Sequence[float]) -> Sequence[float]:
        input_contextualized = tuple(in_value) + (self.memory, )
        self.memory = self._get_memory(input_contextualized)
        return super().output(input_contextualized)

    def fit(self, in_value: Sequence[float], target_value: Sequence[float], drag: int):
        output_value = super().output(tuple(in_value) + (self.memory, ))

        e = self.error_memory(output_value, target_value)
        # todo: replace by always adding parametrized noise
        p = 1. / (1. + e)   # probability of keeping memory
        if random.random() >= p:
            self.memory = random.random()

        if self.last_input is not None:
            self.regression_memory.fit(self.last_input, self.memory, drag)  # smaller, fixed drag?

        input_contextualized = tuple(in_value) + (self.memory, )
        super().fit(input_contextualized, target_value, drag)

    def get_parameters(self) -> Sequence[float]:
        return tuple(super().get_parameters()) + tuple(self.regression_memory.get_parameters())


class MultivariatePolynomialRecurrentRegression(MultivariateRecurrentRegression):
    def __init__(self, no_arguments: int, degree: int, no_outputs: int):
        addends_basic = MultiplePolynomialRegression.polynomial_addends(no_arguments + 1, degree)
        addends_memory = MultiplePolynomialRegression.polynomial_addends(no_arguments + 1, degree)
        super().__init__(no_outputs, addends_basic, addends_memory)


class Classification(Approximation[int]):
    @staticmethod
    def from_dict(d: Dict[str, Any]) -> Approximation:
        pass

    def to_dict(self) -> Dict[str, Any]:
        pass

    def __init__(self, regression: MultivariateRegression, no_classes: int):
        self.regression = regression
        self.no_classes = no_classes
        self.last_output = None

    @staticmethod
    def max_single(vector: Sequence[float]) -> int:
        index_max = -1
        value_max = 0.
        for i, v in enumerate(vector):
            if index_max < i or value_max < v:
                index_max = i
                value_max = v

            elif value_max == v:
                return -1

        return index_max

    @staticmethod
    def error_class(output_value: Sequence[float], target_value: Sequence[float]) -> float:
        index_output = Classification.max_single(output_value)
        index_target = Classification.max_single(target_value)
        return float(index_output != index_target or 0 >= index_target)

    def class_to_one_hot(self, index_class: int) -> Sequence[float]:
        return tuple(float(i == index_class) for i in range(self.no_classes))

    def one_hot_to_class(self, values: Sequence[float]) -> int:
        assert len(values) == self.no_classes
        index_class, _ = max(enumerate(values), key=lambda x: x[1])
        return index_class

    def get_details_last_output(self) -> Dict[str, Union[float, Sequence[float]]]:
        if self.last_output is None:
            return dict()

        cropped = tuple(min(1., max(0., v)) for v in self.last_output)
        s = sum(cropped)
        return {
            "raw output": tuple(self.last_output),
            "knowledgeability": s / self.no_classes,
            "decidedness": 0. if 0. >= s else max(cropped) / s,
        }

    def output(self, in_value: Sequence[float]) -> int:
        self.last_output = self.regression.output(in_value)
        output_class = self.one_hot_to_class(self.last_output)
        return output_class

    def fit(self, in_value: Sequence[float], target_class: int, drag: int):
        target_values = self.class_to_one_hot(target_class)
        self.regression.fit(in_value, target_values, drag)

    def get_parameters(self) -> Sequence[float]:
        return self.get_parameters()


class PolynomialClassification(Classification):
    def __init__(self, no_arguments: int, degree: int, no_classes: int):
        regression = MultivariatePolynomialRegression(no_arguments, degree, no_classes)
        super().__init__(regression, no_classes)


class RecurrentClassification(Classification):
    def __init__(self, no_classes: int, addends: Sequence[Callable[[Sequence[float]], float]], addends_memory: Sequence[Callable[[Sequence[float]], float]]):
        regression = MultivariateRecurrentRegression(no_classes, addends, addends_memory, error_memory=Classification.error_class)
        super().__init__(regression, no_classes)


class RecurrentPolynomialClassification(RecurrentClassification):
    def __init__(self, no_arguments: int, degree: int, no_classes: int):
        addends_basic = MultiplePolynomialRegression.polynomial_addends(no_arguments + 1, degree)
        addends_memory = MultiplePolynomialRegression.polynomial_addends(no_arguments + 1, degree)
        super().__init__(no_classes, addends_basic, addends_memory)


def get_classification_examples() -> Iterable[Tuple[Tuple[float, ...], int]]:
    return (((x, ), math.floor(4. * x)) for x in (random.random() for _ in range(1000)))


def classification_test():
    random.seed(234234525)

    examples = list(get_classification_examples())

    no_classes = 4
    r = PolynomialClassification(1, 3, no_classes)

    for i, (input_value, target_class) in enumerate(examples):
        output_class = r.output(input_value)
        print(f"{', '.join(f'{x:.2f}' for x in input_value):s} => {output_class:02d} / {target_class:02d} " + ("true" if output_class == target_class else "false"))
        r.fit(input_value, target_class, i)


def regression_test():
    # r = MultiplePolynomialRegression(1, 1)
    no_inputs = 10
    no_outputs = 10
    r = MultivariatePolynomialRecurrentRegression(no_inputs, 2, no_outputs)
    for t in range(100):
        input_values = [random.random() for _ in range(no_inputs)]
        output_values = r.output(input_values)

        target_values = [random.random() for _ in range(no_outputs)]
        r.fit(input_values, target_values, t + 1)

        print(output_values)
        print(target_values)
        print()


def main():
    # classification_test()
    regression_test()


if __name__ == "__main__":
    main()
