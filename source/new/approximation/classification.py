from typing import Dict, Any, Sequence, Union, Callable

from source.new.learning._abstract import Approximation
from source.new.learning.regression import MultiplePolynomialRegression, MultivariateRegression, MultivariatePolynomialRegression, MultivariateRecurrentRegression


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
    def __init__(self, no_classes: int, addends: Sequence[Callable[[Sequence[float]], float]], addends_memory: Sequence[Callable[[Sequence[float]], float]], no_memories: int = 1):
        regression = MultivariateRecurrentRegression(no_classes, addends, addends_memory, resolution_memory=no_memories, error_memory=Classification.error_class)
        super().__init__(regression, no_classes)


class RecurrentPolynomialClassification(RecurrentClassification):
    def __init__(self, no_arguments: int, degree: int, no_classes: int, no_memories: int = 1):
        addends_basic = MultiplePolynomialRegression.polynomial_addends(no_arguments + no_memories, degree)
        addends_memory = MultiplePolynomialRegression.polynomial_addends(no_arguments + no_memories, degree)
        super().__init__(no_classes, addends_basic, addends_memory, no_memories=no_memories)