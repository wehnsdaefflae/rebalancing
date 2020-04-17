from typing import Sequence, Callable

from source.approximation.classification import ClassificationRegression
from source.approximation.regression import RegressionMultiplePolynomial
from source.approximation.regression_advanced import RegressionMultivariateRecurrent


class ClassificationRecurrent(ClassificationRegression):
    def __init__(self, no_classes: int, addends: Sequence[Callable[[Sequence[float]], float]], addends_memory: Sequence[Callable[[Sequence[float]], float]], no_memories: int = 1):
        regression = RegressionMultivariateRecurrent(no_classes, addends, addends_memory, resolution_memory=no_memories, error_memory=ClassificationRegression.error_class)
        super().__init__(regression)


class ClassificationRecurrentPolynomial(ClassificationRecurrent):
    def __init__(self, no_arguments: int, degree: int, no_classes: int, no_memories: int = 1):
        addends_basic = RegressionMultiplePolynomial.polynomial_addends(no_arguments + no_memories, degree)
        addends_memory = RegressionMultiplePolynomial.polynomial_addends(no_arguments + no_memories, degree)
        super().__init__(no_classes, addends_basic, addends_memory, no_memories=no_memories)
