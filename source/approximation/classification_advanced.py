from typing import Sequence, Callable, TypeVar, Generic, Tuple, Dict, Any

from source.approximation.classification import ClassificationRegression, ClassificationNaiveBayes
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


INPUT_VALUE = TypeVar("INPUT_VALUE")


class MixinClassificationHistoric(Generic[INPUT_VALUE]):
    def __init__(self, length_history: int):
        self.length_history = length_history
        self.history = [-1 for _ in range(length_history)]

    def output(self, in_value: INPUT_VALUE) -> int:
        history_tuple = tuple(self.history[1:] + [in_value])
        return super().output(history_tuple)

    def output_info(self, in_value: INPUT_VALUE) -> Tuple[int, Dict[str, Any]]:
        history_tuple = tuple(self.history[1:] + [in_value])
        return super().output_info(history_tuple)

    def fit(self, in_value: INPUT_VALUE, target_value: int, drag: int):
        self.history.append(in_value)
        del(self.history[:-self.length_history])
        history_tuple = tuple(self.history)
        super().fit(history_tuple, target_value, drag)


class ClassificationNaiveBayesHistoric(MixinClassificationHistoric[INPUT_VALUE], ClassificationNaiveBayes[Tuple[INPUT_VALUE, ...]], Generic[INPUT_VALUE]):
    def __init__(self, length_history: int):
        ClassificationNaiveBayes.__init__(self)
        MixinClassificationHistoric.__init__(self, length_history)
