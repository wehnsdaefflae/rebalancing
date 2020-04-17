from typing import Generic, TypeVar, Tuple, Sequence, Dict, Any, Callable

from source.approximation.abstract import ApproximationProbabilistic, Approximation
from source.approximation.classification import ClassificationNaiveBayes

INPUT_VALUE = TypeVar("INPUT_VALUE")
OUTPUT_VALUE = TypeVar("OUTPUT_VALUE")


class ApproximationSemioticModel(Approximation[INPUT_VALUE, OUTPUT_VALUE], Generic[INPUT_VALUE, OUTPUT_VALUE]):
    @staticmethod
    def from_dict(d: Dict[str, Any]) -> Approximation:
        pass

    def to_dict(self) -> Dict[str, Any]:
        pass

    def __init__(self,
                 threshold: float,
                 factory_approximation: Callable[[], ApproximationProbabilistic[Tuple[INPUT_VALUE, ...], OUTPUT_VALUE]],
                 ):
        self.threshold = threshold
        self.classifier_parent = None
        self.make_approximation = factory_approximation
        self.classifier_current = factory_approximation()
        self.index_classifier_current = 0
        self.classifiers = [self.classifier_current]

    def output(self, in_value: INPUT_VALUE) -> int:
        return self.classifier_current.output(in_value)

    def fit(self, in_value: INPUT_VALUE, target_value: OUTPUT_VALUE, drag: int):
        probability = self.classifier_current.get_probability(in_value, target_value)
        if probability < self.threshold:
            if self.classifier_parent is None:
                self.classifier_parent = ApproximationSemioticModel[Tuple[int, ...]](self.threshold, 1, ClassificationNaiveBayes[Tuple[int, ...]])

            in_value_parent = (self.index_classifier_current, ) + (in_value, )
            index_successor = self.classifier_parent.output(in_value_parent)

            probability = self.classifiers[index_successor].get_probability(in_value, target_value)

            if probability < self.threshold:
                index_successor = max(
                    range(len(self.classifiers)),
                    key=lambda x: self.classifiers[x].get_probability(in_value, target_value)
                )
                probability = self.classifiers[index_successor].get_probability(in_value, target_value)

            if probability < self.threshold:
                index_successor = len(self.classifiers)
                self.classifiers.append(self.make_approximation())

            self.classifier_parent.fit(in_value_parent, index_successor)
            self.index_classifier_current = index_successor
            self.classifier_current = self.classifiers[self.index_classifier_current]

        self.classifier_current.fit(in_value, target_value, -1)

    def get_parameters(self) -> Sequence[float]:
        pass
