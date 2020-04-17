from typing import Generic, Type, TypeVar, Tuple, Sequence, Dict, Any

from source.approximation.abstract import ApproximationProbabilistic, Approximation
from source.approximation.classification import ClassificationNaiveBayes

INPUT_VALUE = TypeVar("INPUT_VALUE")
TARGET_VALUE = TypeVar("TARGET_VALUE")


class ApproximationSemioticModel(Approximation[INPUT_VALUE, TARGET_VALUE], Generic[INPUT_VALUE, TARGET_VALUE]):
    @staticmethod
    def from_dict(d: Dict[str, Any]) -> Approximation:
        pass

    def to_dict(self) -> Dict[str, Any]:
        pass

    def __init__(self,
                 threshold: float, length_history: int,
                 approximation_base_class: Type[ApproximationProbabilistic[Tuple[INPUT_VALUE, ...], TARGET_VALUE]],
                 ):
        self.threshold = threshold
        self.classifier_parent = None
        self.class_base = approximation_base_class
        self.classifier_current = approximation_base_class()
        self.index_classifier_current = 0
        self.classifiers = [self.classifier_current]

        self.length_history = length_history
        self.history = [-1 for _ in range(length_history)]

    def output(self, in_value: INPUT_VALUE) -> int:
        history_tuple = tuple(self.history[1:] + [in_value])
        return self.classifier_current.output(history_tuple)

    def fit(self, in_value: INPUT_VALUE, target_value: TARGET_VALUE, drag: int):
        self.history.append(in_value)
        del(self.history[:-self.length_history])
        history_tuple = tuple(self.history)

        probability = self.classifier_current.get_probability(history_tuple, target_value)
        if probability < self.threshold:
            if self.classifier_parent is None:
                self.classifier_parent = ApproximationSemioticModel[Tuple[int, ...]](self.threshold, 1, ClassificationNaiveBayes[Tuple[int, ...]])

            in_value_parent = (self.index_classifier_current, ) + history_tuple
            index_successor = self.classifier_parent.output(in_value_parent)

            probability = self.classifiers[index_successor].get_probability(history_tuple, target_value)

            if probability < self.threshold:
                index_successor = max(
                    range(len(self.classifiers)),
                    key=lambda x: self.classifiers[x].get_probability(history_tuple, target_value)
                )
                probability = self.classifiers[index_successor].get_probability(history_tuple, target_value)

            if probability < self.threshold:
                index_successor = len(self.classifiers)
                self.classifiers.append(self.class_base())

            self.classifier_parent.fit(in_value_parent, index_successor)
            self.index_classifier_current = index_successor
            self.classifier_current = self.classifiers[self.index_classifier_current]

        self.classifier_current.fit(history_tuple, target_value, -1)

    def get_parameters(self) -> Sequence[float]:
        pass
