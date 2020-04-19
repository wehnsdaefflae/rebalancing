from typing import Generic, TypeVar, Tuple, Sequence, Dict, Any, Callable

from source.approximation.abstract import ApproximationProbabilistic, Approximation
from source.approximation.classification import ClassificationNaiveBayes

INPUT_VALUE = TypeVar("INPUT_VALUE")
OUTPUT_VALUE = TypeVar("OUTPUT_VALUE")


class CapsuleIterating(ApproximationProbabilistic[INPUT_VALUE, OUTPUT_VALUE], Generic[INPUT_VALUE, OUTPUT_VALUE]):
    def __init__(self, approximation: ApproximationProbabilistic[INPUT_VALUE, OUTPUT_VALUE]):
        self.approximation = approximation
        self.iterations = 0

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> Approximation:
        pass

    def to_dict(self) -> Dict[str, Any]:
        pass

    def output(self, in_value: INPUT_VALUE) -> OUTPUT_VALUE:
        return self.approximation.output(in_value)

    def fit(self, in_value: INPUT_VALUE, target_value: OUTPUT_VALUE, drag: int = -1):
        self.approximation.fit(in_value, target_value, self.iterations)
        self.iterations += 1

    def get_probability(self, input_value: INPUT_VALUE, target_value: OUTPUT_VALUE) -> float:
        return self.approximation.get_probability(input_value, target_value)

    def get_parameters(self) -> Sequence[float]:
        return self.approximation.get_parameters()


class ApproximationSemioticModel(Approximation[INPUT_VALUE, OUTPUT_VALUE], Generic[INPUT_VALUE, OUTPUT_VALUE]):
    @staticmethod
    def from_dict(d: Dict[str, Any]) -> Approximation:
        pass

    def to_dict(self) -> Dict[str, Any]:
        pass

    def __init__(self,
                 minimal_probability: float,
                 factory_approximation: Callable[[], ApproximationProbabilistic[Tuple[INPUT_VALUE, ...], OUTPUT_VALUE]],
                 ):
        self.minimal_probability = minimal_probability
        self.classifier_parent = None
        self.make_approximation = lambda: CapsuleIterating(factory_approximation())
        self.classifier_current = self.make_approximation()
        self.index_classifier_current = 0
        self.classifiers = [self.classifier_current]

    def get_structure(self) -> Sequence[int]:
        structure = []
        model = self
        while model is not None:
            structure.append(len(model.classifiers))
            model = model.classifier_parent

        return structure

    def output(self, in_value: INPUT_VALUE) -> int:
        return self.classifier_current.output(in_value)

    def fit(self, in_value: INPUT_VALUE, target_value: OUTPUT_VALUE, drag: int):
        probability = self.classifier_current.get_probability(in_value, target_value)
        if probability < self.minimal_probability:
            #print("this doesnt fit")
            if self.classifier_parent is None:
                self.classifier_parent = ApproximationSemioticModel[Tuple[int, ...], int](self.minimal_probability, lambda: ClassificationNaiveBayes())

            in_value_parent = (self.index_classifier_current, )   # + (in_value, ) todo: maybe add input when classifying?
            index_successor = self.classifier_parent.output(in_value_parent)

            if index_successor < 0:
                probability = -1.
            else:
                probability = self.classifiers[index_successor].get_probability(in_value, target_value)

            if probability < self.minimal_probability:
                #print("next doesnt fit")
                index_successor = max(
                    range(len(self.classifiers)),
                    key=lambda x: self.classifiers[x].get_probability(in_value, target_value)
                )
                probability = self.classifiers[index_successor].get_probability(in_value, target_value)

            if probability < self.minimal_probability:
                #print("none fits")
                index_successor = len(self.classifiers)
                self.classifiers.append(self.make_approximation())

            self.classifier_parent.fit(in_value_parent, index_successor, -1)
            self.index_classifier_current = index_successor
            self.classifier_current = self.classifiers[self.index_classifier_current]

        self.classifier_current.fit(in_value, target_value)

    def get_parameters(self) -> Sequence[float]:
        pass
