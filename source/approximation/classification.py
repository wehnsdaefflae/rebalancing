from typing import Dict, Any, Sequence, Tuple, Generic, TypeVar, Hashable

from source.approximation.abstract import Approximation, ApproximationProbabilistic
from source.approximation.regression import RegressionMultivariate, RegressionMultivariatePolynomial
from source.tools.functions import max_single

INPUT_VALUE = TypeVar("INPUT_VALUE")


class Classification(Approximation[INPUT_VALUE, int], Generic[INPUT_VALUE]):
    @staticmethod
    def from_dict(d: Dict[str, Any]) -> Approximation:
        pass

    def to_dict(self) -> Dict[str, Any]:
        pass

    def output(self, in_value: INPUT_VALUE) -> int:
        raise NotImplementedError()

    def output_info(self, in_value: INPUT_VALUE) -> Tuple[int, Dict[str, Any]]:
        raise NotImplementedError()

    def fit(self, in_value: INPUT_VALUE, target_value: int, drag: int):
        raise NotImplementedError()

    def get_parameters(self) -> Sequence[float]:
        raise NotImplementedError()


INPUT_HASHABLE = TypeVar("INPUT_HASHABLE", bound=Hashable)


class ClassificationNaiveBayes(Classification[INPUT_HASHABLE], ApproximationProbabilistic[INPUT_HASHABLE, int], Generic[INPUT_HASHABLE]):
    @staticmethod
    def from_dict(d: Dict[str, Any]) -> Approximation:
        pass

    def to_dict(self) -> Dict[str, Any]:
        pass

    def __init__(self):
        self.frequencies = dict()

    def output(self, in_value: INPUT_HASHABLE) -> int:
        sub_dict = self.frequencies.get(in_value)
        if sub_dict is None:
            return -1
        output_class, _ = max(sub_dict.items(), key=lambda x: x[1])
        return output_class

    def get_probability(self, in_value: INPUT_HASHABLE, target: int, no_classes: int = -1):
        sub_dict = self.frequencies.get(in_value)
        if sub_dict is None:
            return 1.

        f = sub_dict.get(target, 0) + 1
        f_tot = sum(sub_dict.values()) + max(no_classes, len(sub_dict))
        return f / f_tot

    def output_info(self, in_value: INPUT_HASHABLE) -> Tuple[int, Dict[str, Any]]:
        sub_dict = self.frequencies.get(in_value)
        if sub_dict is None:
            return -1, {"k_total": 0, "p_normalized": 1.}
        f_tot = sum(sub_dict.values())
        output_class, f_max = max(sub_dict.items(), lambda x: x[1])
        f_tot_smooth = f_tot + len(sub_dict)
        info = {
            "k_total": f_tot,
            "p_normalized": 1. if 0 >= f_tot_smooth else (f_max + 1) / f_tot_smooth,
        }
        return output_class, info

    def fit(self, in_value: INPUT_HASHABLE, target_value: int, drag: int):
        sub_dict = self.frequencies.get(in_value)
        if sub_dict is None:
            sub_dict = {target_value: 1}
            self.frequencies[in_value] = sub_dict
        else:
            sub_dict[target_value] = sub_dict.get(target_value, 0) + 1

    def get_parameters(self) -> Sequence[float]:
        pass


class ClassificationRegression(Approximation[Sequence[float], int]):
    @staticmethod
    def from_dict(d: Dict[str, Any]) -> Approximation:
        pass

    def to_dict(self) -> Dict[str, Any]:
        pass

    def get_parameters(self) -> Sequence[float]:
        return self.regression.get_parameters()

    def __init__(self, regression: RegressionMultivariate):
        self.regression = regression
        self.no_classes = len(regression.regressions)

    @staticmethod
    def error_class(output_value: Sequence[float], target_value: Sequence[float]) -> float:
        index_output = max_single(output_value)
        index_target = max_single(target_value)
        return float(index_output != index_target or 0 >= index_target)

    @staticmethod
    def class_to_one_hot(index_class: int, no_classes: int) -> Sequence[float]:
        return tuple(float(i == index_class) for i in range(no_classes))

    @staticmethod
    def one_hot_to_class(values: Sequence[float]) -> int:
        index_class, _ = max(enumerate(values), key=lambda x: x[1])
        return index_class

    def output_info(self, in_value: Sequence[float]) -> Tuple[int, Dict[str, Any]]:
        output_values = self.regression.output(in_value)
        output_class, output_max = max(enumerate(output_values), key=lambda x: x[1])
        s = sum(output_values)
        info = {
            "output_values": output_values,
            "k": s / self.no_classes,
            "d": 0. if 0. >= s else output_max / s,
        }
        return output_class, info

    def output(self, in_value: Sequence[float]) -> int:
        output_values = self.regression.output(in_value)
        output_class, _ = max(enumerate(output_values), key=lambda x: x[1])
        return output_class

    def fit(self, in_value: Sequence[float], target_class: int, drag: int):
        target_values = ClassificationRegression.class_to_one_hot(target_class, self.no_classes)
        self.regression.fit(in_value, target_values, drag)


class ClassificationPolynomial(ClassificationRegression):
    def __init__(self, no_arguments: int, degree: int, no_classes: int):
        regression = RegressionMultivariatePolynomial(no_arguments, degree, no_classes)
        super().__init__(regression)
