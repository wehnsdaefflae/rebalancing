from typing import Dict, Any, Sequence, Callable, Tuple

from source.approximation.abstract import Approximation
from source.approximation.regression import MultivariateRegression, MultivariatePolynomialRegression, MultivariateRecurrentRegression, MultiplePolynomialRegression


class Classification(Approximation[int, int]):
    @staticmethod
    def from_dict(d: Dict[str, Any]) -> Approximation:
        pass

    def to_dict(self) -> Dict[str, Any]:
        pass

    def output(self, in_value: int) -> int:
        raise NotImplementedError()

    def output_info(self, in_value: int) -> Tuple[int, Dict[str, Any]]:
        raise NotImplementedError()

    def fit(self, in_value: int, target_value: int, drag: int):
        raise NotImplementedError()

    def get_parameters(self) -> Sequence[float]:
        raise NotImplementedError()


class NaiveBayesClassification(Classification):
    def __init__(self, regression: MultivariateRegression, length_history: int):
        super().__init__(regression)
        self.length_history = length_history
        self.history = [-1 for _ in range(length_history)]
        self.frequencies = dict()

    def output(self, in_value: int) -> int:
        history_tuple = tuple(self.history[1:] + [in_value])
        sub_dict = self.frequencies.get(history_tuple)
        if sub_dict is None:
            return -1
        output_class, _ = max(sub_dict.items(), lambda x: x[1])
        return output_class

    def output_info(self, in_value: int) -> Tuple[int, Dict[str, Any]]:
        history_tuple = tuple(self.history[1:] + [in_value])
        sub_dict = self.frequencies.get(history_tuple)
        if sub_dict is None:
            return -1, dict()
        f = tuple(sub_dict.values())
        s = sum(f)
        output_class, f_max = max(sub_dict.items(), lambda x: x[1])
        info = {
            "k_total": s,
            "d_normalized": 0. if 0. >= s else f_max / s,
        }
        return output_class, info

    def fit(self, in_value: int, target_value: int, drag: int):
        self.history.append(in_value)
        del(self.history[:-self.length_history])
        history_tuple = tuple(self.history)
        sub_dict = self.frequencies.get(history_tuple)
        if sub_dict is None:
            sub_dict = {target_value: 1}
            self.frequencies[history_tuple] = sub_dict
        else:
            sub_dict[target_value] = sub_dict.get(target_value, 0) + 1

    def get_parameters(self) -> Sequence[float]:
        pass


class SemioticModelClassification(Classification):
    def __init__(self, threshold: float):
        self.threshold = threshold

    def output(self, in_value: int) -> int:
        pass

    def output_info(self, in_value: int) -> Tuple[int, Dict[str, Any]]:
        pass

    def fit(self, in_value: int, target_value: int, drag: int):
        pass

    def get_parameters(self) -> Sequence[float]:
        pass

    def _get_sub_classification(self) -> Classification:
        pass


class RegressionClassification(Approximation[Sequence[float], int]):
    @staticmethod
    def from_dict(d: Dict[str, Any]) -> Approximation:
        pass

    def to_dict(self) -> Dict[str, Any]:
        pass

    def get_parameters(self) -> Sequence[float]:
        return self.regression.get_parameters()

    def __init__(self, regression: MultivariateRegression):
        self.regression = regression
        self.no_classes = len(regression.regressions)

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
        index_output = RegressionClassification.max_single(output_value)
        index_target = RegressionClassification.max_single(target_value)
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
        target_values = RegressionClassification.class_to_one_hot(target_class, self.no_classes)
        self.regression.fit(in_value, target_values, drag)


class PolynomialClassification(RegressionClassification):
    def __init__(self, no_arguments: int, degree: int, no_classes: int):
        regression = MultivariatePolynomialRegression(no_arguments, degree, no_classes)
        super().__init__(regression)


class RecurrentClassification(RegressionClassification):
    def __init__(self, no_classes: int, addends: Sequence[Callable[[Sequence[float]], float]], addends_memory: Sequence[Callable[[Sequence[float]], float]], no_memories: int = 1):
        regression = MultivariateRecurrentRegression(no_classes, addends, addends_memory, resolution_memory=no_memories, error_memory=RegressionClassification.error_class)
        super().__init__(regression)


class RecurrentPolynomialClassification(RecurrentClassification):
    def __init__(self, no_arguments: int, degree: int, no_classes: int, no_memories: int = 1):
        addends_basic = MultiplePolynomialRegression.polynomial_addends(no_arguments + no_memories, degree)
        addends_memory = MultiplePolynomialRegression.polynomial_addends(no_arguments + no_memories, degree)
        super().__init__(no_classes, addends_basic, addends_memory, no_memories=no_memories)
