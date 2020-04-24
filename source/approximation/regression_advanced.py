import random
from typing import Sequence, Callable, List, Any, Tuple, Dict, Optional

from source.approximation.abstract import ApproximationProbabilistic, INPUT_VALUE, OUTPUT_VALUE, Approximation
from source.approximation.regression import RegressionMultivariate, RegressionMultiple, RegressionMultiplePolynomial, RegressionMultivariatePolynomial
from source.tools.functions import z_score_normalized_generator


class RegressionMultivariateRecurrent(RegressionMultivariate):
    def __init__(self,
                 no_outputs: int,
                 addends: Sequence[Callable[[Sequence[float]], float]], addends_memory: Sequence[Callable[[Sequence[float]], float]],
                 resolution_memory: int = 1,
                 error_memory: Callable[[Sequence[float], Sequence[float]], float] = RegressionMultivariate.error_distance
                 ):
        super().__init__(no_outputs, addends)
        self.regressions_memory = tuple(RegressionMultiple(addends_memory) for _ in range(resolution_memory))
        self.values_memory = [0. for _ in range(resolution_memory)]
        self.error_memory = error_memory
        self.last_input = None
        self.normalization = z_score_normalized_generator()
        next(self.normalization)

    def _get_memory(self, in_values: Sequence[float]) -> List[float]:
        return [each_memory.output(in_values) for each_memory in self.regressions_memory]

    def get_state(self) -> Any:
        return self.regressions_memory, self.values_memory

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
                # todo: fit with last target as input, because that's what functional approximations cannot do
                each_memory.fit(self.last_input, each_value, drag)  # smaller, fixed drag?

        input_contextualized = tuple(in_value) + tuple(self.values_memory)
        super().fit(input_contextualized, target_value, drag)

    def get_parameters(self) -> Sequence[float]:
        return tuple(super().get_parameters()) + tuple(x for each_memory in self.regressions_memory for x in each_memory.get_parameters())


class RegressionMultivariatePolynomialRecurrent(RegressionMultivariateRecurrent):
    def __init__(self,
                 no_arguments: int, degree: int, no_outputs: int,
                 resolution_memory: int = 1,
                 error_memory: Callable[[Sequence[float], Sequence[float]], float] = RegressionMultivariate.error_distance
                 ):
        addends_basic = RegressionMultiplePolynomial.polynomial_addends(no_arguments + resolution_memory, degree)
        addends_memory = RegressionMultiplePolynomial.polynomial_addends(no_arguments + resolution_memory, degree)
        super().__init__(no_outputs, addends_basic, addends_memory, resolution_memory=resolution_memory, error_memory=error_memory)


class RegressionMultivariatePolynomialFailure(RegressionMultivariatePolynomial):
    def __init__(self,
                 no_arguments: int,
                 degree: int,
                 no_outputs: int,
                 error_tolerance: float,
                 error_context: Callable[[Sequence[float], Sequence[float]], float] = RegressionMultivariate.error_distance,
                 ):
        super().__init__(no_arguments + 1, degree, no_outputs)
        self.approximation_context = None
        self.no_arguments_context = no_arguments + 1
        self.degree = degree
        self.error_context = error_context
        self.context = 0.

        assert 1. >= error_tolerance >= 0.
        self.error_tolerance = error_tolerance

    def get_state(self) -> Any:
        return self.approximation_context, self.context

    def _optimize_context(self, input_values: Sequence[float], target_values: Sequence[float]) -> Tuple[float, float]:
        error_minimal = -1.
        context_best = None
        # todo: gradient optimization
        for i in range(100):
            context_this = i / 100.
            output_values = super().output(tuple(input_values) + (context_this,))
            e = self.error_context(output_values, target_values)
            if e < error_minimal or context_best is None:
                error_minimal = e
                context_best = context_this

        return context_best, error_minimal

    def fit(self, in_value: Sequence[float], target_value: Sequence[float], drag: int):
        output_values = super().output(tuple(in_value) + (self.context,))
        e = self.error_context(output_values, target_value)

        if self.error_tolerance < e:
            # print("standard context wrong")
            if self.approximation_context is None:
                self.approximation_context = RegressionMultivariatePolynomial(self.no_arguments_context, self.degree, 1)

            context_new, = self.approximation_context.output(tuple(in_value) + (self.context,))
            output_values = super().output(tuple(in_value) + (context_new,))
            e = self.error_context(output_values, target_value)
            drag_context = drag

            if self.error_tolerance < e:
                # print("next context wrong")
                context_new, e = self._optimize_context(in_value, target_value)
                # drag_context = 1

                if self.error_tolerance < e:
                    # print("all contexts wrong")
                    context_new = random.random()
                    drag_context = 1

            self.approximation_context.fit(tuple(in_value) + (self.context,), (context_new,), drag_context)
            self.context = context_new

        super().fit(tuple(in_value) + (self.context,), target_value, drag)

    def output(self, in_value: Sequence[float]) -> Sequence[float]:
        return super().output(tuple(in_value) + (self.context,))


class RegressionMultivariatePolynomialProbabilistic(RegressionMultivariatePolynomial, ApproximationProbabilistic[Sequence[float], Sequence[float]]):
    def __init__(self, no_arguments: int, degree: int, no_outputs: int):
        super().__init__(no_arguments, degree, no_outputs)
        # self.normalize = z_score_normalized_generator()
        # next(self.normalize)

    def get_probability(self, input_value: INPUT_VALUE, output_value: OUTPUT_VALUE) -> float:
        expected_value = self.output(input_value)
        error = RegressionMultivariate.error_distance(expected_value, output_value)
        return 1. / (1. + error)
        # error_normalized = self.normalize.send(error)
        # return 1. - error_normalized


ARGUMENTS = Sequence[float]
FUNCTION_UNI = Callable[[ARGUMENTS], float]

PARAMETERS = Sequence[float]
SHAPE_UNI = Callable[[ARGUMENTS, PARAMETERS], float]


class Shape:
    def __init__(self, shape: Callable[[ARGUMENTS, PARAMETERS], float], no_arguments: int, no_parameters: int):
        self._shape = shape
        self.no_arguments = no_arguments
        self.no_parameters = no_parameters

    def c(self, arguments: ARGUMENTS, parameters: PARAMETERS) -> float:
        assert len(arguments) == self.no_arguments
        assert len(parameters) == self.no_parameters
        return self._shape(arguments, parameters)


class GradientDescent(Approximation[Sequence[float], float]):
    def __init__(self, shape: Shape, derivative: Optional[Shape] = None, difference_gradient: float = .00001, learning_rate: float = .1):
        self.shape = shape
        self.derivative = derivative
        self.difference_gradient = difference_gradient
        self.learning_rate = learning_rate
        self.parameters = [0.] * self.shape.no_parameters

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> Approximation:
        pass

    def to_dict(self) -> Dict[str, Any]:
        pass

    @staticmethod
    def gradient(function: FUNCTION_UNI, arguments: Sequence[float], difference: float) -> Sequence[float]:
        difference_half = difference / 2.
        g = tuple(
            (
                    function(tuple(_p + float(_i == i) * difference_half for _i, _p in enumerate(arguments))) -
                    function(tuple(_p - float(_i == i) * difference_half for _i, _p in enumerate(arguments)))
            ) / difference
            for i, p in enumerate(arguments)
        )
        return g

    @staticmethod
    def error(function: FUNCTION_UNI, arguments: Sequence[float], target: float):
        output = function(arguments)
        return (target - output) ** 2.

    def output(self, in_value: INPUT_VALUE) -> OUTPUT_VALUE:
        return self.shape.c(in_value, self.parameters)

    def fit(self, in_value: INPUT_VALUE, target_value: OUTPUT_VALUE, drag: int):
        if self.derivative is None:
            def parameters_to_error(parameters: Sequence[float]) -> float:
                output_value = self.shape.c(in_value, parameters)
                return (output_value - target_value) ** 2.

            # replace by adam optimizer?
            # https://gluon.mxnet.io/chapter06_optimization/adam-scratch.html
            # add momentum? (https://moodle2.cs.huji.ac.il/nu15/pluginfile.php/316969/mod_resource/content/1/adam_pres.pdf)

            step = GradientDescent.gradient(parameters_to_error, self.parameters, self.difference_gradient)

        else:
            step = self.derivative.c(in_value, self.parameters)
            raise NotImplementedError("doesn't work. needs derivative of error function, not model function.")

        for i, d in enumerate(step):
            self.parameters[i] -= self.learning_rate * d


class GradientDescentMultivariate(Approximation[Sequence[float], Sequence[float]]):
    @staticmethod
    def from_dict(d: Dict[str, Any]) -> Approximation:
        pass

    def to_dict(self) -> Dict[str, Any]:
        pass

    def __init__(self, shapes: Sequence[Shape], derivatives: Optional[Sequence[Shape]] = None, difference_gradient: float = .00001, learning_rate: float = .1):
        if derivatives is None:
            self.gradient_descents = tuple(
                GradientDescent(each_shape, difference_gradient=difference_gradient, learning_rate=learning_rate)
                for each_shape in shapes
            )

        else:
            assert len(derivatives) == len(shapes)
            self.gradient_descents = tuple(
                GradientDescent(each_shape, derivative=each_derivative, difference_gradient=difference_gradient, learning_rate=learning_rate)
                for each_shape, each_derivative in zip(shapes, derivatives)
            )

        self.difference_gradient = difference_gradient
        self.learning_rate = learning_rate

    def output(self, in_value: INPUT_VALUE) -> OUTPUT_VALUE:
        return tuple(each_descent.output(in_value) for each_descent in self.gradient_descents)

    def fit(self, in_value: INPUT_VALUE, target_value: OUTPUT_VALUE, drag: int):
        for each_descent, each_target in zip(self.gradient_descents, target_value):
            each_descent.fit(in_value, each_target, drag)

