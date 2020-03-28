import math
import random
import time
from typing import Iterable, Tuple

from source.new.learning.classification import PolynomialClassification
from source.new.learning.regression import MultivariatePolynomialRecurrentRegression
from source.new.learning.tools import MovingGraph, z_score_multiple_normalized_generator


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


def z_score_test():
    no_values = 10
    fg = MovingGraph(no_values, 10)

    z = z_score_multiple_normalized_generator(no_values)
    next(z)

    while True:
        values = tuple(random.random() * (random.random() * 10.) - (random.random() * 10.) for _ in range(no_values))
        value_zs = z.send(values)
        fg.add_snapshot(value_zs)
        fg.draw()
        time.sleep(.1)


def main():
    # classification_test()
    # regression_test()
    z_score_test()
