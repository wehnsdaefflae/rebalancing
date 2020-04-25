import random

from source.approximation.abstract_advanced import ApproximationSemioticModel
from source.approximation.regression import RegressionMultiplePolynomial, RegressionMultivariatePolynomial
from source.approximation.regression_advanced import RegressionMultivariatePolynomialProbabilistic, Shape, GradientDescent, RegressionMultivariateRecurrentPolynomial
from source.experiments.tasks.speculation import ExperimentMarket, TraderHistoric

from source.experiments.tasks.debugging import TransformRational, ExperimentTimeseries, ExperimentStatic, TransformHistoric
from source.tools.functions import get_pairs_from_filesystem

"""
experiments provides snapshots,
approximations take input and target examples,
applications translate snapshots to examples

"""


def speculation():
    random.seed(454546547)

    no_assets_market = 10
    pairs = get_pairs_from_filesystem()
    pairs = random.sample(pairs, no_assets_market)

    fee = .1 / 100.
    certainty = 1.1  # 1. / (1. - fee)
    length_history = 10

    # factory = lambda: RegressionMultivariatePolynomialProbabilistic(no_assets_market, 2, no_assets_market)
    # approximation = ApproximationSemioticModel(.9, factory, max_approximations=50)
    # application = TraderApproximation("semiotic", approximation, no_assets_market, certainty=certainty)

    approximation = RegressionMultiplePolynomial(length_history, 2)
    application = TraderHistoric("historic", no_assets_market, approximation, length_history, certainty=certainty, one_hot=True)
    # application = Balancing("balancing", no_assets_market, 60*24)

    m = ExperimentMarket(application, pairs, fee)
    m.start()


def debug_dynamic():
    len_history = 2
    approximation = RegressionMultivariatePolynomial(len_history, 1, 1)
    application = TransformHistoric(approximation.__class__.__name__, approximation, len_history)

    factory = lambda: RegressionMultivariatePolynomialProbabilistic(1, 2, 1)
    approximation = ApproximationSemioticModel(.9, factory)
    application = TransformRational("semiotic", approximation)

    approximation = RegressionMultivariateRecurrentPolynomial(1, 1, 1)
    application = TransformRational("recurrent", approximation)

    t = ExperimentTimeseries(application, ExperimentTimeseries.f_square())
    t.start()


def debug_static():
    shape = Shape(lambda a, p: p[0]*a[0]**0. + p[1]*a[0]**1. + p[2]*a[0]**2. + p[3]*a[0]**3., 1, 4)
    approximation = GradientDescent(shape, difference_gradient=.001, learning_rate=.000001)

    # approximation = RegressionMultivariatePolynomial(1, 5, 1)

    application = TransformRational(approximation.__class__.__name__, approximation)
    t = ExperimentStatic(application)
    t.start()


# todo: recurrent regression test
# todo: gradient descent implementation

# todo: test trigonometric addends

# todo: implement reinforcement learning

# todo: make all applications persistable
# todo: reinforcement discrete action approximation


# todo: debug randomly change function
# todo: implement recurrency
#   todo: implement sgd
# todo: implement neural nets

# todo: non-one-hotified output informative. use max index only if o[i] / sum(max(_o, 0.) for _o in o) > x * 1. / no_assets
# todo: make historic inherit with super()
# inheritance patterns: _not implemented, super(), encapsulation

def main():
    # debug_static()
    debug_dynamic()
    # speculation()


if __name__ == "__main__":
    main()
