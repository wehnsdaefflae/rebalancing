import random

from source.approximation.abstract_advanced import ApproximationSemioticModel
from source.approximation.regression import RegressionMultivariatePolynomial, RegressionMultiplePolynomial
from source.approximation.regression_advanced import RegressionMultivariatePolynomialProbabilistic
from source.experiments.tasks.speculation import ExperimentMarket, TraderFrequency, TraderApproximation, Balancing, TraderHistoric

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

    # test only one
    # done: plot portfolio
    # todo: non-one-hotified output informative. use max index only if o[i] / sum(max(_o, 0.) for _o in o) > x * 1. / no_assets
    # todo: make historic inherit with super()
    # inheritance patterns: _not implemented, super(), encapsulation
    # todo: implement recurrency
    #   todo: implement sgd
    # todo: implement neural nets

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
    approximation = RegressionMultiplePolynomial(len_history, 1)
    applications = [TransformHistoric(approximation.__class__.__name__, approximation, len_history)]
    t = ExperimentTimeseries(applications, ExperimentTimeseries.nf_trigonometry())
    t.start()


def debug_static():
    approximation = RegressionMultivariatePolynomial(1, 10, 1)
    application = TransformRational(approximation.__class__.__name__, approximation)
    t = ExperimentStatic(application)
    t.start()


def debug_nonfunctional():
    # ExperimentTimeseries.f_square()
    # ExperimentTimeseries.nf_triangle()
    # ExperimentTimeseries.nf_square()
    # ExperimentTimeseries.nf_trigonometry()

    factory = lambda: RegressionMultivariatePolynomialProbabilistic(1, 3, 1)
    approximation = ApproximationSemioticModel(.9, factory)

    #approximation = RegressionMultivariatePolynomialFailure(1, 1, 1, .2)
    applications = [TransformRational(approximation.__class__.__name__, approximation)]
    t = ExperimentTimeseries(applications, ExperimentTimeseries.nf_trigonometry())
    t.start()


# todo: failure regression test
# todo: failure regression equidistant sampling

# todo: implement reinforcement learning

# todo: make all applications persistable
# todo: reinforcement discrete action approximation


def main():
    speculation()
    # debug_dynamic()
    # debug_static()
    # debug_nonfunctional()


if __name__ == "__main__":
    main()
