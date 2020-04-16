import random

from source.approximation.regression import MultivariatePolynomialRegression, MultivariatePolynomialRecurrentRegression, MultivariatePolynomialFailureRegression
from source.experiments.tasks.speculation import ExperimentMarket, TraderFrequency, TraderApproximation, Balancing

from source.experiments.tasks.debugging import TransformRational, ExperimentTimeseries, ExperimentStatic
from source.tools.functions import get_pairs_from_filesystem

"""
experiments provides snapshots,
approximations take input and target examples,
applications translate snapshots to examples

"""


def speculation():
    random.seed(4548547)

    no_assets_market = 10
    pairs = get_pairs_from_filesystem()
    pairs = random.sample(pairs, no_assets_market)

    fee = .1 / 100.
    certainty = 1. / (1. - fee)
    approximations = (
        MultivariatePolynomialRegression(no_assets_market, 2, no_assets_market),
        MultivariatePolynomialRecurrentRegression(no_assets_market, 2, no_assets_market),
        MultivariatePolynomialFailureRegression(no_assets_market, 2, no_assets_market, .5),

    )
    applications = (
        TraderApproximation("square", approximations[0], no_assets_market, certainty=certainty),
        # TraderFrequency("freq 1", no_assets_market, certainty, length_history=1, inertia=100),
        # TraderFrequency("freq 2", no_assets_market, certainty, length_history=2, inertia=100),
        # TraderFrequency("freq 3", no_assets_market, certainty, length_history=3, inertia=100),
        TraderFrequency("freq 4", no_assets_market, certainty, length_history=4, inertia=100),
        # TraderApproximation("square rec", approximations[1], no_assets_market, certainty=certainty),
        # TraderApproximation("square fail", approximations[2], no_assets_market, fee, certainty=certainty),
        Balancing("balancing", no_assets_market, 60),
        # TraderDistribution("distribution", no_assets_market, fee),
    )

    m = ExperimentMarket(applications, pairs, fee)  # , delay=60 * 24)
    m.start()


def debug_dynamic():
    approximation = MultivariatePolynomialRegression(1, 1, 1)
    applications = [TransformRational(approximation.__class__.__name__, approximation)]
    t = ExperimentTimeseries(applications, ExperimentTimeseries.nf_trigonometry())
    t.start()


def debug_static():
    approximation = MultivariatePolynomialRegression(1, 10, 1)
    application = TransformRational(approximation.__class__.__name__, approximation)
    t = ExperimentStatic(application)
    t.start()


def debug_nonfunctional():
    # ExperimentTimeseries.f_square()
    # ExperimentTimeseries.nf_triangle()
    # ExperimentTimeseries.nf_square()
    # ExperimentTimeseries.nf_trigonometry()

    approximation = MultivariatePolynomialFailureRegression(1, 3, 1, .2)
    applications = [TransformRational(approximation.__class__.__name__, approximation)]
    t = ExperimentTimeseries(applications, ExperimentTimeseries.nf_trigonometry())
    t.start()


# todo: failure regression test
# todo: failure regression equidistant sampling

# todo: implement reinforcement learning

# todo: make all applications persistable
# todo: reinforcement discrete action approximation


def main():
    # speculation()
    # debug_dynamic()
    # debug_static()
    debug_nonfunctional()


if __name__ == "__main__":
    main()
