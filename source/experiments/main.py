import random

from source.approximation.regression import MultivariatePolynomialRegression, MultivariatePolynomialRecurrentRegression, MultivariatePolynomialFailureRegression
from source.experiments.tasks.speculation import Trader, ExperimentMarket, Balancing
from source.experiments.tasks.trigonometry import ExperimentTrigonometry, SineToCosine

"""
experiments provides snapshots,
approximations take input and target examples,
applications translate snapshots to examples

"""


def speculation():
    random.seed(45454547)

    no_assets_market = 8
    fee = .1
    certainty = .901
    approximations = (
        MultivariatePolynomialRegression(no_assets_market, 2, no_assets_market),
        MultivariatePolynomialRecurrentRegression(no_assets_market, 2, no_assets_market),
        MultivariatePolynomialFailureRegression(no_assets_market, 2, no_assets_market, .5),

    )
    applications = (
        Trader("square", approximations[0], no_assets_market, fee, certainty=certainty),
        Trader("square rec", approximations[1], no_assets_market, fee, certainty=certainty),
        Trader("square fail", approximations[2], no_assets_market, fee, certainty=certainty),
        Balancing("balancing", no_assets_market, 60 * 24, fee),
    )

    m = ExperimentMarket(applications, no_assets_market, delay=60 * 24)
    m.start()


def trigonometry():
    approximation = MultivariatePolynomialFailureRegression(1, 4, 1, .5)
    # approximation = MultivariatePolynomialRecurrentRegression(1, 4, 1)
    # approximation = MultivariatePolynomialRegression(1, 4, 1)
    application = SineToCosine("non functional approximation", approximation)
    t = ExperimentTrigonometry(application)
    t.start()


# todo: make all applications persistable
# todo: compare greedy with dynamic programming (no approximation! but "strategy" path: Sequence[int] )
# todo: reinforcement discrete action approximation
# todo: failure regression test
# todo: failure regression equidistant sampling


def main():
    # speculation()
    trigonometry()


if __name__ == "__main__":
    main()
