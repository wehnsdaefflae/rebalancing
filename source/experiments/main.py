import random

from source.approximation.regression import MultivariatePolynomialRegression, MultivariatePolynomialRecurrentRegression, MultivariatePolynomialFailureRegression
from source.experiments.tasks.speculation import ExperimentMarket, TraderFrequency

# from source.experiments.tasks.trigonometry import ExperimentTrigonometry, SineToCosine
from source.tools.functions import get_pairs_from_filesystem

"""
experiments provides snapshots,
approximations take input and target examples,
applications translate snapshots to examples

"""


def speculation():
    random.seed(45447)

    no_assets_market = 10
    pairs = get_pairs_from_filesystem()
    pairs = random.sample(pairs, no_assets_market)

    fee = .1 / 100.
    certainty = 3. / no_assets_market
    approximations = (
        MultivariatePolynomialRegression(no_assets_market, 2, no_assets_market),
        MultivariatePolynomialRecurrentRegression(no_assets_market, 2, no_assets_market),
        MultivariatePolynomialFailureRegression(no_assets_market, 2, no_assets_market, .5),

    )
    applications = (
        # TraderApproximation("square", approximations[0], no_assets_market, fee, certainty=certainty),
        TraderFrequency("freq 1", no_assets_market, fee, certainty, length_history=1, pseudo_count=100),
        TraderFrequency("freq 2", no_assets_market, fee, certainty, length_history=2, pseudo_count=100),
        TraderFrequency("freq 3", no_assets_market, fee, certainty, length_history=3, pseudo_count=100),
        TraderFrequency("freq 4", no_assets_market, fee, certainty, length_history=4, pseudo_count=100),
        #TraderApproximation("square rec", approximations[1], no_assets_market, fee, certainty=certainty),
        #TraderApproximation("square fail", approximations[2], no_assets_market, fee, certainty=certainty),
        #Balancing("balancing", no_assets_market, 60 * 24, fee),
        #TraderDistribution("distribution", no_assets_market, fee)
    )

    m = ExperimentMarket(applications, pairs, fee)  # , delay=60 * 24)
    m.start()


def trigonometry():
    approximation = MultivariatePolynomialFailureRegression(1, 4, 1, .5)
    # approximation = MultivariatePolynomialRecurrentRegression(1, 4, 1)
    # approximation = MultivariatePolynomialRegression(1, 4, 1)
    #application = SineToCosine("non functional approximation", approximation)
    #t = ExperimentTrigonometry(application)
    #t.start()


# todo: implement reinforcement learning
# todo: separate investor application and base class (determine portfolio ratio externally)

# todo: make all applications persistable
# todo: compare greedy with dynamic programming (no approximation! but "strategy" path: Sequence[int] )
# todo: reinforcement discrete action approximation
# todo: failure regression test
# todo: failure regression equidistant sampling


def main():
    speculation()
    # trigonometry()


if __name__ == "__main__":
    main()
