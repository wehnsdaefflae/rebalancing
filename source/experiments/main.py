import random

from source.approximation.regression import MultivariatePolynomialRegression, MultivariatePolynomialRecurrentRegression, MultivariatePolynomialFailureRegression
from source.experiments.tasks.speculation import TraderApproximation, ExperimentMarket, Balancing, TraderDistribution, TraderFrequency

# from source.experiments.tasks.trigonometry import ExperimentTrigonometry, SineToCosine

"""
experiments provides snapshots,
approximations take input and target examples,
applications translate snapshots to examples

"""


def speculation():
    random.seed(45454547)

    no_assets_market = 10
    fee = .001
    certainty = 1.
    approximations = (
        MultivariatePolynomialRegression(no_assets_market, 2, no_assets_market),
        MultivariatePolynomialRecurrentRegression(no_assets_market, 2, no_assets_market),
        MultivariatePolynomialFailureRegression(no_assets_market, 2, no_assets_market, .5),

    )
    applications = (
        TraderApproximation("square", approximations[0], no_assets_market, fee, certainty=certainty),
        TraderFrequency("freq 1", no_assets_market, fee, length_history=1),
        TraderFrequency("freq 2", no_assets_market, fee, length_history=2),
        TraderFrequency("freq 3", no_assets_market, fee, length_history=3),
        TraderFrequency("freq 4", no_assets_market, fee, length_history=4),
        #TraderApproximation("square rec", approximations[1], no_assets_market, fee, certainty=certainty),
        #TraderApproximation("square fail", approximations[2], no_assets_market, fee, certainty=certainty),
        #Balancing("balancing", no_assets_market, 60 * 24, fee),
        #TraderDistribution("distribution", no_assets_market, fee)
    )

    m = ExperimentMarket(applications, no_assets_market, fee)  # , delay=60 * 24)
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
