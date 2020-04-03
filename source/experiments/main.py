from source.approximation.regression import MultivariatePolynomialRegression, MultivariatePolynomialRecurrentRegression, MultivariatePolynomialFailureRegression
from source.experiments.tasks.speculation import Investor, ExperimentMarket, Balancing

"""
experiments provides snapshots,
approximations take input and target examples,
applications translate snapshots to examples

"""

if __name__ == "__main__":
    # todo: make all applications persistable
    # todo: compare greedy with dynamic programming (no approximation! but "strategy" path: Sequence[int] )
    # todo: reinforcement approximation
    # todo: test failure regression
    # todo: normalize output?
    # todo: equidistant sampling

    no_assets_market = 3
    fee = .1
    approximations = (
        MultivariatePolynomialRegression(no_assets_market, 2, no_assets_market),
        #MultivariatePolynomialRegression(no_assets_market, 3, no_assets_market),
        MultivariatePolynomialRecurrentRegression(no_assets_market, 2, no_assets_market),
        #MultivariatePolynomialRecurrentRegression(no_assets_market, 3, no_assets_market),
        MultivariatePolynomialFailureRegression(no_assets_market, 2, no_assets_market, .5),
        #MultivariatePolynomialFailureRegression(no_assets_market, 3, no_assets_market, .5),

    )
    applications = (
        Investor("square", approximations[0], no_assets_market, fee),
        #Investor("cubic", approximations[1], no_assets_market, fee),
        Investor("square rec", approximations[2], no_assets_market, fee),
        #Investor("cubic rec", approximations[3], no_assets_market, fee),
        Investor("square fail", approximations[4], no_assets_market, fee),
        #Investor("cubic fail", approximations[5], no_assets_market, fee),
        Balancing("balancing", no_assets_market, 60 * 24 * 7, fee),
    )

    m = ExperimentMarket(applications, no_assets_market)
    m.start()
