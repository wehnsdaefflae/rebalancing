from source.approximation.regression import MultivariatePolynomialRegression, MultivariatePolynomialRecurrentRegression
from source.experiments.tasks.speculation import Investor, ExperimentMarket


if __name__ == "__main__":
    # todo: make all applications persistable
    # todo: compare greedy with dynamic programming (no approximation!)
    # todo: reinforcement approximation
    # todo: test failure regression
    # todo: normalize output?
    # todo: equidistant sampling

    no_assets_market = 3
    fee = .1
    approximations = (
        MultivariatePolynomialRegression(no_assets_market, 2, no_assets_market),
        MultivariatePolynomialRegression(no_assets_market, 3, no_assets_market),
        MultivariatePolynomialRecurrentRegression(no_assets_market, 2, no_assets_market),
        MultivariatePolynomialRecurrentRegression(no_assets_market, 3, no_assets_market),
    )
    applications = (
        Investor("square", approximations[0], no_assets_market, fee),
        Investor("cubic", approximations[1], no_assets_market, fee),
        Investor("square rec", approximations[2], no_assets_market, fee),
        Investor("cubic rec", approximations[3], no_assets_market, fee),
    )

    m = ExperimentMarket(applications, no_assets_market)
    m.start()
