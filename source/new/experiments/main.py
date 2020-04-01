from source.new.experiments.applications.speculation import Investor, ExperimentMarket
from source.new.learning.regression import MultivariatePolynomialRegression

if __name__ == "__main__":
    # todo: compare greedy with dynamic programming (no learning!)
    # todo: reinforcement learning
    # todo: test failure regression
    # todo: normalize output?
    # todo: equidistant sampling

    no_assets_market = 3
    fee = .1
    approximations = MultivariatePolynomialRegression(no_assets_market, 3, no_assets_market), MultivariatePolynomialRegression(no_assets_market, 2, no_assets_market)
    investors = tuple(Investor(each_approximation, no_assets_market, fee) for each_approximation in approximations)
    m = ExperimentMarket(investors, no_assets_market)
    m.start()
