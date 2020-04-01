import datetime
import random
from typing import Sequence, Generator, Tuple, Any

from source.new.data.binance_examples import get_pairs_from_filesystem
from source.new.data.snapshot_generation import merge_generator
from source.new.experiments.tasks.abstract import Application, Experiment

from source.new.experiments.tools.moving_graph import MovingGraph
from source.new.learning.approximation import Approximation
from source.new.learning.regression import MultivariatePolynomialRegression, MultivariatePolynomialRecurrentRegression
from source.new.learning.tools import ratio_generator_multiple
from source.tools.timer import Timer


class Investor(Application):
    def __init__(self, name: str, approximation: Approximation[Sequence[float]], no_assets: int, fees: float):
        self.name = name
        self.approximation = approximation
        self.asset_current = -1
        self.amount_current = 1.
        self.after_fee = 1. - fees

        self.ratio_generator = ratio_generator_multiple(no_assets)
        next(self.ratio_generator)

        self.error = 1.
        self.ratios_last = None
        self.iteration = 0
        self.trades = 0

    def __str__(self) -> str:
        return self.name

    def _learn(self, ratios: Sequence[float]):
        output_value = self.approximation.output(self.ratios_last)
        asset_output, _ = max(enumerate(output_value), key=lambda x: x[1])
        asset_target, _ = max(enumerate(ratios), key=lambda x: x[1])
        self.error = float(asset_output != asset_target)
        self.approximation.fit(self.ratios_last, ratios, self.iteration)
        self.iteration += 1

    def _invest(self, ratios: Sequence[float]):
        output_value = self.approximation.output(ratios)
        asset_output, asset_ratio = max(enumerate(output_value), key=lambda x: x[1])

        if asset_output != self.asset_current and 1. / self.after_fee < asset_ratio:
            self.amount_current *= self.after_fee
            self.asset_current = asset_output
            self.trades += 1

        self.amount_current *= ratios[self.asset_current]

    def cycle(self, _: Sequence[float], rates: Sequence[float]) -> Sequence[float]:
        ratios = self.ratio_generator.send(rates)
        if ratios is not None:
            self._invest(ratios)

            if self.ratios_last is not None:
                self._learn(ratios)

            self.ratios_last = ratios

        return self.error, self.amount_current


class ExperimentMarket(Experiment):
    def __init__(self, investors: Sequence[Investor], no_assets: int, visualize: bool = True):
        super().__int__(investors)
        self.graph = None
        if visualize:
            names_graphs_amounts = [f"{str(each_approximation):s} amount" for each_approximation in investors]
            names_graphs_errors = [f"{str(each_approximation):s} error" for each_approximation in investors]
            name_market_average = ["market"]
            self.graph = MovingGraph("amount", names_graphs_amounts + name_market_average, "error", names_graphs_errors, 20, limits_secondary=(0., 1.))
        self.no_assets = no_assets
        self.market_average = 1.

    def _examples(self) -> Generator[Tuple[Any, Sequence[float], Sequence[float]], None, None]:
        pairs = get_pairs_from_filesystem()
        pairs = random.sample(pairs, self.no_assets)

        keys_rates = tuple("-".join(each_pair) + "_close" for each_pair in pairs)

        time_range = 1532491200000, 1577836856000
        interval_minutes = 1

        generator_snapshots = merge_generator(pairs, timestamp_range=time_range, interval_minutes=interval_minutes)

        ratios_rates = ratio_generator_multiple(self.no_assets)
        next(ratios_rates)

        self.ratio_last = None
        for each_snapshot in generator_snapshots:
            if each_snapshot is None:
                continue

            rates = tuple(each_snapshot[key] for key in keys_rates)
            ratio = ratios_rates.send(rates)
            if self.ratio_last is not None:
                yield each_snapshot["close_time"], self.ratio_last, ratio

            self.ratio_last = ratio

    def _apply(self, key: Any, ratios_last: Sequence[float], ratios_now: Sequence[float], results: Sequence[Sequence[float]]):
        self.market_average *= sum(ratios_now) / self.no_assets
        if self.graph is not None:
            errors, amounts = zip(*results)
            dt = datetime.datetime.utcfromtimestamp(key // 1000)
            self.graph.add_snapshot(dt, amounts + (self.market_average, ), errors)

        if Timer.time_passed(1000):
            for each_investor in self.applications:
                print(f"no trades of {str(each_investor):s}: {each_investor.trades}")
                each_investor.trades = 0


if __name__ == "__main__":
    # todo: compare greedy with dynamic programming (no learning!)
    # todo: reinforcement learning
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
