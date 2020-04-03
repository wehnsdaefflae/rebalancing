import datetime
import random
from typing import Sequence, Tuple

from source.approximation.abstract import Approximation
from source.data.abstract import SNAPSHOT, STREAM_SNAPSHOTS, EXAMPLE
from source.data.generators.snapshots_binance import rates_binance_generator
from source.experiments.tasks.abstract import Application, Experiment
from source.strategies.infer_investment_path.optimal_trading_alternative import get_pairs_from_filesystem

from source.tools.functions import ratio_generator_multiple
from source.tools.moving_graph import MovingGraph
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

    def _make_example(self, snapshot: SNAPSHOT) -> EXAMPLE:
        timestamp = snapshot["close_time"]
        rates = tuple(
            snapshot[x]
            for x in sorted(snapshot.keys())
            if x.startswith("rate_")
        )

        target_values = self.ratio_generator.send(rates)
        if target_values is None:
            target_values = tuple(1. for _ in rates)

        if self.ratios_last is None:
            self.ratios_last = tuple(1. for _ in rates)

        input_values = self.ratios_last
        self.ratios_last = target_values

        return timestamp, input_values, target_values

    def _learn(self, input_values: Sequence[float], target_values: Sequence[float]):
        self.approximation.fit(input_values, target_values, self.iteration)

    def _get_asset(self, values: Sequence[float]) -> Tuple[int, float]:
        index_max, value_max = max(enumerate(values), key=lambda x: x[1])
        return index_max, value_max

    def _invest(self, asset: int):
        self.amount_current *= self.after_fee
        self.asset_current = asset
        self.trades += 1

    def _update_error(self, ratios: Sequence[float]):
        asset_target_last, _ = self._get_asset(ratios)
        self.error = float(self.asset_current != asset_target_last)

    def _cycle(self, example: EXAMPLE) -> Sequence[float]:
        _, ratios_last, ratios = example

        self._update_error(ratios)

        output_value = self.approximation.output(ratios)
        asset_output, amount_output = self._get_asset(output_value)
        if asset_output != self.asset_current and 1. / self.after_fee < amount_output:
            self._invest(asset_output)

        self._learn(ratios_last, ratios)

        self.amount_current *= ratios[self.asset_current]
        self.iteration += 1

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
        self.initial = -1.

    def _snapshots(self) -> STREAM_SNAPSHOTS:
        pairs = get_pairs_from_filesystem()
        pairs = random.sample(pairs, self.no_assets)

        time_range = 1532491200000, 1577836856000
        interval_minutes = 1

        generator_snapshots = rates_binance_generator(pairs, timestamp_range=time_range, interval_minutes=interval_minutes)
        yield from generator_snapshots

    def _apply(self, snapshot: SNAPSHOT, results: Sequence[Sequence[float]]):
        timestamp = snapshot["close_time"]
        rates = tuple(
            snapshot[x]
            for x in sorted(snapshot.keys())
            if x.startswith("rate_")
        )

        rate_average = sum(rates) / self.no_assets
        if self.initial < 0.:
            self.initial = 1. / rate_average

        if self.graph is not None:
            errors, amounts = zip(*results)
            dt = datetime.datetime.utcfromtimestamp(timestamp // 1000)
            self.graph.add_snapshot(dt, amounts + (self.initial * rate_average,), errors)

        if Timer.time_passed(1000):
            for each_investor in self.applications:
                print(f"no trades of {str(each_investor):s}: {each_investor.trades}")
                each_investor.trades = 0
