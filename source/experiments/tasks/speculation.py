import datetime
import random
from typing import Sequence

from source.approximation.abstract import Approximation
from source.data.abstract import SNAPSHOT, STREAM_SNAPSHOTS, EXAMPLE
from source.data.generators.snapshots_binance import rates_binance_generator
from source.experiments.tasks.abstract import Application, Experiment
from source.strategies.infer_investment_path.optimal_trading_alternative import get_pairs_from_filesystem

from source.tools.functions import ratio_generator_multiple, index_max
from source.tools.moving_graph import MovingGraph
from source.tools.timer import Timer


class Balancing(Application):
    def __init__(self, name: str, no_assets: int, minutes_to_balance: int, fee: float):
        self.name = name
        self.no_assets = no_assets
        assert minutes_to_balance >= 0
        self.minutes_to_balance = minutes_to_balance
        self.after_fee = 1. - fee

        self.timestamp_balancing_last = -1

        self.ratio_generator = ratio_generator_multiple(no_assets)
        next(self.ratio_generator)

        self.amounts = [1. / no_assets for _ in range(no_assets)]
        self.trades = 0

    def __str__(self) -> str:
        return self.name

    def _rebalance(self):
        print("rebalancing...")
        s = self._total_amount()
        amount_each = s / self.no_assets
        for i in range(self.no_assets):
            self.amounts[i] = amount_each * self.after_fee
        self.trades += self.no_assets

    def _total_amount(self) -> float:
        return sum(self.amounts)

    def _make_example(self, snapshot: SNAPSHOT) -> EXAMPLE:
        timestamp = Application.get_timestamp(snapshot)
        rates = Application.get_rates(snapshot)
        return timestamp, rates, tuple(-1. for _ in range(self.no_assets))

    def _cycle(self, example: EXAMPLE, act: bool) -> Sequence[float]:
        timestamp, rates, _ = example

        ratios = self.ratio_generator.send(rates)
        if ratios is not None:
            if act and 0 < self.minutes_to_balance and (self.timestamp_balancing_last < 0 or (timestamp - self.timestamp_balancing_last) // 60000 >= self.minutes_to_balance):
                self._rebalance()
                self.timestamp_balancing_last = timestamp

            for i, each_ratio in enumerate(ratios):
                self.amounts[i] *= each_ratio

        return 1., self._total_amount()


class Investor(Application):
    def __init__(self, name: str, approximation: Approximation[Sequence[float]], no_assets: int, fees: float, certainty: float = 1.):
        self.name = name
        self.approximation = approximation
        self.asset_current = -1
        self.amount_current = 1.
        self.after_fee = 1. - fees
        self.certainty = certainty

        self.ratio_generator = ratio_generator_multiple(no_assets)
        next(self.ratio_generator)

        self.error = 1.
        self.ratios_last = None
        self.iteration = 0
        self.trades = 0

    def __str__(self) -> str:
        return self.name

    def _make_example(self, snapshot: SNAPSHOT) -> EXAMPLE:
        timestamp = Application.get_timestamp(snapshot)
        rates = Application.get_rates(snapshot)

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

    def _invest(self, asset: int):
        self.amount_current *= self.after_fee
        self.asset_current = asset
        self.trades += 1

    def _update_error(self, ratios: Sequence[float]):
        asset_target_last, _ = index_max(ratios)
        self.error = float(self.asset_current != asset_target_last)

    def _cycle(self, example: EXAMPLE, act: bool) -> Sequence[float]:
        _, ratios_last, ratios = example

        self._update_error(ratios)

        output_value = self.approximation.output(ratios)
        asset_output, amount_output = index_max(output_value)
        if act and asset_output != self.asset_current and self.certainty / self.after_fee < amount_output:
            self._invest(asset_output)

        self._learn(ratios_last, ratios)

        self.amount_current *= ratios[self.asset_current]
        self.iteration += 1

        return self.error, self.amount_current


class ExperimentMarket(Experiment):
    def __init__(self, investors: Sequence[Investor], no_assets: int, delay: int = 0, visualize: bool = True):
        super().__init__(investors, delay)
        self.graph = None
        if visualize:
            names_graphs_amounts = [f"{str(each_approximation):s} amount" for each_approximation in investors]
            names_graphs_errors = [f"{str(each_approximation):s} error" for each_approximation in investors]
            name_market_average = ["market"]
            self.graph = MovingGraph("amount", names_graphs_amounts + name_market_average, "error", names_graphs_errors, 20, limits_secondary=(-.1, 1.1))
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
                print(f"no. trades: {each_investor.trades: 5d} for {str(each_investor):s}")
                each_investor.trades = 0
