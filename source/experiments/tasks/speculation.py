import datetime
import random
from typing import Sequence

from source.approximation.abstract import Approximation
from source.data.abstract import SNAPSHOT, STREAM_SNAPSHOTS, EXAMPLE
from source.data.generators.snapshots_binance import rates_binance_generator
from source.experiments.tasks.abstract import Application, Experiment, RESULT

from source.tools.functions import ratio_generator_multiple, index_max, get_pairs_from_filesystem
from source.tools.moving_graph import MovingGraph
from source.tools.timer import Timer


class Investor(Application):
    def __init__(self):
        self.trades = 0

    @staticmethod
    def get_timestamp(snapshot: SNAPSHOT) -> int:
        return snapshot["close_time"]

    @staticmethod
    def get_rates(snapshot: SNAPSHOT) -> Sequence[float]:
        rates = tuple(
            snapshot[x]
            for x in sorted(snapshot.keys())
            if x.startswith("rate_")
        )
        return rates

    def get_trades(self) -> int:
        return self.trades

    def reset_trades(self):
        self.trades = 0

    def increment_trades(self, by: int = 1):
        self.trades = self.trades + by

    def __str__(self) -> str:
        raise NotImplementedError()

    @staticmethod
    def is_valid_snapshot(snapshot: SNAPSHOT) -> bool:
        return "close_time" in snapshot and any(x.startswith("rate_") for x in snapshot)

    @staticmethod
    def is_valid_result(result: RESULT) -> bool:
        return "error" in result and "amount" in result

    def _make_example(self, snapshot: SNAPSHOT) -> EXAMPLE:
        raise NotImplementedError()

    def _cycle(self, example: EXAMPLE, act: bool) -> RESULT:
        raise NotImplementedError()


class Balancing(Investor):
    def __init__(self, name: str, no_assets: int, minutes_to_balance: int, fee: float):
        super().__init__()
        self.name = name
        self.no_assets = no_assets
        assert minutes_to_balance >= 0
        self.minutes_to_balance = minutes_to_balance
        self.after_fee = 1. - fee

        self.timestamp_balancing_last = -1

        self.ratio_generator = ratio_generator_multiple(no_assets)
        next(self.ratio_generator)

        self.amounts = [1. / no_assets for _ in range(no_assets)]

    def __str__(self) -> str:
        return self.name

    def _rebalance(self):
        print("rebalancing...")
        s = self._total_amount()
        amount_each = s / self.no_assets
        for i in range(self.no_assets):
            self.amounts[i] = amount_each * self.after_fee
        self.increment_trades(by=self.no_assets)

    def _total_amount(self) -> float:
        return sum(self.amounts)

    def _make_example(self, snapshot: SNAPSHOT) -> EXAMPLE:
        timestamp = Investor.get_timestamp(snapshot)
        rates = Investor.get_rates(snapshot)
        return timestamp, rates, tuple(-1. for _ in range(self.no_assets))

    def _cycle(self, example: EXAMPLE, act: bool) -> RESULT:
        timestamp, rates, _ = example

        ratios = self.ratio_generator.send(rates)
        if ratios is not None:
            if act and 0 < self.minutes_to_balance and (self.timestamp_balancing_last < 0 or (timestamp - self.timestamp_balancing_last) // 60000 >= self.minutes_to_balance):
                self._rebalance()
                self.timestamp_balancing_last = timestamp

            for i, each_ratio in enumerate(ratios):
                self.amounts[i] *= each_ratio

        return {"error": 1., "amount": self._total_amount()}


class Trader(Investor):
    def __init__(self, name: str, approximation: Approximation[Sequence[float]], no_assets: int, fees: float, certainty: float = 1.):
        super().__init__()
        self.name = name
        self.approximation = approximation
        self.asset_current = -1
        self.amount_current = 1.
        self.after_fee = 1. - fees
        self.certainty = certainty

        self.ratio_generator = ratio_generator_multiple(no_assets)
        next(self.ratio_generator)

        self.ratios_last = None
        self.iteration = 0

    def __str__(self) -> str:
        return self.name

    def _make_example(self, snapshot: SNAPSHOT) -> EXAMPLE:
        timestamp = Investor.get_timestamp(snapshot)
        rates = Investor.get_rates(snapshot)

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
        self.increment_trades()

    def _get_error(self, ratios: Sequence[float]) -> float:
        asset_target_last, _ = index_max(ratios)
        return float(self.asset_current != asset_target_last)

    def _cycle(self, example: EXAMPLE, act: bool) -> RESULT:
        _, ratios_last, ratios = example

        output_value = self.approximation.output(ratios)
        asset_output, amount_output = index_max(output_value)
        if act and asset_output != self.asset_current and self.certainty / self.after_fee < amount_output:
            self._invest(asset_output)

        self._learn(ratios_last, ratios)
        error = self._get_error(ratios)

        self.amount_current *= ratios[self.asset_current]
        self.iteration += 1

        return {"error": error, "amount": self.amount_current}


class ExperimentMarket(Experiment):
    def __init__(self, investors: Sequence[Investor], no_assets: int, delay: int = 0, visualize: bool = True):
        super().__init__(investors, delay)
        self.graph = None
        if visualize:
            names_graphs_amounts = [f"{str(each_approximation):s} amount" for each_approximation in investors]
            names_graphs_errors = [f"{str(each_approximation):s} error" for each_approximation in investors]
            name_market_average = ["market"]
            self.graph = MovingGraph(
                "amount", names_graphs_amounts + name_market_average, "error", names_graphs_errors, 20,
                moving_average_primary=True, moving_average_secondary=False, limits_secondary=(-.1, 1.1)
            )
        self.no_assets = no_assets
        self.initial = -1.

    def _snapshots(self) -> STREAM_SNAPSHOTS:
        pairs = get_pairs_from_filesystem()
        pairs = random.sample(pairs, self.no_assets)

        time_range = 1532491200000, 1577836856000
        interval_minutes = 1

        generator_snapshots = rates_binance_generator(pairs, timestamp_range=time_range, interval_minutes=interval_minutes)
        yield from generator_snapshots

    def _postprocess_results(self, snapshot: SNAPSHOT, results: Sequence[RESULT]):
        if self.graph is not None:
            timestamp = Investor.get_timestamp(snapshot)
            rates = Investor.get_rates(snapshot)

            rate_average = sum(rates) / self.no_assets
            if self.initial < 0.:
                self.initial = 1. / rate_average

            errors = tuple(each_result["error"] for each_result in results)
            amounts = tuple(each_result["amount"] for each_result in results)
            dt = datetime.datetime.utcfromtimestamp(timestamp // 1000)
            self.graph.add_snapshot(dt, amounts + (self.initial * rate_average,), errors)

        if Timer.time_passed(1000):
            for each_investor in self.investors:    # type: Investor
                print(f"no. trades: {each_investor.get_trades(): 5d} for {str(each_investor):s}")
                each_investor.reset_trades()
