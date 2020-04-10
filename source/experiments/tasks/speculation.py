import datetime
import random
from typing import Sequence

from matplotlib import pyplot

from source.approximation.abstract import Approximation
from source.data.abstract import SNAPSHOT, STREAM_SNAPSHOTS, EXAMPLE
from source.data.generators.snapshots_binance import rates_binance_generator
from source.experiments.tasks.abstract import Application, Experiment, RESULT

from source.tools.functions import generate_ratios_send, index_max, get_pairs_from_filesystem, smear
from source.tools.moving_graph import MovingGraph
from source.tools.timer import Timer


class Investor(Application):
    def __init__(self, name: str, asset_initial: int = 0):
        super().__init__(name)
        self.trades = 0
        self.asset_current = asset_initial

    @staticmethod
    def is_valid_snapshot(snapshot: SNAPSHOT) -> bool:
        return "close_time" in snapshot and any(x.startswith("rate_") for x in snapshot)

    @staticmethod
    def is_valid_result(result: RESULT) -> bool:
        return "error" in result and "amount" in result and "ratio_portfolio" in result

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

    def cycle(self, snapshot: SNAPSHOT, act: bool = True) -> RESULT:
        raise NotImplementedError()


class Balancing(Investor):
    def __init__(self, name: str, no_assets: int, minutes_to_balance: int, fee: float):
        super().__init__(name)
        self.no_assets = no_assets
        assert minutes_to_balance >= 0
        self.minutes_to_balance = minutes_to_balance
        self.after_fee = 1. - fee

        self.timestamp_balancing_last = -1

        self.ratio_generator = generate_ratios_send(no_assets)
        next(self.ratio_generator)

        self.amounts = [float(i == self.asset_current) for i in range(no_assets)]

    def _rebalance(self):
        print("rebalancing...")
        s = sum(self.amounts)
        amount_each = s / self.no_assets
        for i in range(self.no_assets):
            self.amounts[i] = amount_each * self.after_fee
        self.increment_trades(by=self.no_assets)

    def cycle(self, snapshot: SNAPSHOT, act: bool = True) -> RESULT:
        timestamp = Investor.get_timestamp(snapshot)
        rates = Investor.get_rates(snapshot)

        ratio_portfolio = 1.
        error = 1.
        ratios = self.ratio_generator.send(rates)
        if ratios is not None:
            ratio_market = sum(ratios) / self.no_assets
            ratio_portfolio = sum(r * a for r, a in zip(ratios, self.amounts)) / sum(self.amounts)
            error = float(ratio_market >= ratio_portfolio)

            if act and 0 < self.minutes_to_balance and (self.timestamp_balancing_last < 0 or (timestamp - self.timestamp_balancing_last) // 60000 >= self.minutes_to_balance):
                self._rebalance()
                self.timestamp_balancing_last = timestamp

            for i, each_ratio in enumerate(ratios):
                self.amounts[i] *= each_ratio

        return {
            "error": error,
            "amount": sum(self.amounts),
            "ratio_portfolio": ratio_portfolio,
        }


class TraderDistribution(Investor):
    def __init__(self, name: str, no_assets: int, fee: float, trail: int = 100):
        super().__init__(name)
        self.no_assets = no_assets
        self.after_fee = 1. - fee
        self.frequencies_model = [1. for _ in range(no_assets)]
        self.frequencies_current = [0. for _ in range(no_assets)]
        self.trail = trail
        self.iterations = 0
        self.generate_ratios = generate_ratios_send(no_assets)
        next(self.generate_ratios)

        self.amount = 1.

    def _update_distributions(self, asset_target: int):
        for i, v in enumerate(self.frequencies_model):
            self.frequencies_model[i] = smear(v, float(i == asset_target), self.trail)

        for i, v in enumerate(self.frequencies_current):
            self.frequencies_current[i] = smear(v, float(i == asset_target), self.trail // 2)

    def _get_target(self, ratios: Sequence[float]) -> int:
        # asset_target, ratio_best = max(enumerate(ratios), key=lambda x: x[1] * self.after_fee ** int(x[0] != self.asset_current))
        asset_target, ratio_best = max(enumerate(ratios), key=lambda x: x[1])
        if 1.01 < ratio_best:
            return asset_target
        return -1

    def cycle(self, snapshot: SNAPSHOT, act: bool = True) -> RESULT:
        rates = Investor.get_rates(snapshot)
        ratios = self.generate_ratios.send(rates)

        ratio_portfolio = 1.
        error = 1.
        if ratios is not None:
            ratio_market = sum(ratios) / self.no_assets
            ratio_portfolio = ratios[self.asset_current]
            error = float(ratio_market >= ratio_portfolio)

            asset_target_last = self._get_target(ratios)

            self._update_distributions(asset_target_last)
            asset_output, difference = index_max(f_m - f_c for f_m, f_c in zip(self.frequencies_model, self.frequencies_current))

            if asset_output != self.asset_current and act:
                self.amount *= self.after_fee
                self.asset_current = asset_output
                self.increment_trades()

            self.amount *= ratios[self.asset_current]

        self.iterations += 1
        return {
            "error": error,
            "amount": self.amount,
            "ratio_portfolio": ratio_portfolio,
        }


class InvestorSupervised(Investor):
    def _make_example(self, snapshot: SNAPSHOT) -> EXAMPLE:
        raise NotImplementedError()

    def _cycle(self, example: EXAMPLE, act: bool) -> RESULT:
        # includes testing, learning, and applying
        raise NotImplementedError()

    def cycle(self, snapshot: SNAPSHOT, act: bool = True) -> RESULT:
        example = self._make_example(snapshot)
        return self._cycle(example, act)


class TraderApproximation(InvestorSupervised):
    def __init__(self, name: str, approximation: Approximation[Sequence[float]], no_assets: int, fees: float, certainty: float = 1.):
        super().__init__(name)
        self.approximation = approximation
        self.no_assets = no_assets
        self.amount_current = 1.
        self.after_fee = 1. - fees
        self.certainty = certainty

        self.ratio_generator = generate_ratios_send(no_assets)
        next(self.ratio_generator)

        self.ratios_last = None
        self.iteration = 0

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

    def _cycle(self, example: EXAMPLE, act: bool) -> RESULT:
        _, ratios_last, ratios = example

        ratio_market = sum(ratios) / self.no_assets
        ratio_portfolio = ratios[self.asset_current]
        error = float(ratio_market >= ratio_portfolio)

        self._learn(ratios_last, ratios)

        output_value = self.approximation.output(ratios)
        asset_output, amount_output = index_max(output_value)
        if act and asset_output != self.asset_current and self.certainty / self.after_fee < amount_output:
            self._invest(asset_output)

        self.amount_current *= ratios[self.asset_current]

        self.iteration += 1

        return {
            "error": error,
            "amount": self.amount_current,
            "ratio_portfolio": ratio_portfolio,
        }


class ExperimentMarket(Experiment):
    def __init__(self, investors: Sequence[Investor], no_assets: int, delay: int = 0, visualize: bool = True):
        super().__init__(investors, delay)
        self.graphs = []
        self.generate_ratio = generate_ratios_send(no_assets)
        next(self.generate_ratio)

        if visualize:
            fig, (axes_correct_incorrect, axes_error_amount) = pyplot.subplots(nrows=2, ncols=1, sharex="all")

            names_graphs_ratio_best_correct = [f"{str(each_application):s} best correct" for each_application in investors]
            names_graphs_ratio_best_incorrect = [f"{str(each_application):s} best incorrect" for each_application in investors]
            self.graphs.append(
                MovingGraph(
                    axes_correct_incorrect,
                    "best ratio", names_graphs_ratio_best_correct + names_graphs_ratio_best_incorrect,
                    "best ratio incorrect", [], 20,
                    moving_average_primary=False, moving_average_secondary=False,
                )
            )

            names_errors = [f"{str(each_application):s} error" for each_application in investors]
            names_amounts = [f"{str(each_application):s} amount" for each_application in investors] + ["market"]
            self.graphs.append(
                MovingGraph(
                    axes_error_amount,
                    "error", names_errors,
                    "amount", names_amounts, 20,
                    moving_average_primary=False, moving_average_secondary=True,
                    limits_primary=(-.1, 1.1),
                )
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
        timestamp = Investor.get_timestamp(snapshot)
        dt = datetime.datetime.utcfromtimestamp(timestamp // 1000)

        # graph 0
        rates = Investor.get_rates(snapshot)
        ratio = self.generate_ratio.send(rates)
        if ratio is None:
            ratio = tuple(1. for _ in range(self.no_assets))
        ratio_best = max(ratio)
        ratios_best_correct = tuple(ratio_best if each_result["error"] < .5 else each_result["ratio_portfolio"] for each_result in results)
        ratios_best_incorrect = tuple(ratio_best if each_result["error"] >= .5 else each_result["ratio_portfolio"] for each_result in results)
        self.graphs[0].add_snapshot(dt, ratios_best_correct + ratios_best_incorrect, [])

        # graph 1
        rate_average = sum(rates) / self.no_assets
        if self.initial < 0.:
            self.initial = 1. / rate_average

        errors = tuple(each_result["error"] for each_result in results)
        amounts = tuple(each_result["amount"] for each_result in results)
        self.graphs[1].add_snapshot(dt, errors, amounts + (self.initial * rate_average,))

        if Timer.time_passed(1000):
            for each_investor in self.investors:    # type: Investor
                print(f"no. trades: {each_investor.get_trades(): 5d} for {str(each_investor):s}")
                each_investor.reset_trades()
