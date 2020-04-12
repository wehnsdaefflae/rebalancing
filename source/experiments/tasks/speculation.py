import datetime
import random
from typing import Sequence

from source.approximation.abstract import Approximation
from source.data.abstract import SNAPSHOT, STREAM_SNAPSHOTS, INPUT_VALUE, TARGET_VALUE
from source.data.generators.snapshots_binance import rates_binance_generator
from source.experiments.tasks.abstract import Application, Experiment, ACTION

from source.tools.functions import generate_ratios_send, index_max, get_pairs_from_filesystem, smear
from source.tools.moving_graph import MovingGraph
from source.tools.timer import Timer


DISTRIBUTION_VALUE_ASSET = Sequence[float]


class Investor(Application):
    def __init__(self, name: str, no_assets: int, fee: float):
        super().__init__(name)
        self.trades = 0
        self.no_assets = no_assets
        self.after_fee = 1. - fee

    def learn(self, snapshot: SNAPSHOT):
        raise NotImplementedError()

    def act(self) -> DISTRIBUTION_VALUE_ASSET:
        raise NotImplementedError()

    @staticmethod
    def get_rates(snapshot: SNAPSHOT) -> Sequence[float]:
        rates = tuple(
            snapshot[x]
            for x in sorted(snapshot.keys())
            if x.startswith("rate_")
        )
        return rates

    @staticmethod
    def get_ratios(snapshot: SNAPSHOT) -> Sequence[float]:
        ratios = tuple(
            snapshot[x]
            for x in sorted(snapshot.keys())
            if x.startswith("ratio_")
        )
        return ratios
    
    @staticmethod
    def get_amounts(snapshot: SNAPSHOT) -> Sequence[float]:
        amounts = tuple(
            snapshot[x]
            for x in sorted(snapshot.keys())
            if x.startswith("amount_")
        )
        return amounts


class Balancing(Investor):
    def __init__(self, name: str, no_assets: int, iterations_to_balance: int, fee: float):
        super().__init__(name, no_assets, fee)
        assert iterations_to_balance >= 0
        self.iterations_to_balance = iterations_to_balance
        self.iterations_since_balance = 0
        self.rates = tuple(1. for _ in range(no_assets))
        self.amount_assets = tuple(0. for _ in range(no_assets))

    def _rebalance(self) -> DISTRIBUTION_VALUE_ASSET:
        print("rebalancing...")
        value_after_fee = sum(r * a for r, a in zip(self.rates, self.amount_assets)) * self.after_fee
        value_individual = value_after_fee / self.no_assets
        return tuple(value_individual for _ in range(self.no_assets))

    def learn(self, snapshot: SNAPSHOT):
        self.rates = Investor.get_rates(snapshot)
        self.amount_assets = Investor.get_amounts(snapshot)

    def act(self) -> DISTRIBUTION_VALUE_ASSET:
        self.iterations_since_balance += 1
        if self.iterations_since_balance % self.iterations_to_balance == 0:
            self.iterations_since_balance = 0
            return self._rebalance()
        return tuple(-1. for _ in range(self.no_assets))


class TraderDistribution(Investor):
    def __init__(self, name: str, no_assets: int, fee: float, trail: int = 100):
        super().__init__(name, no_assets, fee)
        self.frequencies_model = [1. for _ in range(no_assets)]
        self.frequencies_current = [0. for _ in range(no_assets)]
        self.trail = trail

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

    def learn(self, snapshot: SNAPSHOT):
        ratios = Investor.get_ratios(snapshot)
        asset_target_last = self._get_target(ratios)
        self._update_distributions(asset_target_last)

    def act(self) -> DISTRIBUTION_VALUE_ASSET:
        asset_output, difference = index_max(f_m - f_c for f_m, f_c in zip(self.frequencies_model, self.frequencies_current))
        return tuple(float(i == asset_output) for i in range(self.no_assets))


class InvestorSupervised(Investor):
    def __init__(self, name: str, no_assets: int, fee: float):
        super().__init__(name, no_assets, fee)
        self.input = tuple(1. for _ in range(no_assets))

    def _get_input(self, snapshot: SNAPSHOT) -> INPUT_VALUE:
        raise NotImplementedError()

    def _get_target_last(self, snapshot: SNAPSHOT) -> TARGET_VALUE:
        raise NotImplementedError()

    def _test(self, input_value: INPUT_VALUE, target_value: TARGET_VALUE):
        raise NotImplementedError()

    def _learn(self, input_value: INPUT_VALUE, target_value: TARGET_VALUE):
        raise NotImplementedError()

    def _act(self, input_values: INPUT_VALUE) -> DISTRIBUTION_VALUE_ASSET:
        raise NotImplementedError()

    def learn(self, snapshot: SNAPSHOT):
        target_last = self._get_target_last(snapshot)
        self._test(self.input, target_last)
        self._learn(self.input, target_last)
        self.input = self._get_input(snapshot)

    def act(self) -> DISTRIBUTION_VALUE_ASSET:
        return self._act(self.input)


class TraderApproximation(InvestorSupervised):
    def __init__(self, name: str, approximation: Approximation[Sequence[float]], no_assets: int, fee: float, certainty: float = 1.):
        super().__init__(name, no_assets, fee)
        self.approximation = approximation
        self.certainty = certainty
        self.ratios = tuple(1. for _ in range(no_assets))
        self.iterations = 0

    def _get_input(self, snapshot: SNAPSHOT) -> INPUT_VALUE:
        return self.ratios

    def _get_target_last(self, snapshot: SNAPSHOT) -> TARGET_VALUE:
        self.ratios = Investor.get_ratios(snapshot) 
        return self.ratios

    def _test(self, input_value: INPUT_VALUE, target_value: TARGET_VALUE):
        pass

    def _learn(self, input_value: INPUT_VALUE, target_value: TARGET_VALUE):
        self.approximation.fit(input_value, target_value, self.iterations)

    def _act(self, input_values: INPUT_VALUE) -> DISTRIBUTION_VALUE_ASSET:
        output_value = self.approximation.output(input_values)
        asset_output, amount_output = index_max(output_value)
        if self.certainty / self.after_fee < amount_output:
            return tuple(float(asset_output == i) for i in range(self.no_assets))
        return tuple(-1. for _ in range(self.no_assets))


class ExperimentMarket(Experiment):
    def __init__(self, investors: Sequence[Investor], no_assets: int, delay: int = 0, visualize: bool = True):
        super().__init__(investors, delay)
        self.values = [1. for _ in investors]
        self.qualities = [1. for _ in investors]
        self.amounts = tuple([0. for _ in range(no_assets)] for _ in investors)

        self.graph = None
        if visualize:
            subplot_amounts = (
                "value",
                tuple(f"{str(each_application):s}" for each_application in investors) + ("market", ),
                True,
                None,
            )

            subplot_ratios = (
                "quality",
                tuple(f"{str(each_application):s}" for each_application in investors),  # + ("market", "minimum", "maximum"),
                False,
                None,
            )

            self.graph = MovingGraph((subplot_amounts, subplot_ratios), 100)

        self.generate_ratio = generate_ratios_send(no_assets)
        next(self.generate_ratio)

        self.no_assets = no_assets
        self.amount_initial_market = 1.

    def _redistribute(self, index_investor: int, distribution: DISTRIBUTION_VALUE_ASSET):
        value_total = sum
    def _perform(self, index_application: int, action: ACTION):
        investor = self.applications[index_application]
        distribution_value_assets = investor.act()

    def _snapshots(self) -> STREAM_SNAPSHOTS:
        pairs = get_pairs_from_filesystem()
        pairs = random.sample(pairs, self.no_assets)

        time_range = 1532491200000, 1577836856000
        interval_minutes = 1

        generator_snapshots = rates_binance_generator(pairs, timestamp_range=time_range, interval_minutes=interval_minutes)
        yield from generator_snapshots

    def get_value(self, amount_assets: Sequence[float], rates: Sequence[float]) -> float:
        return sum(r * a for r, a in zip(rates, amount_assets))

    def update_measures(self):
        each_investor: Investor
        values_new = tuple(each_investor.get_value() for each_investor in self.applications)

        for i, (v_n, v_o) in enumerate(zip(values_new, self.values)):
            self.qualities[i] = v_n / v_o

        for i, v in enumerate(values_new):
            self.values[i] = v

    def _process_results(self, snapshot: SNAPSHOT, results: Sequence[RESULT]):
        assert len(results) == len(self.applications)
        timestamp = Investor.get_timestamp(snapshot)
        dt = datetime.datetime.utcfromtimestamp(timestamp // 1000)

        rates = Investor.get_rates(snapshot)
        ratio = self.generate_ratio.send(rates)
        if ratio is None:
            ratio = tuple(1. for _ in range(self.no_assets))
        ratio_market = sum(ratio) / self.no_assets

        # amount points
        points_amounts = {f"{str(each_application):s}": each_result["value"] for each_application, each_result in zip(self.applications, results)}
        points_amounts["market"] = self.amount_initial_market
        self.amount_initial_market *= ratio_market

        # ratio points
        points_ratios = {f"{str(each_application):s}": each_result["quality"] / ratio_market for each_application, each_result in zip(self.applications, results)}
        points_ratios["market"] = ratio_market
        points_ratios["minimum"] = min(ratio)
        points_ratios["maximum"] = max(ratio)

        self.graph.add_snapshot(dt, (points_amounts, points_ratios))

        if Timer.time_passed(1000):
            for each_investor in self.applications:    # type: Investor
                print(f"no. trades: {each_investor.get_trades(): 5d} for {str(each_investor):s}")
                each_investor.reset_trades()
