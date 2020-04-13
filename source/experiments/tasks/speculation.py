import datetime
import random
from typing import Sequence

from source.approximation.abstract import Approximation
from source.data.abstract import SNAPSHOT, STREAM_SNAPSHOTS, INPUT_VALUE, TARGET_VALUE, EXAMPLE
from source.data.generators.snapshots_binance import rates_binance_generator
from source.experiments.tasks.abstract import Application, Experiment

from source.tools.functions import generate_ratios_send, index_max, get_pairs_from_filesystem, smear
from source.tools.moving_graph import MovingGraph
from source.tools.timer import Timer


class Investor(Application):
    def __init__(self, name: str, no_assets: int, fee: float):
        super().__init__(name)
        self.trades = 0
        self.no_assets = no_assets
        self.after_fee = 1. - fee

    def _learn(self, input_value: INPUT_VALUE, target_value: TARGET_VALUE):
        raise NotImplementedError()

    def learn(self, input_value: INPUT_VALUE, target_value: TARGET_VALUE):
        assert all(x >= 0. for x in input_value)
        self._learn(input_value, target_value)

    def _act(self, input_value: INPUT_VALUE) -> TARGET_VALUE:
        raise NotImplementedError()

    def act(self, input_value: INPUT_VALUE) -> TARGET_VALUE:
        assert all(x >= 0. for x in input_value)
        return self._act(input_value)


class Balancing(Investor):
    def __init__(self, name: str, no_assets: int, iterations_to_balance: int, fee: float):
        super().__init__(name, no_assets, fee)
        assert iterations_to_balance >= 0
        self.iterations_to_balance = iterations_to_balance
        self.iterations_since_balance = 0
        self.rates = tuple(1. for _ in range(no_assets))
        self.amount_assets = tuple(0. for _ in range(no_assets))

    def _rebalance(self) -> TARGET_VALUE:
        value_each = 1. / self.no_assets
        return tuple(value_each for _ in range(self.no_assets))

    def _learn(self, input_value: INPUT_VALUE, target_value: TARGET_VALUE):
        pass

    def _act(self, input_value: INPUT_VALUE) -> TARGET_VALUE:
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

    def _get_target_asset(self, ratios: Sequence[float]) -> int:
        # asset_target, ratio_best = max(enumerate(ratios), key=lambda x: x[1] * self.after_fee ** int(x[0] != self.asset_current))
        asset_target, ratio_best = max(enumerate(ratios), key=lambda x: x[1])
        if 1.01 < ratio_best:
            return asset_target
        return -1

    def _learn(self, input_value: INPUT_VALUE, target_value: TARGET_VALUE):
        ratios = tuple(r / r_l for r, r_l in zip(target_value, input_value))
        asset_target_last = self._get_target_asset(ratios)
        self._update_distributions(asset_target_last)

    def _act(self, input_value: INPUT_VALUE) -> TARGET_VALUE:
        asset_output, difference = index_max(f_m - f_c for f_m, f_c in zip(self.frequencies_model, self.frequencies_current))
        return tuple(float(i == asset_output) for i in range(self.no_assets))


class TraderApproximation(Investor):
    def __init__(self, name: str, approximation: Approximation[Sequence[float]], no_assets: int, fee: float, certainty: float = 1.):
        super().__init__(name, no_assets, fee)
        self.approximation = approximation
        self.certainty = certainty
        self.ratios = tuple(1. for _ in range(no_assets))
        self.generate_ratios = generate_ratios_send(no_assets)
        self.iterations = 0

    def _learn(self, input_value: INPUT_VALUE, target_value: TARGET_VALUE):
        self.approximation.fit(input_value, target_value, self.iterations)

    def _act(self, input_value: INPUT_VALUE) -> TARGET_VALUE:
        output_value = self.approximation.output(input_value)
        asset_output, amount_output = index_max(output_value)
        if self.certainty / self.after_fee < amount_output:
            return tuple(float(asset_output == i) for i in range(self.no_assets))
        return tuple(-1. for _ in range(self.no_assets))


class TraderFrequency(Investor):
    pass


class ExperimentMarket(Experiment):
    @staticmethod
    def get_rates(snapshot: SNAPSHOT) -> Sequence[float]:
        rates = tuple(
            snapshot[x]
            for x in sorted(snapshot.keys())
            if x.startswith("rate_")
        )
        return rates

    @staticmethod
    def get_timestamp(snapshot: SNAPSHOT) -> int:
        return snapshot["close_time"]

    def __init__(self, investors: Sequence[Investor], no_assets: int, fee: float, asset_initial: int = 0, delay: int = 0, visualize: bool = True):
        super().__init__(investors, delay)
        self.after_fee = 1. - fee
        self.asset_initial = asset_initial

        self.no_assets = no_assets

        self.amount_market = 0.

        self.amounts_assets = tuple([0. for _ in range(no_assets)] for _ in investors)
        self.values_last = [1. for _ in investors]
        self.no_trades = [0 for _ in investors]

        self.rates_last = None
        self.rates = None

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
                tuple(f"{str(each_application):s}" for each_application in investors) + ("market", "minimum", "maximum"),
                False,
                None,
            )

            self.graph = MovingGraph((subplot_amounts, subplot_ratios), 100)

    def _snapshots(self) -> STREAM_SNAPSHOTS:
        pairs = get_pairs_from_filesystem()
        pairs = random.sample(pairs, self.no_assets)

        time_range = 1532491200000, 1577836856000
        interval_minutes = 1

        generator_snapshots = rates_binance_generator(pairs, timestamp_range=time_range, interval_minutes=interval_minutes)
        yield from generator_snapshots

    def _pre_process(self, snapshot: SNAPSHOT):
        self.rates = ExperimentMarket.get_rates(snapshot)
        if 0 < self.iteration:
            return
        self.amount_market = self.no_assets / sum(self.rates)
        for each_amounts in self.amounts_assets:
            each_amounts[self.asset_initial] = 1. / self.rates[self.asset_initial]

    def _get_example(self, snapshot: SNAPSHOT) -> EXAMPLE:
        return self.rates_last, self.rates

    def _perform(self, index_application: int, distribution_value_target: TARGET_VALUE):
        if any(x < 0. for x in distribution_value_target):
            return
        amounts_assets = self.amounts_assets[index_application]
        self.no_trades[index_application] += self.no_assets - amounts_assets.count(0.)
        self.__redistribute(index_application, distribution_value_target)

    def _post_process(self, snapshot: SNAPSHOT):
        timestamp = ExperimentMarket.get_timestamp(snapshot)
        dt = datetime.datetime.utcfromtimestamp(timestamp // 1000)

        # value points
        values = tuple(self.__evaluate(i) for i in range(len(self.applications)))
        points_values = {f"{str(each_application):s}": v for each_application, v in zip(self.applications, values)}
        rate_market = sum(self.rates) / self.no_assets
        value_market = rate_market * self.amount_market
        points_values["market"] = value_market

        # quality points
        if self.rates_last is None:
            ratio_market = tuple(1. for _ in range(self.no_assets))
        else:
            ratio_market = tuple(r / r_l for r, r_l in zip(self.rates, self.rates_last))

        points_quality = {f"{str(each_application):s}": v / v_l for each_application, v, v_l in zip(self.applications, values, self.values_last)}
        points_quality["market"] = sum(ratio_market) / self.no_assets
        points_quality["minimum"] = min(ratio_market)
        points_quality["maximum"] = max(ratio_market)

        self.graph.add_snapshot(dt, (points_values, points_quality))

        self.rates_last = self.rates
        self.values_last = values

        if Timer.time_passed(1000):
            for index_investor, each_investor in enumerate(self.applications):
                print(f"no. trades: {self.no_trades[index_investor]: 5d} for {str(each_investor):s}")
                self.no_trades[index_investor] = 0

    def __redistribute(self, index_investor: int, distribution: TARGET_VALUE):
        _s = sum(distribution)
        distribution_normalized = tuple(x / _s for x in distribution)
        value_total = self.__evaluate(index_investor) * self.after_fee
        value_distributed = tuple(x * value_total for x in distribution_normalized)
        self.amounts_assets = tuple(d / r for d, r in zip(value_distributed, self.rates))

    def __evaluate(self, index_investor: int) -> float:
        amount_assets = self.amounts_assets[index_investor]
        return sum(r * amount_assets[i] for i, r in enumerate(self.rates))
