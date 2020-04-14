import datetime
import random
from typing import Sequence, Tuple

from source.approximation.abstract import Approximation
from source.data.abstract import SNAPSHOT, STREAM_SNAPSHOTS, INPUT_VALUE, TARGET_VALUE, EXAMPLE
from source.data.generators.snapshots_binance import rates_binance_generator
from source.experiments.tasks.abstract import Application, Experiment

from source.tools.functions import generate_ratios_send, index_max, get_pairs_from_filesystem, smear, z_score_normalized_generator
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
        target_asset, _ = index_max(target_value)
        target_amplified = tuple(float(i == target_asset) for i in range(self.no_assets))
        self.approximation.fit(input_value, target_amplified, self.iterations)

    def _act(self, input_value: INPUT_VALUE) -> TARGET_VALUE:
        output_value = self.approximation.output(input_value)
        asset_output, amount_output = index_max(output_value)
        if self.certainty < amount_output:
            return tuple(float(asset_output == i) for i in range(self.no_assets))
        return tuple(-1. for _ in range(self.no_assets))


class TraderFrequency(Investor):
    def __init__(self, name: str, no_assets: int, fee: float, certainty_min: float, length_history: int = 1, inertia=100):
        super().__init__(name, no_assets, fee)
        self.frequencies = dict()
        self.certainty_min = certainty_min / no_assets
        self.length_history = length_history
        self.history = [-1 for _ in range(length_history)]
        self.rate_last = None
        self.inertia = inertia

    def _update_frequency(self, history: Tuple[int], ratio: Sequence[float]):
        sub_dict = self.frequencies.get(history)
        if sub_dict is None:
            sub_dict = dict()
            self.frequencies[history] = sub_dict

        # to maintain a moving distribution
        for i, each_ratio in enumerate(ratio):
            sub_dict[i] = smear(sub_dict.get(i, 0.), each_ratio, self.inertia)

    def _get_target(self, history: Tuple[int]) -> Tuple[int, float]:
        sub_dict = self.frequencies.get(history)
        if sub_dict is None:
            return -1, 0.
        asset_target, asset_frequency = max(sub_dict.items(), key=lambda x: x[1])
        return asset_target, (asset_frequency + 1.) / (sum(sub_dict.values()) + self.no_assets)

    def _get_ratio(self, rate: INPUT_VALUE) -> Sequence[float]:
        if self.rate_last is None:
            ratio = tuple(1. for _ in range(self.no_assets))
        else:
            ratio = tuple(1. if 0. >= r_l else r / r_l for r, r_l in zip(rate, self.rate_last))
        return ratio

    def _learn(self, input_value: INPUT_VALUE, target_value: TARGET_VALUE):
        ratio_last = self._get_ratio(input_value)

        asset_best_prev, _ = index_max(ratio_last)
        self.history.append(asset_best_prev)
        del(self.history[:-self.length_history])

        ratio = tuple(r / r_l for r, r_l in zip(target_value, input_value))

        self._update_frequency(tuple(self.history), ratio)

        self.rate_last = input_value

    def _act(self, input_value: INPUT_VALUE) -> TARGET_VALUE:
        ratio = self._get_ratio(input_value)

        asset_best_prev, _ = index_max(ratio)
        history_new = self.history[1:] + [asset_best_prev]
        asset_target, certainty = self._get_target(tuple(history_new))
        # certainty_normalized = self.normalization.send(certainty)
        if asset_target < 0 or certainty < self.certainty_min:
            return tuple(-1. for _ in range(self.no_assets))

        return tuple(float(i == asset_target) for i in range(self.no_assets))


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

    def __init__(self,
                 investors: Sequence[Investor],
                 pairs: Sequence[Tuple[str, str]],
                 fee: float,
                 asset_initial: int = 0, delay: int = 0, visualize: bool = True):

        super().__init__(investors, delay)
        self.after_fee = 1. - fee
        self.asset_initial = asset_initial

        self.pairs = pairs
        self.no_assets = len(self.pairs)

        self.amount_market = 0.

        self.amounts_assets = tuple([0. for _ in range(self.no_assets)] for _ in investors)
        self.values_last = [1. for _ in investors]
        self.no_trades = [0 for _ in investors]
        self.has_started = [False for _ in self.applications]

        self.rates_last = None
        self.rates = None

        self.timestamp = -1
        self.timestamp_last = -1

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
                tuple(f"{str(each_application):s}" for each_application in investors) + ("market", ),  # "minimum", "maximum"),
                False,
                None,
            )

            self.graph = MovingGraph((subplot_amounts, subplot_ratios), 50)

    def _snapshots(self) -> STREAM_SNAPSHOTS:
        time_range = 1532491200000, 1577836856000
        interval_minutes = 1

        generator_snapshots = rates_binance_generator(self.pairs, timestamp_range=time_range, interval_minutes=interval_minutes)
        yield from generator_snapshots

    def _pre_process(self, snapshot: SNAPSHOT):
        self.timestamp = ExperimentMarket.get_timestamp(snapshot)
        self.rates = ExperimentMarket.get_rates(snapshot)

    def _skip(self) -> bool:
        if any(0. >= x for x in self.rates):
            print(f"skipping timestamp {self.timestamp:d} due to illegal rates <{', '.join(f'{x:5.2f}' for x in self.rates):s}>.")
            return True

        if 0. >= self.amount_market:
            self.amount_market = self.no_assets / sum(self.rates)

        return False

    def _get_example(self, snapshot: SNAPSHOT) -> EXAMPLE:
        return self.rates_last, self.rates

    def _perform(self, index_application: int, distribution_value_target: TARGET_VALUE):
        if any(x < 0. for x in distribution_value_target):
            return

        amounts_assets = self.amounts_assets[index_application]
        amounts_assets_copy = list(amounts_assets)

        _s = sum(distribution_value_target)
        distribution_normalized = tuple(x / _s for x in distribution_value_target)

        value_total = self.__evaluate(index_application) * self.after_fee
        value_distributed = tuple(x * value_total for x in distribution_normalized)
        for i, (v, r) in enumerate(zip(value_distributed, self.rates)):
            amounts_assets[i] = v / r

        self.no_trades[index_application] += sum(int(v_n != v_o) for v_n, v_o in zip(amounts_assets, amounts_assets_copy))

        if not self.has_started[index_application]:
            self.has_started[index_application] = True

    def _post_process(self, snapshot: SNAPSHOT):
        faulty = any(0. >= x for x in self.rates)

        # value points
        if faulty:
            values = tuple(0. for _ in self.applications)
            value_market = 0.

        else:
            values = tuple(self.__evaluate(i) if self.has_started[i] else 1. for i in range(len(self.applications)))
            rate_market = sum(self.rates) / self.no_assets
            value_market = rate_market * self.amount_market
        points_values = {f"{str(each_application):s}": v for each_application, v in zip(self.applications, values)}
        points_values["market"] = value_market

        # quality points
        if self.rates_last is None or faulty:
            ratio_market = 1.
        else:
            ratio_market = sum(r / r_l for r, r_l in zip(self.rates, self.rates_last)) / self.no_assets

        if faulty:
            points_quality = {f"{str(each_application):s}": 1. for each_application in self.applications}
        else:
            points_ratio = tuple(v / v_l for v, v_l in zip(values, self.values_last))
            points_quality = {
                f"{str(each_application):s}": r / ratio_market if s else 0.
                for s, each_application, r in zip(self.has_started, self.applications, points_ratio)
            }
        points_quality["market"] = 1.

        dt = datetime.datetime.utcfromtimestamp(self.timestamp // 1000)
        self.graph.add_snapshot(dt, (points_values, points_quality))

        if not faulty:
            self.rates_last = self.rates
            self.values_last = values

    def _information_sample(self) -> str:
        info = []
        if self.timestamp_last >= 0:
            interval = self.timestamp - self.timestamp_last
            interval_dt = datetime.timedelta(milliseconds=interval)
            info.append(f"time interval {str(interval_dt):s}")

        for index_investor, each_investor in enumerate(self.applications):
            info.append(f"no. trades: {self.no_trades[index_investor]: 5d} for {str(each_investor):s}")
            self.no_trades[index_investor] = 0

        self.timestamp_last = self.timestamp
        return "\n".join(info)

    def __evaluate(self, index_investor: int) -> float:
        if not self.has_started[index_investor]:
            return 1.

        amount_assets = self.amounts_assets[index_investor]
        return sum(r * amount_assets[i] for i, r in enumerate(self.rates))

    def start(self):
        super().start()
        if self.graph is not None:
            self.graph.show()
