import datetime
from typing import Sequence, Tuple

from source.approximation.abstract import Approximation
from source.approximation.abstract_advanced import ApproximationSemioticModel
from source.approximation.regression import RegressionMultiple
from source.data.abstract import INPUT_VALUE, OUTPUT_VALUE, STATE, EXAMPLE
from source.data.generators.snapshots_binance import rates_binance_generator, get_timestamp, get_rates
from source.experiments.tasks.abstract import Application, Experiment

from source.tools.functions import max_index, smear, normalize
from source.tools.moving_graph import MovingGraph
from source.tools.timer import Timer


class Investor(Application):
    def __init__(self, name: str, no_assets: int):
        super().__init__(name)
        self.trades = 0
        self.no_assets = no_assets

    def _learn(self, rates_last: INPUT_VALUE, rates_now: OUTPUT_VALUE):
        raise NotImplementedError()

    def learn(self, input_value: INPUT_VALUE, target_value: OUTPUT_VALUE):
        if all(x >= 0. for x in input_value) and all(x >= 0. for x in target_value):
            self._learn(input_value, target_value)

    def _act(self, rates_now: INPUT_VALUE) -> OUTPUT_VALUE:
        raise NotImplementedError()

    def act(self, input_value: INPUT_VALUE) -> OUTPUT_VALUE:
        if all(x >= 0. for x in input_value):
            return self._act(input_value)

        return tuple(-1. for _ in range(self.no_assets))

    @staticmethod
    def one_hotify(vector: OUTPUT_VALUE, value_min: float) -> OUTPUT_VALUE:
        index_max, value_max = max(enumerate(vector), key=lambda x: x[1])

        if value_max >= value_min:
            return tuple(float(i == index_max) for i in range(len(vector)))

        return tuple(-1. for _ in vector)


class Balancing(Investor):
    def __init__(self, name: str, no_assets: int, iterations_to_balance: int):
        super().__init__(name, no_assets)
        assert iterations_to_balance >= 0
        self.iterations_to_balance = iterations_to_balance
        self.iterations_since_balance = 0
        self.rates = tuple(1. for _ in range(no_assets))
        self.amount_assets = tuple(0. for _ in range(no_assets))

    def _rebalance(self) -> OUTPUT_VALUE:
        value_each = 1. / self.no_assets
        return tuple(value_each for _ in range(self.no_assets))

    def _learn(self, rates_last: INPUT_VALUE, rates_now: OUTPUT_VALUE):
        pass

    def _act(self, rates_now: INPUT_VALUE) -> OUTPUT_VALUE:
        self.iterations_since_balance += 1
        if self.iterations_since_balance % self.iterations_to_balance == 0:
            self.iterations_since_balance = 0
            return self._rebalance()
        return tuple(-1. for _ in range(self.no_assets))


class InvestorRatio(Investor):
    def __init__(self, name: str, no_assets: int):
        super().__init__(name, no_assets)
        self.rate_last = tuple(-1. for _ in range(no_assets))
        # self.ratio_last = tuple(1. for _ in range(no_assets))

    @staticmethod
    def get_ratio(rates_last: Sequence[float], rates_now: Sequence[float]) -> Sequence[float]:
        return tuple(1. if 0. >= r_l else r / r_l for r, r_l in zip(rates_now, rates_last))

    def _learn(self, ratios_last: INPUT_VALUE, ratios_now: OUTPUT_VALUE):
        raise NotImplementedError()

    def learn(self, rates_last: INPUT_VALUE, rates_now: OUTPUT_VALUE):
        if all(x >= 0. for x in rates_last) and all(x >= 0. for x in rates_now):
            ratio_last = InvestorRatio.get_ratio(self.rate_last, rates_last)
            ratio_this = InvestorRatio.get_ratio(rates_last, rates_now)
            self._learn(ratio_last, ratio_this)

        self.rate_last = rates_last

    def _act(self, ratios_now: INPUT_VALUE) -> OUTPUT_VALUE:
        raise NotImplementedError()

    def act(self, rates_now: INPUT_VALUE) -> OUTPUT_VALUE:
        if all(x >= 0. for x in rates_now):
            ratio_this = InvestorRatio.get_ratio(self.rate_last, rates_now)
            return self._act(ratio_this)
        return tuple(-1. for _ in range(self.no_assets))


class TraderDistribution(InvestorRatio):
    def __init__(self, name: str, no_assets: int, certainty: float = 1., trail: int = 100):
        super().__init__(name, no_assets)
        self.frequencies_model = [1. for _ in range(no_assets)]
        self.frequencies_current = [0. for _ in range(no_assets)]
        self.certainty = certainty
        self.trail = trail

    def _update_distributions(self, asset_target: int):
        for i, v in enumerate(self.frequencies_model):
            self.frequencies_model[i] = smear(v, float(i == asset_target), self.trail)

        for i, v in enumerate(self.frequencies_current):
            self.frequencies_current[i] = smear(v, float(i == asset_target), self.trail // 2)

    def _get_target_asset(self, ratios: Sequence[float]) -> int:
        asset_target, ratio_best = max(enumerate(ratios), key=lambda x: x[1])
        if self.certainty < ratio_best:
            return asset_target
        return -1

    def _learn(self, ratios_last: INPUT_VALUE, ratios_now: OUTPUT_VALUE):
        asset_target_last = self._get_target_asset(ratios_now)
        self._update_distributions(asset_target_last)

    def _act(self, ratios_now: INPUT_VALUE) -> OUTPUT_VALUE:
        # todo: incorporate one_hotify
        asset_output, difference = max_index(f_m - f_c for f_m, f_c in zip(self.frequencies_model, self.frequencies_current))
        return tuple(float(i == asset_output) for i in range(self.no_assets))


class TraderApproximation(InvestorRatio):
    def __init__(self, name: str, approximation: Approximation[Sequence[float], Sequence[float]], no_assets: int, certainty: float = 1.):
        super().__init__(name, no_assets)
        self.approximation = approximation
        self.certainty = certainty
        self.iterations = 0

    def _learn(self, ratios_last: INPUT_VALUE, ratios_now: OUTPUT_VALUE):
        self.approximation.fit(ratios_last, ratios_now, self.iterations)

    def _act(self, ratios_now: INPUT_VALUE) -> OUTPUT_VALUE:
        output_value = self.approximation.output(ratios_now)
        # todo: maybe output distribution, not one hot vector?
        return Investor.one_hotify(output_value, self.certainty)


class TraderFrequency(InvestorRatio):
    def __init__(self, name: str, no_assets: int, certainty_min: float = 1., length_history: int = 1, inertia=100):
        super().__init__(name, no_assets)
        self.frequencies = dict()
        self.certainty_min = certainty_min / no_assets
        self.length_history = length_history
        self.history = [-1 for _ in range(length_history)]
        self.inertia = inertia

    def _update_frequency(self, history: Tuple[int], ratio: Sequence[float]):
        sub_dict = self.frequencies.get(history)
        if sub_dict is None:
            sub_dict = dict()
            self.frequencies[history] = sub_dict

        # maintain a moving distribution (intertia)
        for i, each_ratio in enumerate(ratio):
            sub_dict[i] = smear(sub_dict.get(i, 0.), each_ratio, self.inertia)

    def _get_target(self, history: Tuple[int]) -> Tuple[int, float]:
        sub_dict = self.frequencies.get(history)
        if sub_dict is None:
            return -1, 0.
        items = sub_dict.items()
        asset_target, asset_frequency = max(items, key=lambda x: x[1])
        return asset_target, (asset_frequency + 1.) / (sum(x[1] for x in items) + self.no_assets)

    def _learn(self, ratios_last: INPUT_VALUE, ratios_now: OUTPUT_VALUE):
        asset_best_prev, _ = max_index(ratios_last)
        self.history.append(asset_best_prev)
        del(self.history[:-self.length_history])
        self._update_frequency(tuple(self.history), ratios_now)

    def _act(self, ratios_now: INPUT_VALUE) -> OUTPUT_VALUE:
        asset_best_prev, _ = max_index(ratios_now)
        history_new = self.history[1:] + [asset_best_prev]
        asset_target, certainty = self._get_target(tuple(history_new))

        # todo: maybe output distribution, not one hot vector?

        if asset_target < 0 or certainty < self.certainty_min:
            return tuple(-1. for _ in range(self.no_assets))

        return tuple(float(i == asset_target) for i in range(self.no_assets))

# todo: make one hotify reusable


class TraderHistoric(InvestorRatio):
    # todo: make reusable
    def __init__(self, name: str, no_assets: int, regression: RegressionMultiple, length_history: int, certainty: float = 1., one_hot: bool = True):
        super().__init__(name, no_assets)
        self.regression = regression
        self.certainty = certainty
        self.length_history = length_history
        self.history_ratios = tuple([1. for _ in range(self.length_history)] for _ in range(self.no_assets))
        self.iterations = 0
        self.one_hot = one_hot

    def _learn(self, ratios_last: INPUT_VALUE, ratios_now: OUTPUT_VALUE):
        for each_history, each_input, each_target in zip(self.history_ratios, ratios_last, ratios_now):
            each_history.append(each_input)
            del(each_history[:-self.length_history])
            self.regression.fit(each_history, each_target, self.iterations)

        self.iterations += 1

    def _act(self, ratios_now: INPUT_VALUE) -> OUTPUT_VALUE:
        output_full = tuple(
            self.regression.output(each_history[1:] + [each_ratio])
            for each_history, each_ratio in zip(self.history_ratios, ratios_now)
        )
        if self.one_hot:
            return Investor.one_hotify(output_full, self.certainty)
        return output_full


class ExperimentMarket(Experiment):
    def __init__(self,
                 investors: Sequence[Investor],
                 pairs: Sequence[Tuple[str, str]],
                 fee: float,
                 visualize: bool = True):

        super().__init__(investors)
        self.after_fee = 1. - fee
        self.pairs = pairs
        self.graph = None
        if visualize:
            subplot_amounts = (
                "value",
                tuple(f"{str(each_application):s}" for each_application in investors) + ("market", ),
                "moving",
                None,
                "regular"
            )

            subplot_ratios = (
                "relative growth",
                tuple(f"{str(each_application):s}" for each_application in investors) + ("market", ),  # "minimum", "maximum"),
                "full",
                None,
                "regular"
            )

            subplot_trades = (
                "no. trades",
                tuple(f"{str(each_application):s}" for each_application in investors),
                "accumulate",
                None,
                "step"
            )

            self.graph = MovingGraph((subplot_amounts, subplot_ratios, subplot_trades), 50)

    def _initialize_state(self):
        self.state_experiment.clear()
        self.state_experiment["portfolios"] = tuple([-1. for _ in self.pairs] for _ in self.applications)
        self.state_experiment["no_trades"] = [0. for _ in self.applications]
        self.state_experiment["value_market_last"] = 1.
        self.state_experiment["value_portfolios_last"] = [-1. for _ in self.applications]

    def _states(self) -> STATE:
        time_range = 1532491200000, 1577836856000
        interval_minutes = 1
        yield from rates_binance_generator(self.pairs, timestamp_range=time_range, interval_minutes=interval_minutes)

    def _update_experiment(self, state_environment: STATE):
        self.state_experiment["timestamp"] = get_timestamp(state_environment)
        rates = tuple(-1. if 0. >= r else r for r in get_rates(state_environment))
        rates_last = self.state_experiment.get("rates")
        self.state_experiment["rates"] = rates
        growths = tuple(1. for _ in rates) if rates_last is None else tuple(-1. if 0. >= r_l or 0. >= r else r / r_l for r, r_l in zip(rates, rates_last))
        self.state_experiment["growths"] = growths

    def _get_offset_example(self) -> EXAMPLE:
        rates = self.state_experiment["rates"]
        return self.state_experiment["timestamp"], rates, rates

    def _pre_process(self):
        no_trades = self.state_experiment["no_trades"]
        for i in range(len(no_trades)):
            no_trades[i] = 0

    def _perform(self, index_investor: int, distribution_value_target: OUTPUT_VALUE):
        distribution_normalized = normalize(distribution_value_target)
        if 0. >= max(distribution_normalized):
            return

        rates = self.state_experiment["rates"]
        if any(0. >= x for x in rates):
            print(f"erroneous rates: {str(rates):s}. skipping...")
            return

        portfolios = self.state_experiment["portfolios"]
        no_trades = self.state_experiment["no_trades"]

        portfolio = portfolios[index_investor]
        portfolio_copy = list(portfolio)

        value = self.__evaluate(index_investor) * self.after_fee  # actually just for those assets that do not stay the same
        value_distributed = tuple(x * value for x in distribution_normalized)
        for i, (v, r) in enumerate(zip(value_distributed, rates)):
            portfolio[i] = v / r

        no_trades[index_investor] = sum(int(a_n != a_o) for a_n, a_o in zip(portfolio, portfolio_copy))

    def _post_process(self):
        growths = self.state_experiment["growths"]
        if any(0. >= x for x in growths):
            print(f"erroneous growths: {str(growths):s}. skipping...")
            return

        rates = self.state_experiment["rates"]
        if any(0. >= x for x in rates):
            print(f"erroneous rates: {str(rates):s}. skipping...")
            return

        value_market_last = self.state_experiment["value_market_last"]
        portfolios = self.state_experiment["portfolios"]
        value_portfolios_last = self.state_experiment["value_portfolios_last"]

        growth_market = self.__get_growth_market(growths)
        value_market = value_market_last * growth_market
        value_portfolios = tuple(self.__evaluate(i) for i in range(len(self.applications)))
        for each_value in value_portfolios:
            assert 0. < each_value

        growth_portfolios = self.__get_growth_portfolios(portfolios, value_portfolios_last, rates)
        self.__add_to_graph(growth_market, value_market, growth_portfolios, value_portfolios)

        if Timer.time_passed(2000):
            for each_application in self.applications:
                if not isinstance(each_application, TraderApproximation):
                    continue
                each_approximation = each_application.approximation
                if isinstance(each_approximation, ApproximationSemioticModel):
                    print(f"{each_approximation.index_classifier_current:>3d}: {str(each_approximation.get_structure()):s}")

        self.state_experiment["value_market_last"] = value_market
        self.state_experiment["value_portfolios_last"] = value_portfolios

    def __get_growth_portfolios(self, portfolios: Sequence[Sequence[float]], value_portfolios_last: Sequence[float], rates: Sequence[float]) -> Sequence[float]:
        return tuple(
            0. if each_value_last == 0.
            else
            1.
            if -1. >= max(each_portfolio) or each_value_last < 0.
            else
            sum(a * r for a, r in zip(each_portfolio, rates)) / each_value_last
            for i, (each_portfolio, each_value_last) in enumerate(zip(portfolios, value_portfolios_last))
        )

    def __get_growth_market(self, growths: Sequence[float]) -> float:
        return sum(growths) / len(growths)

    def __add_to_graph(self, growth_market: float, value_market: float, growth_investors: Sequence[float], value_portfolios: Sequence[float]):
        no_trades = self.state_experiment["no_trades"]

        # value points
        points_values = {f"{str(each_application):s}": v for each_application, v in zip(self.applications, value_portfolios)}
        points_values["market"] = value_market

        # relative growth points
        points_growth = {
            f"{str(each_application):s}": r / growth_market
            for each_application, r in zip(self.applications, growth_investors)
        }
        points_growth["market"] = 1.

        # trade points
        points_trades = {f"{str(each_application):s}": no_trades[i] for i, each_application in enumerate(self.applications)}
        dt = datetime.datetime.utcfromtimestamp(self.timestamp // 1000)

        self.graph.add_snapshot(dt, (points_values, points_growth, points_trades))

    def __evaluate(self, index_application: int) -> float:
        portfolios = self.state_experiment["portfolios"]
        portfolio = portfolios[index_application]
        if -1. >= max(portfolio):
            return 1.
        rates = self.state_experiment["rates"]
        return sum(a * r for a, r in zip(portfolio, rates))

    def start(self):
        super().start()
        if self.graph is not None:
            self.graph.show()
