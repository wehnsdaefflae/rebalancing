import datetime
from typing import Dict, Tuple, Generic, Callable, Any, List, Iterator

from matplotlib import pyplot
from matplotlib.axes import Axes

from source.data.data_generation import DEBUG_SERIES
from source.tactics.signals.signals import TradingSignal, SIGNAL_INPUT, RATE_INFO, SymmetricChannelSignal

PORTFOLIO_INFO = Dict[str, float]
TECH_INFO = Tuple[RATE_INFO, PORTFOLIO_INFO]
SIGNALS_OUTPUT = Dict[str, float]
SIGNALS_INPUT = Dict[str, SIGNAL_INPUT]
BALANCING = Callable[[TECH_INFO], PORTFOLIO_INFO]
RATE_GENERATOR = Iterator[Tuple[datetime.datetime, float]]  # covers generators?!

# parameters of data generator
"""
date_start: str, date_end: str,
interval_minutes: int,
source_dir: str
"""


class TradingBot(Generic[SIGNAL_INPUT]):
    def __init__(self,
                 signals: Dict[str, TradingSignal],
                 data_sources: Dict[str, RATE_GENERATOR],
                 risk: float = .5,
                 balance: float = 0.,
                 trading_fee_percentage: float = .0025,
                 base_asset: str = "ETH",
                 min_base_transaction_volume: float = .025):

        if base_asset in signals:
            raise ValueError("No base asset signal for now.")

        self.signals = signals
        self.data_sources = data_sources
        self.base_asset = base_asset
        self.risk = risk
        self.balance = balance
        self.trading_factor = 1. - trading_fee_percentage
        self.min_base_transaction_volume = min_base_transaction_volume

        self.asset_development = []  # type: List[Tuple[datetime.datetime, Dict[str, float]]]

    def _get_rates(self) -> Tuple[datetime.datetime, RATE_INFO]:
        rate_info = dict()                                                      # type: Dict[str, float]
        utc_timestamps = []                                                     # type: List[float]
        for each_asset, each_source in self.data_sources.items():
            each_date, each_rate = next(each_source)                            # type: datetime.datetime, float
            utc_timestamps.append(each_date.timestamp())
            rate_info[each_asset] = each_rate
        average_timestamp = sum(utc_timestamps) / len(utc_timestamps)           # type: float
        current_time = datetime.datetime.utcfromtimestamp(average_timestamp)    # type: datetime.datetime
        return current_time, rate_info

    def _get_portfolio(self) -> PORTFOLIO_INFO:
        raise NotImplementedError()

    def __log_portfolio(self, time: datetime.datetime, portfolio: PORTFOLIO_INFO):
        new_entry = time, dict(portfolio)
        self.asset_development.append(new_entry)

    def _get_signals_input(self) -> SIGNALS_INPUT:
        raise NotImplementedError()

    def _get_signals(self, signals_input: SIGNALS_INPUT) -> SIGNALS_OUTPUT:
        tendencies = dict()
        for each_asset, each_input in signals_input.items():
            each_signal = self.signals.get(each_asset, None)
            if each_signal is None:
                continue
            each_output = each_signal.get_tendency(each_input)
            tendencies[each_asset] = each_output
        return tendencies

    @staticmethod
    def __get_target(portfolio: PORTFOLIO_INFO, base_asset: str, rates: RATE_INFO, signals: SIGNALS_OUTPUT,
                     balance: float) -> PORTFOLIO_INFO:
        if not 1. >= balance >= 0.:
            raise ValueError("Balance must be between 0. and 1.")
        no_signals = len(signals)
        if no_signals < 1:
            return dict(portfolio)
        normalized_signals = {_k: (_v + 1.) / 2. for _k, _v in signals.items()}
        signal_sum = sum(normalized_signals.values())
        normalized_ratios = {_k: 0. if .0 >= signal_sum else _v / signal_sum for _k, _v in normalized_signals.items()}
        total_base_value = sum(portfolio[_k] * rates[_k] for _k in signals) + portfolio.get(base_asset, 0.)
        target_distribution = {_k: total_base_value * normalized_ratios[_k] / rates[_k] for _k, _v in signals.items()}
        # now rebalance
        return target_distribution

    def __redistribute(self, delta: PORTFOLIO_INFO, rates: RATE_INFO):
        if self.base_asset in delta:
            raise ValueError("No base asset transactions!")

        for each_asset, each_value in delta.items():
            if each_value < 0.:
                asset_rate = rates[each_asset]
                base_value = each_value / asset_rate
                if -base_value < self.min_base_transaction_volume:
                    continue
                self._transfer(-each_value, each_asset, self.base_asset, asset_rate)

            elif 0. < each_value:
                asset_rate = rates[each_asset]
                base_value = each_value / asset_rate
                if base_value < self.min_base_transaction_volume:
                    continue
                self._transfer(base_value, self.base_asset, each_asset, 1. / asset_rate)

    def _transfer(self, source_value: float, source_asset: str, target_asset: str, rate: float):
        raise NotImplementedError()

    def _wait(self):
        raise NotImplementedError()

    def run(self):
        while True:
            try:
                time, rates = self._get_rates()
            except StopIteration as e:
                print(e)
                break

            try:
                portfolio = self._get_portfolio()
                self.__log_portfolio(time, portfolio)
            except Exception as e:
                print(e)
                continue

            try:
                signal_inputs = self._get_signals_input()
            except Exception as e:
                print(e)
                continue

            signals = self._get_signals(signal_inputs)

            if not all(_x == .0 for _x in signals.values()):
                a = 0

            target_portfolio = self.__get_target(portfolio, self.base_asset, rates, signals, self.balance)
            portfolio_delta = {_k: target_portfolio.get(_k, portfolio[_k]) - _v for _k, _v in signals.items()}

            try:
                self.__redistribute(portfolio_delta, rates)
            except ValueError as e:
                raise e
            self._wait()

    def plot_stack(self, axis: Axes):
        # TODO: write to file instead
        all_assets = sorted(set(_x for _y in self.asset_development for _x in _y[1].keys()))
        all_plots = {_k: [] for _k in all_assets}
        time_axis = []
        for each_time, each_portfolio in self.asset_development:
            time_axis.append(each_time)
            for each_asset in all_assets:
                each_development = all_plots[each_asset]
                each_development.append(each_portfolio.get(each_asset, .0))

        axis.stackplot(time_axis, *[all_plots[_k] for _k in all_assets], labels=all_assets)


class Simulation(TradingBot[RATE_INFO]):
    def __init__(self,
                 portfolio: PORTFOLIO_INFO,
                 signals: Dict[str, TradingSignal],
                 data_sources: Dict[str, RATE_GENERATOR],
                 risk: float = .5,
                 balance: float = 0.,
                 trading_fee_percentage: float = .0025,
                 base_asset: str = "ETH",
                 min_base_transaction_volume: float = .025):
        super().__init__(
            signals,
            data_sources,
            risk=risk,
            balance=balance,
            trading_fee_percentage=trading_fee_percentage,
            base_asset=base_asset,
            min_base_transaction_volume=min_base_transaction_volume
        )
        self.portfolio = dict(portfolio)
        self.rates = dict()                 # type: RATE_INFO

    def _get_rates(self) -> Tuple[datetime.datetime, RATE_INFO]:
        time, self.rates = super()._get_rates()
        return time, self.rates

    def _get_portfolio(self) -> PORTFOLIO_INFO:
        return self.portfolio

    def _get_signals_input(self) -> RATE_INFO:
        return self.rates

    def _transfer(self, source_value: float, source_asset: str, target_asset: str, rate: float):
        target_value = source_value * rate
        self.portfolio[source_asset] -= source_value
        self.portfolio[target_asset] += self.trading_factor * target_value

    def _wait(self):
        pass


def main():
    portfolio = {"IOTA": 0., "ADA": 0., "ETH": 10.}
    signals = {_k: SymmetricChannelSignal() for _k in portfolio if _k != "ETH"}
    data_sources = {_k: DEBUG_SERIES(_k) for _k in portfolio if _k != "ETH"}
    simulation = Simulation(portfolio, signals, data_sources)
    simulation.run()

    pyplot.clf()
    pyplot.close()
    fix, ax1 = pyplot.subplots(1, sharex="all")
    simulation.plot_stack(ax1)
    pyplot.show()


if __name__ == "__main__":
    main()
