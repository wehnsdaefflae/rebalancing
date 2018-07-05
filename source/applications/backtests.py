import datetime
from typing import Dict, Tuple, Generic, Collection, Callable, Any, Sequence, Optional, List

from matplotlib import pyplot
from matplotlib.axes import Axes

from source.tactics.signals.signals import TradingSignal, SIGNAL_INPUT, RATE_INFO, SymmetricChannelSignal


PORTFOLIO_INFO = Dict[str, float]
TECH_INFO = Tuple[RATE_INFO, PORTFOLIO_INFO]
SIGNALS_OUTPUT = Dict[str, float]
SIGNALS_INPUT = Dict[str, SIGNAL_INPUT]
BALANCING = Callable[TECH_INFO, PORTFOLIO_INFO]
RATE_GENERATOR = Callable[Any, Tuple[datetime.datetime, float]]  # covers generators?!

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
                 balancing: BALANCING = None,
                 trading_fee_percentage: float = .0025,
                 base_asset: str = "ETH",
                 min_base_transaction_volume: float = .025):

        if base_asset not in data_sources:
            raise ValueError("No data source for base asset.")

        if base_asset in signals:
            raise ValueError("No bas asset signal for now.")

        self.signals = signals
        self.data_sources = data_sources
        self.base_asset = base_asset
        self.risk = risk
        self.balancing = balancing
        self.trading_factor = 1. - trading_fee_percentage
        self.min_base_transaction_volume = min_base_transaction_volume

        self.asset_development = []     # type: List[Tuple[datetime.datetime, Dict[str, float]]]

    def __get_rates(self) -> Tuple[datetime.datetime, RATE_INFO]:
        rate_info = dict()
        dates = set()
        for each_asset, each_source in self.data_sources.items():
            each_date, each_rate = next(each_source)
            dates.add(each_date)
            if 1 < len(dates):
                raise ValueError("Data sources returned inconsistent time information.")
            rate_info[each_asset] = each_rate
        current_time, = dates
        return current_time, rate_info

    def _get_portfolio(self):
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
    def __get_delta(portfolio: PORTFOLIO_INFO, base_asset: str, signals: SIGNALS_OUTPUT, risk: float) -> PORTFOLIO_INFO:
        base_volume = portfolio.get(base_asset, 0.)
        pos_sum = sum(_v for _v in signals.values() if .0 < _v)
        delta = dict()
        for _k, _s in signals.items():
            if _s < 0.:     # sell
                asset_volume = portfolio.get(_k, -1.)
                if 0. < asset_volume:
                    delta[_k] = risk * asset_volume * _s

            elif 0. < _s:   # buy
                delta[_k] = risk * base_volume * _s / pos_sum  # * asset_rate

        return delta

    def __change_portfolio(self, portfolio: PORTFOLIO_INFO, delta: PORTFOLIO_INFO, no_fees: bool = False):
        all_assets = set(portfolio.keys()) | set(delta.keys())
        if no_fees:
            return {_k: (portfolio.get(_k, .0) + delta.get(_k, .0)) for _k in all_assets}
        return {_k: (portfolio.get(_k, .0) + delta.get(_k, .0)) * self.trading_factor ** 2 for _k in all_assets}

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
                self._transfer(base_value, self.base_asset, each_asset, asset_rate)

    def _transfer(self, source_value: float, source_asset: str, target_asset: str, rate: float):
        raise NotImplementedError()

    def _wait(self):
        raise NotImplementedError()

    def run(self):
        while True:
            try:
                time, rates = self.__get_rates()
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

            portfolio_delta = TradingBot.__get_delta(portfolio, self.base_asset, signals, self.risk)
            if self.balancing is not None:
                portfolio_target = self.__change_portfolio(portfolio, portfolio_delta, no_fees=True)
                tech_info = portfolio_target, rates
                portfolio_delta = self.balancing(tech_info)

            self.__redistribute(portfolio_delta, rates)
            self._wait()

    def plot_stack(self, axis: Axes):
        all_assets = sorted(set(_x for _y in self.asset_development for _x in _y[1].keys()))
        all_plots = {_k: [] for _k in all_assets}
        time_axis = []
        for each_time, each_portfolio in self.asset_development:
            time_axis.append(each_time)
            for each_asset in all_assets:
                each_development = all_plots[each_asset]
                each_development.appen(each_portfolio.get(each_asset, .0))

        axis.stackplot(time_axis, *[all_plots[_k] for _k in all_assets], labels=all_assets)


class Simulation(TradingBot):
    def run(self):
        self._simulate(None, None)


class TechnicalBacktest(Backtest[TECH_INFO]):
    def __init__(self, *arguments, **keywords):
        super().__init__(*arguments, **keywords)

    def _get_tendencies(self, source_info: TECH_INFO) -> SIGNALS_OUTPUT:
        portfolio, rates = source_info
        tendencies = dict()
        for each_asset in self.signals:
            each_signal = self.signals[each_asset]
            each_tendency = each_signal.get_tendency(rates)
            tendencies[each_asset] = each_tendency
        return tendencies

    def _redistribute(self, rates: RATE_INFO):
        # get current portfolio and signal tendencies
        portfolio = self._get_portfolio()
        tendencies = self._get_tendencies((portfolio, rates))

        # sell assets for base asset
        redistribution = dict()
        for each_asset, each_tendency in tendencies.items():
            if self.dead_zone[0] >= each_tendency and each_asset != self.base_asset:
                asset_value = portfolio[each_asset]
                diff = asset_value * each_tendency
                diff_base = diff / rates[each_asset]
                if diff_base < self.trade_min_base_volume:
                    continue

                portfolio[each_asset] -= diff
                portfolio[self.base_asset] += self.trading_factor * diff_base

            elif each_tendency >= self.dead_zone[1]:
                redistribution[each_asset] = each_tendency

        # buy assets from base asset
        total = sum(redistribution.values())
        base_value = portfolio[self.base_asset]
        for each_asset, each_tendency in redistribution.items():
            if each_tendency < self.dead_zone[1] or each_asset == self.base_asset:
                continue

            each_ratio = each_tendency / total
            diff_base = each_ratio * base_value
            if diff_base < self.trade_min_base_volume:
                continue

            portfolio[self.base_asset] -= diff_base
            diff = diff_base * rates[each_asset]
            portfolio[each_asset] += self.trading_factor * diff

        return portfolio


class FundamentalBacktest(Backtest):
    def __init__(self, *arguments, **keywords):
        super().__init__(*arguments, **keywords)


if __name__ == "__main__":
    config = {"base_asset": "ETH",
              "initial_assets": {"ADA": 0.,
                                 "ETH": 10.},
              "asset_signals": {"ADA": SymmetricChannelSignal(window_size=50)},
              "source_dir": "../../data/binance/23Jun2017-23Jun2018-1m/",
              "date_start": "2018-06-01_00:00:00_UTC",
              "date_end": "2018-06-07_00:00:00_UTC",
              "interval_minutes": 10,
              "dead_zone": (-.2, .2),
              "trading_fee": .0025,
              "trade_min_base_volume": .025
              }
    tbt = TechnicalBacktest(**config)
    tbt.simulate()

    fig, ax1 = pyplot.subplots(1, sharex="all")
    tbt.plot_stack(ax1)
    pyplot.show()
