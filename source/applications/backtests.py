import datetime
from typing import Dict, Tuple, Generic, Collection, Callable, Any

from matplotlib import pyplot
from matplotlib.axes import Axes

from source.data.data_generation import series_generator
from source.tactics.signals.signals import TradingSignal, SIGNAL_INPUT, RATE_INFO, SymmetricChannelSignal


PORTFOLIO_INFO = Dict[str, float]
TECH_INFO = Tuple[RATE_INFO, PORTFOLIO_INFO]
SIGNALS_OUTPUT = Dict[str, float]
SIGNALS_INPUT = Dict[str, SIGNAL_INPUT]
REBALANCER = Callable[TECH_INFO, PORTFOLIO_INFO]
RATE_GENERATOR = Callable[Any, Tuple[datetime.datetime, float]]  # covers generators?!

# parameters of data generator
"""
date_start: str, date_end: str,
interval_minutes: int,
source_dir: str
"""


class TradingBot(Generic[SIGNAL_INPUT]):
    def __init__(self,
                 portfolio: PORTFOLIO_INFO,
                 signals: Dict[str, TradingSignal],
                 data_sources: Dict[str, RATE_GENERATOR],
                 risk: float = .5,
                 rebalancer: REBALANCER = None,
                 trading_fee_percentage: float = .0025,
                 base_asset: str = "ETH",
                 min_base_transaction_volume: float = .025):

        if base_asset not in data_sources:
            raise ValueError("No data source for base asset.")

        self.signals = signals
        self.data_sources = data_sources
        self.base_asset = base_asset
        self.risk = risk
        self.rebalancer = rebalancer
        self.trading_factor = 1. - trading_fee_percentage
        self.min_base_transaction_volume = min_base_transaction_volume
        self.rates = None
        self.current_time = None

        self.asset_development = []  # Sequence[Tuple[datetime.datetime, Dict[str, float]]]

    def _log_portfolio(self, time: datetime.datetime, portfolio: PORTFOLIO_INFO):
        new_entry = time, dict(portfolio)
        self.asset_development.append(new_entry)

    @staticmethod
    def _get_delta(portfolio: PORTFOLIO_INFO, base_asset: str, signals: SIGNALS_OUTPUT, ratio: float) -> PORTFOLIO_INFO:
        base_volume = portfolio.get(base_asset, 0.)
        pos_sum = sum(_v for _v in signals.values() if _v < 0.)
        delta = dict()
        for _k, _s in signals.items():
            if _s < 0.:
                asset_volume = portfolio.get(_k, -1.)
                if 0. < asset_volume:
                    delta[_k] = ratio * asset_volume * _s
            elif 0. < _s:
                delta[_k] = ratio * base_volume * _s / pos_sum

        return delta

    def __get_rates(self) -> RATE_INFO:
        rate_info = dict()
        dates = set()
        for each_asset, each_source in self.data_sources.items():
            each_date, each_rate = next(each_source)
            dates.add(each_date)
            if 1 < len(dates):
                raise ValueError("Data sources returned inconsistent time information.")
            self.current_time, = dates
            rate_info[each_asset] = each_rate
        return rate_info

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

    def run(self):
        while True:
            try:
                self.rates = self.__get_rates()
            except StopIteration as e:
                break

            signal_inputs = self._get_signals_input()   # internet problems?
            portfolio = self._get_portfolio()           # no access to api?
            signals = self._get_signals(signal_inputs)

            self._log_portfolio(self.current_time, portfolio)

            portfolio_delta = TradingBot._get_delta(portfolio, self.base_asset, signals, self.risk)
            if self.rebalancer is not None:
                portfolio_target = self._change_portfolio(portfolio, portfolio_delta, no_fees=True)
                tech_info = portfolio_target, self.rates
                portfolio_delta = self.rebalancer(tech_info)

            portfolio_target = self._change_portfolio(portfolio, portfolio_delta)
            self._redistribute(portfolio, portfolio_target)

    def _change_portfolio(self, portfolio: PORTFOLIO_INFO, delta: PORTFOLIO_INFO, no_fees: bool = False):
        all_assets = set(portfolio.keys()) | set(delta.keys())
        if no_fees:
            return {_k: (portfolio.get(_k, .0) + delta.get(_k, .0)) for _k in all_assets}
        return {_k: (portfolio.get(_k, .0) + delta.get(_k, .0)) * self.trading_factor ** 2 for _k in all_assets}

    def _redistribute(self, current_portfolio: PORTFOLIO_INFO, target_portfolio: PORTFOLIO_INFO):
        # call self._transfer
        pass

    def _transfer(self, source_value: float, source_asset: str, target_value: float, target_asset: str):
        raise NotImplementedError()

    def _get_portfolio(self):
        raise NotImplementedError()

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
