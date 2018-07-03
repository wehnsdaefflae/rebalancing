import datetime
from typing import Dict, Tuple, Generic, Sequence

from matplotlib import pyplot
from matplotlib.axes import Axes

from source.data.data_generation import series_generator
from source.signal_system.signals import TradingSignal, TECH_INFO, SIGNAL_INPUT, PORTFOLIO_INFO, RATE_INFO, \
    SIGNAL_OUTPUT, ChannelSignal


class Backtest(Generic[SIGNAL_INPUT]):
    def __init__(self,
                 base_asset: str,
                 initial_assets: Dict[str, float],
                 asset_signals: Dict[str, TradingSignal],
                 source_dir: str,
                 date_start: str, date_end: str,
                 interval_minutes: int,
                 dead_zone: Tuple[float] = (-.2, .2),
                 trading_fee: float = .0025,
                 trade_min_base_volume: float = 0.025):

        if not all(_x in initial_assets.keys() for _x in asset_signals):
            raise ValueError("Every signal requires an asset.")
        if base_asset not in initial_assets:
            raise ValueError("Base asset not among initial assets.")

        self.base_asset = base_asset
        self.asset_development = {_k: [_v] for _k, _v in initial_assets.items()}
        self.signals = asset_signals
        self.source_dir = source_dir
        self.start_date = date_start
        self.end_date = date_end
        self.interval_minutes = interval_minutes
        self.dead_zone = dead_zone
        self.trading_factor = 1. - trading_fee
        self.trade_min_base_volume = trade_min_base_volume
        self.time_axis = []

    def _get_portfolio(self) -> PORTFOLIO_INFO:
        return {_k: _v[-1] for _k, _v in self.asset_development.items()}

    def _log_portfolio(self, portfolio: PORTFOLIO_INFO):
        for each_asset, each_value in portfolio.items():
            each_development = self.asset_development[each_asset]
            each_development.append(each_value)

    def simulate(self):
        # get input rates
        start_date = datetime.datetime.strptime(self.start_date, "%Y-%m-%d_%H:%M:%S_%Z")
        end_date = datetime.datetime.strptime(self.end_date, "%Y-%m-%d_%H:%M:%S_%Z")
        self.time_axis.clear()
        self.time_axis.append(start_date)
        while self.time_axis[-1] + datetime.timedelta(minutes=self.interval_minutes) < end_date:
            self.time_axis.append(self.time_axis[-1] + datetime.timedelta(minutes=self.interval_minutes))

        # replace get_series with generator
        series = dict()
        for each_cur in self.signals:
            source_path = self.source_dir + "{:s}{:s}.csv".format(each_cur, self.base_asset)
            series[each_cur] = series_generator(source_path,
                                                range_start=start_date, range_end=end_date,
                                                interval_minutes=self.interval_minutes)

        for i in range(len(self.time_axis) - 1):
            self._iterate({_c: series[_c][i] for _c in self.signals})

    def _iterate(self, rates: Dict[str, float]):
        portfolio = self._redistribute(rates)
        self._log_portfolio(portfolio)

    def _redistribute(self, rates: RATE_INFO) -> PORTFOLIO_INFO:
        raise NotImplementedError()

    def _get_tendencies(self, source_info: SIGNAL_INPUT) -> SIGNAL_OUTPUT:
        raise NotImplementedError()

    def plot_stack(self, axis: Axes):
        if len(self.time_axis) < 1:
            raise ReferenceError("Simulation has not been run.")
        portfolio = self._get_portfolio()
        assets = sorted(set(portfolio.keys()))
        axis.stackplot(self.time_axis, *[self.asset_development[_k] for _k in assets], labels=assets)


class TechnicalBacktest(Backtest[TECH_INFO]):
    def __init__(self, *arguments, **keywords):
        super().__init__(*arguments, **keywords)

    def _get_tendencies(self, source_info: TECH_INFO) -> SIGNAL_OUTPUT:
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
              "asset_signals": {"ADA": ChannelSignal("ADA", window_size=50)},
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
