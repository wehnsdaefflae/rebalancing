import datetime
from typing import Dict, Tuple, Generic

from source.signal_system.signals import TradingSignal, TECH_INFO, SIGNAL_INPUT, PORTFOLIO_INFO, RATE_INFO, \
    SIGNAL_OUTPUT


class Backtest(Generic[SIGNAL_INPUT]):
    def __init__(self,
                 base_asset: str,
                 initial_assets: Dict[str, float],
                 asset_signals: Dict[str, TradingSignal],
                 date_start: datetime.datetime, date_end: datetime.datetime,
                 initialization: int = 0,
                 dead_zone: Tuple[float] = (-.2, .2),
                 trading_fee: float = .0025,
                 trade_min_base_volume: float = 0.025):

        if set(initial_assets.keys()) != set(asset_signals.keys()):
            raise ValueError("Every assets requires a signal.")
        if base_asset not in initial_assets:
            raise ValueError("Base asset not among initial assets.")

        self.base_asset = base_asset
        self.asset_development = {_k: [_v] for _k, _v in initial_assets.items()}
        self.signals = asset_signals
        self.start_date = date_start
        self.end_date = date_end
        self.initialization = initialization
        self.dead_zone = dead_zone
        self.trading_factor = 1. - trading_fee
        self.trade_min_base_volume = trade_min_base_volume

    def _get_portfolio(self) -> PORTFOLIO_INFO:
        return {_k: _v[-1] for _k, _v in self.asset_development.items()}

    def _log_portfolio(self, portfolio: PORTFOLIO_INFO):
        for each_asset, each_value in portfolio.values():
            each_development = self.asset_development[each_asset]
            each_development.append(each_value)

    def iterate(self, rates: Dict[str, float]):
        # wait for initialization to finish
        if 0 < self.initialization:
            self.initialization -= 1
            portfolio = self._get_portfolio()
        else:
            portfolio = self._redistribute(rates)

        self._log_portfolio(portfolio)

    def _redistribute(self, rates: RATE_INFO) -> PORTFOLIO_INFO:
        raise NotImplementedError()

    def _get_tendencies(self, source_info: SIGNAL_INPUT) -> SIGNAL_OUTPUT:
        raise NotImplementedError()


class TechnicalBacktest(Backtest[TECH_INFO]):
    def __init__(self, *arguments, **keywords):
        super().__init__(**keywords, *arguments)

    def _get_tendencies(self, source_info: TECH_INFO) -> SIGNAL_OUTPUT:
        tendencies = dict()
        for each_asset in self.asset_development:
            each_signal = self.signals[each_asset]
            each_tendency = each_signal.get_tendency(source_info)
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
        super().__init__(**keywords, *arguments)
