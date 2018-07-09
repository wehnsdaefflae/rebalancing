import datetime
import os
from typing import Dict, Tuple, Generic, Callable, Any, List, Iterator, Collection, Sequence

from matplotlib import pyplot
from matplotlib.axes import Axes

from source.data.data_generation import DEBUG_SERIES
from source.tactics.signals.signals import TradingSignal, SIGNAL_INPUT, RATE_INFO, SymmetricChannelSignal, \
    RelativeStrengthIndexSignal

PORTFOLIO_INFO = Dict[str, float]
TECH_INFO = Tuple[RATE_INFO, PORTFOLIO_INFO]
SIGNALS_OUTPUT = Dict[str, float]
SIGNALS_INPUT = Dict[str, SIGNAL_INPUT]
BALANCING = Callable[[TECH_INFO], PORTFOLIO_INFO]
RATE_GENERATOR = Iterator[Tuple[datetime.datetime, float]]

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
                 risk: float = 1.,
                 dead_zone: Tuple[float, float] = (-.5, .5),
                 balance: float = 0.,
                 trading_fee_percentage: float = .0025,
                 base_asset: str = "ETH",
                 min_base_transaction_volume: float = .025):

        if base_asset in signals:
            raise ValueError("No base asset signal for now.")

        if min_base_transaction_volume < 0.:
            raise ValueError("Minimum transaction value must be positive.")

        self.signals = signals
        self.data_sources = data_sources
        self.base_asset = base_asset
        self.risk = risk
        self.dead_zone = dead_zone
        self.balance = balance
        self.trading_factor = 1. - trading_fee_percentage
        self.min_base_transaction_volume = min_base_transaction_volume

        self.iteration = 0
        now = datetime.datetime.utcnow()
        self.date_format = "%Y-%m-%d_%H-%M-%S_UTC"
        self.log_path = "../../logs/{:s}_{:d}.log".format(now.strftime(self.date_format), id(self))
        self.log_header = []

    def _get_rates(self) -> Tuple[datetime.datetime, RATE_INFO]:
        rate_info = dict()                                                      # type: Dict[str, float]
        utc_timestamps = []                                                     # type: List[float]
        for each_asset, each_source in self.data_sources.items():
            each_date, each_rate = next(each_source)                            # type: datetime.datetime, float
            utc_timestamps.append(each_date.timestamp())
            rate_info[each_asset] = each_rate
        rate_info[self.base_asset] = 1.
        average_timestamp = sum(utc_timestamps) / len(utc_timestamps)           # type: float
        current_time = datetime.datetime.utcfromtimestamp(average_timestamp)    # type: datetime.datetime
        return current_time, rate_info

    def _get_portfolio(self) -> PORTFOLIO_INFO:
        raise NotImplementedError()

    def __log_portfolio(self, time: datetime.datetime, portfolio_base_value: PORTFOLIO_INFO):
        with open(self.log_path, mode="a") as file:
            if len(self.log_header) < 1:
                self.log_header = sorted(portfolio_base_value.keys())
                row = "time", *self.log_header
                file.write("\t".join(row) + "\n")

            formatted_numbers = ["{:.4f}".format(portfolio_base_value.get(_x, 0.)) for _x in self.log_header]
            row = time.strftime(self.date_format), *formatted_numbers
            file.write("\t".join(row) + "\n")

    def _get_signals_input(self) -> SIGNALS_INPUT:
        raise NotImplementedError()

    def __get_signals(self, signals_input: SIGNALS_INPUT) -> SIGNALS_OUTPUT:
        tendencies = dict()
        for each_asset, each_input in signals_input.items():
            each_signal = self.signals.get(each_asset, None)
            if each_signal is None:
                continue
            each_output = each_signal.get_tendency(each_input)
            if each_output < self.dead_zone[0] or self.dead_zone[1] < each_output:
                each_output *= self.risk
            else:
                each_output = 0.

            tendencies[each_asset] = each_output
        return tendencies

    def __get_transactions(self, portfolio: PORTFOLIO_INFO, target: PORTFOLIO_INFO, rates: RATE_INFO) -> Sequence[Tuple[float, str, str]]:
        to_base_asset, from_base_asset = [], []

        for each_asset, each_value in target.items():
            if each_asset == self.base_asset:
                continue

            each_diff = each_value - portfolio[each_asset]
            base_diff = each_diff * rates[each_asset]

            if base_diff < -self.min_base_transaction_volume:
                transaction = -each_diff, each_asset, self.base_asset
                to_base_asset.append(transaction)

            elif self.min_base_transaction_volume < base_diff:
                transaction = base_diff, self.base_asset, each_asset
                from_base_asset.append(transaction)

        return to_base_asset + from_base_asset

    @staticmethod
    def __balanced_portfolio(portfolio: PORTFOLIO_INFO) -> PORTFOLIO_INFO:
        return {_k: 1. for _k in portfolio}

    def __signal_portfolio(self, portfolio: PORTFOLIO_INFO, signals: SIGNALS_OUTPUT, rates: RATE_INFO) -> PORTFOLIO_INFO:
        no_signals = len(signals)
        if no_signals < 1:
            return dict(portfolio)

        base_values = {_k: portfolio[_k] * rates[_k] for _k in portfolio}
        total_value = sum(base_values.values())
        ratios = {_k: base_values[_k] / total_value for _k in signals}

        changed_ratios = {_k: _v * (1. + signals[_k]) for _k, _v in ratios.items()}
        ratio_sum = sum(changed_ratios.values())
        if 0. >= ratio_sum:
            return dict(portfolio)

        changed_ratios[self.base_asset] = 0.
        return changed_ratios

    def __transfer(self, source_value: float, source_asset: str, target_asset: str):
        if source_asset == target_asset or source_value == 0.:
            return
        if source_value < 0.:
            self._transfer(-source_value, target_asset, source_asset)
        else:
            self._transfer(source_value, source_asset, target_asset)

    def _transfer(self, source_value: float, source_asset: str, target_asset: str):
        raise NotImplementedError()

    def _wait(self):
        raise NotImplementedError()

    @staticmethod
    def _get_total_base_value(portfolio: PORTFOLIO_INFO, rates: RATE_INFO):
        return sum(_v * rates[_k] for _k, _v in portfolio.items())

    @staticmethod
    def _distribute(total_base_value: float, distribution: PORTFOLIO_INFO, rates: RATE_INFO) -> PORTFOLIO_INFO:
        dist_total = sum(distribution.values())
        return {_k: total_base_value * _v / (rates[_k] * dist_total) for _k, _v in distribution.items()}

    def run(self):
        while True:
            try:
                time, rates = self._get_rates()
            except StopIteration as e:
                print(e)
                break

            try:
                portfolio = self._get_portfolio()
                base_values = {_k: _v * rates[_k] for _k, _v in portfolio.items()}
                self.__log_portfolio(time, base_values)
            except Exception as e:
                print(e)
                continue

            if self.iteration == 0:
                distribution = TradingBot.__balanced_portfolio(portfolio)

            else:
                try:
                    signal_inputs = self._get_signals_input()
                except Exception as e:
                    print(e)
                    continue

                signals = self.__get_signals(signal_inputs)
                distribution = self.__signal_portfolio(portfolio, signals, rates)

            total_base_value = TradingBot._get_total_base_value(portfolio, rates)
            target_portfolio = TradingBot._distribute(total_base_value, distribution, rates)
            transactions = self.__get_transactions(portfolio, target_portfolio, rates)

            for source_value, source_asset, target_asset in transactions:
                try:
                    #if portfolio[source_asset] < source_value:
                    #    raise ValueError()
                    self.__transfer(source_value, source_asset, target_asset)
                except ValueError as e:
                    raise e

            if self.iteration % 100 == 0:
                print("Total value: {:s} {:.5f}".format(self.base_asset, total_base_value))
            self._wait()
            self.iteration += 1

    def plot_stack(self, axis: Axes):
        time_axis = []
        sequences = []
        with open(self.log_path, mode="r") as file:
            header = file.readline()
            row = header[:-1].split("\t")
            for _ in range(len(row) - 1):
                sequences.append([])

            for each_line in file:
                row = each_line[:-1].split("\t")

                time_axis.append(datetime.datetime.strptime(row[0], self.date_format))
                for cell_index in range(1, len(row)):
                    each_sequence = sequences[cell_index - 1]
                    value = float(row[cell_index])
                    each_sequence.append(value)

        axis.stackplot(time_axis, *sequences, labels=self.log_header)


class Simulation(TradingBot[RATE_INFO]):
    def __init__(self,
                 portfolio: PORTFOLIO_INFO,
                 signals: Dict[str, TradingSignal],
                 data_sources: Dict[str, RATE_GENERATOR],
                 balance: float = 0.,
                 trading_fee_percentage: float = .0025,
                 base_asset: str = "ETH",
                 min_base_transaction_volume: float = .025):
        super().__init__(
            signals,
            data_sources,
            risk=.1,
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

    def _transfer(self, source_value: float, source_asset: str, target_asset: str):
        if source_asset == target_asset or source_value == 0.:
            return

        elif source_value < 0.:
            raise ValueError("No negative values.")

        source_volume = self.portfolio[source_asset]
        source_value = min(source_volume, source_value)

        self.portfolio[source_asset] = source_volume - source_value
        rate = self.rates[source_asset] / self.rates[target_asset]
        self.portfolio[target_asset] += self.trading_factor * source_value * rate

    def _wait(self):
        pass


def main():
    portfolio = {"IOTA": 0., "ADA": 0., "ETH": 10.}
    signals = {_k: SymmetricChannelSignal() for _k in portfolio if _k != "ETH"}
    # signals = {_k: RelativeStrengthIndexSignal(history_length=50) for _k in portfolio if _k != "ETH"}
    data_sources = {_k: DEBUG_SERIES(_k) for _k in portfolio if _k != "ETH"}
    simulation = Simulation(portfolio, signals, data_sources)
    simulation.run()

    pyplot.clf()
    pyplot.close()
    fix, ax1 = pyplot.subplots(1, sharex="all")
    simulation.plot_stack(ax1)
    pyplot.legend()
    pyplot.show()


if __name__ == "__main__":
    main()
