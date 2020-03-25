import datetime
import random
from typing import Iterable, Sequence, Tuple, Generator, Union, Collection, Callable, Type, Any, Optional, Dict

from matplotlib import pyplot, dates
from matplotlib.ticker import MaxNLocator

from source.new.binance_examples import STATS, get_pairs_from_filesystem, binance_matrix, generate_path_from_file, generate_path
from source.new.learning import Classification, MultivariateRegression, PolynomialClassification, RecurrentPolynomialClassification, Approximation, smear, \
    MultivariatePolynomialRegression, MultivariatePolynomialRecurrentRegression
from source.new.optimal_trading import generate_multiple_changes, generate_matrix
from source.new.snapshot_generation import merge_generator
from source.tools.timer import Timer


def types_binance(column: str) -> Type:
    if column == "timestamp":
        return int
    if column == "target":
        return str
    if "open_time" in column:
        return int
    if "close_time" in column:
        return int
    if "open" in column:
        return float
    if "high" in column:
        return float
    if "low" in column:
        return float
    if "close" in column:
        return float
    if "volume" in column:
        return float
    if "quote_asset_volume" in column:
        return float
    if "number_of_trades" in column:
        return int
    if "taker_buy_base_asset_volume" in column:
        return float
    if "taker_buy_quote_asset_volume" in column:
        return float
    if "ignore" in column:
        return float


def binance_columns(assets: Collection[str], columns_available: Collection[str]) -> Collection[str]:
    return ("timestamp",) + tuple(
        f"{each_asset.upper():s}_{each_stat.lower():s}"
        for each_asset in assets
        for each_stat in columns_available
    )


class ApproximationInvestmentQuantified:
    def __init__(self,
                 approximation: Approximation,
                 fees: float,
                 certainty_threshold: float,
                 ):
        self.approximation = approximation
        self.fees = fees
        self.certainty_threshold = certainty_threshold

        self.iterations = 0
        self.iterations_total = 0

        self.error_average = 0.
        self.amount_average = 0.

        self.amount = 1.
        self.asset = -1

        self.no_trades = 0

        self.rates_now = None
        self.ratios_now = None

    def __str__(self) -> str:
        return self.approximation.__class__.__name__

    def reset(self):
        self.iterations = 0
        self.error_average = 0.
        self.amount_average = 0.

    def _get_ratio(self, rates: Sequence[float]) -> Sequence[float]:
        if self.rates_now is None:
            return tuple(1. for _ in rates)

        return tuple(
            0. if 0. >= rate_then or 0. >= rate_this else
            rate_this / rate_then
            for rate_this, rate_then in zip(rates, self.rates_now))

    def _act(self, asset_next: int, ratio_next: float):
        # if (initial trade) or (different asset is better and certainty over threshold and [benefit outweights fees? i dont know that!])
        if (self.asset < 0) or (asset_next != self.asset and self.certainty_threshold < ratio_next):
            self.amount *= (1. - self.fees)
            self.asset = asset_next
            self.no_trades += 1

        self.amount *= self.ratios_now[self.asset]
        self.amount_average = smear(self.amount_average, self.amount, self.iterations)

    def batch(self, examples: Iterable[Tuple[Sequence[float], int]]):
        rates_prev = None
        no_assets = -1
        for t, (rates_next, target_next) in enumerate(examples):
            if rates_prev is None:
                rates_prev = rates_next
                no_assets = len(rates_prev)
                continue
            ratio_next = tuple(1. if 0. >= r_prev else r_next / r_prev for r_next, r_prev in zip(rates_next, rates_prev))
            target_values = tuple(float(i == target_next) for i in range(no_assets))
            # self.approximation.fit(ratio_next, target_values, self.iterations_total + t - 1)
            self.approximation.fit(ratio_next, target_values, t - 1)

            if Timer.time_passed(2000):
                print(f"finished training {t:d} examples in batch...")

    def cycle(self, rates_next: Sequence[float], skip_train: bool = False):
        ratios_next = self._get_ratio(rates_next)

        if self.ratios_now is None:
            asset_output = 0
            ratio_output = 0.

        else:
            values_output = self.approximation.output(self.ratios_now)
            if not skip_train:
                # self.approximation.fit(self.ratios_now, ratios_next, self.iterations_total)
                self.approximation.fit(self.ratios_now, ratios_next, 60 * 24)
            asset_output, ratio_output = max(enumerate(values_output), key=lambda x: x[1])

        self.ratios_now = ratios_next   # end time step
        self.rates_now = rates_next     # start time step

        self._act(asset_output, ratio_output)

        asset_target, _ = max(enumerate(rates_next), key=lambda x: x[1])
        self.error_average = smear(self.error_average, float(asset_output != asset_target), self.iterations)

        self.iterations += 1
        if not skip_train:
            self.iterations_total += 1


class VisualizationMixin:
    def __init__(self, no_approximations: int):
        self.axis_time = []
        self.errors = tuple([] for _ in range(no_approximations))
        self.amounts = tuple([] for _ in range(no_approximations))

        self.rate_average_last = -1.
        self.amount_average = -1.
        self.amount = 1.
        self.ratio_list = []
        self.t_last = -1

        self.fig, self.ax_amount = pyplot.subplots()
        self.ax_error = self.ax_amount.twinx()
        self.max_size = 20

    def _update_market(self, rates: Sequence[float], time_step: int):
        rate_average_next = sum(rates) / len(rates)
        ratio_average = 1. if self.rate_average_last < 0. else 0. if self.rate_average_last == 0. else rate_average_next / self.rate_average_last
        self.rate_average_last = rate_average_next
        self.amount *= ratio_average
        self.amount_average = smear(self.amount_average, self.amount, time_step - self.t_last - 1)

    def _update_plot(self, timestamp: int, approximations: Sequence[ApproximationInvestmentQuantified]):
        self.axis_time.append(timestamp)
        del(self.axis_time[:-self.max_size])
        self.ratio_list.append(self.amount_average)
        del(self.ratio_list[:-self.max_size])

        for i, each_approximation in enumerate(approximations):
            if 1 < len(self.axis_time):
                print(f"duration: {(self.axis_time[-1] - self.axis_time[-2]) / (1000. * 60. * 60.):.2f} hours.")
            print(f"number of trades {each_approximation.no_trades:d} for {str(each_approximation.approximation.__class__.__name__):s}.")
            each_approximation.no_trades = 0
            self.errors[i].append(each_approximation.error_average)
            del(self.errors[i][:-self.max_size])
            self.amounts[i].append(each_approximation.amount_average)
            del(self.amounts[i][:-self.max_size])
            each_approximation.reset()

        self._draw(approximations)

    def _draw(self, approximations: Sequence[ApproximationInvestmentQuantified]):
        self.ax_error.clear()
        self.ax_amount.clear()

        self.ax_amount.set_xlabel("time")
        self.ax_amount.set_ylabel("value in ETH")

        self.ax_error.set_ylabel("average error during interval")
        self.ax_error.yaxis.label.set_color("grey")

        self.ax_amount.xaxis.set_major_formatter(dates.DateFormatter("%d.%m.%Y %H:%M"))
        self.ax_amount.xaxis.set_major_locator(MaxNLocator(10))

        self.ax_error.xaxis.set_major_formatter(dates.DateFormatter("%d.%m.%Y %H:%M"))
        self.ax_error.xaxis.set_major_locator(MaxNLocator(10))

        plots_amount = []
        datetime_axis = tuple(datetime.datetime.utcfromtimestamp(x // 1000) for x in self.axis_time)

        for i, approximation in enumerate(approximations):
            self.ax_error.plot(datetime_axis, self.errors[i], alpha=.25)
            p_a, = self.ax_amount.plot(datetime_axis, self.amounts[i], label=f"{str(approximation):s}", alpha=1.)
            plots_amount.append(p_a)

        p_a, = self.ax_amount.plot(datetime_axis, self.ratio_list, label=f"market average")
        plots_amount.append(p_a)

        self.ax_error.set_ylim((0., 1.))

        val_min_amount = min(min(each_amounts) for each_amounts in self.amounts + (self.ratio_list,))
        val_max_amount = max(max(each_amounts) for each_amounts in self.amounts + (self.ratio_list,))
        self.ax_amount.set_ylim([val_min_amount - .2 * (val_max_amount - val_min_amount),  val_max_amount + .2 * (val_max_amount - val_min_amount)])

        pyplot.setp(self.ax_amount.xaxis.get_majorticklabels(), rotation=90, ha="right", rotation_mode="anchor")
        pyplot.legend(plots_amount, tuple(line.get_label() for line in plots_amount))
        pyplot.tight_layout()
        pyplot.pause(.05)


class ExperimentContinual(VisualizationMixin):
    def __init__(self, approximations: Sequence[Approximation], pairs_assets: Sequence[Tuple[str, str]], certainty_threshold: float, fee: float):
        super().__init__(len(approximations))
        time_range = 1532491200000, 1577836856000
        interval_minutes = 1

        self.fee = fee
        self.keys_rates = tuple("-".join(each_pair) + "_close" for each_pair in pairs_assets)
        self.approximations = tuple(
            ApproximationInvestmentQuantified(each_a, fee, certainty_threshold)
            for each_a in approximations
        )

        self.generator_snapshots = merge_generator(
            pairs_assets, timestamp_range=time_range, interval_minutes=interval_minutes
        )

    def _get_rates(self, snapshot: Dict[str, Any]) -> Sequence[float]:
        return tuple(snapshot[key] for key in self.keys_rates)

    def start(self):
        for t, snapshot in enumerate(self.generator_snapshots):
            timestamp = snapshot["close_time"]
            rates = self._get_rates(snapshot)

            for each_approximation in self.approximations:
                each_approximation.cycle(rates)

            self._update_market(rates, t)

            if Timer.time_passed(1000):
                self._update_plot(timestamp, self.approximations)
                self.t_last = t


class ExperimentPeriodic(ExperimentContinual):
    def __init__(self, approximations: Sequence[Approximation], pairs_assets: Sequence[Tuple[str, str]], certainty_threshold: float, fee: float):
        super().__init__(approximations, pairs_assets, certainty_threshold, fee)
        self.length_training = 60 * 24 * 7
        self.period_training = 60 * 24 * 7

    def train(self, rates: Iterable[Sequence[float]]):
        iterator_rates = (x for x in rates)
        matrix_change = generate_multiple_changes(iterator_rates)
        matrix_invest = generate_matrix(len(self.keys_rates), matrix_change, self.fee, bound=100)
        path_invest = generate_path(list(matrix_invest))

        for i, each_approximation in enumerate(self.approximations):
            print(f"starting batch training {i + 1:d} of {len(self.approximations):d}...")
            each_approximation.batch(list(zip(rates, path_invest)))

    def start(self):
        buffer_rates = []

        for t, snapshot in enumerate(self.generator_snapshots):
            timestamp = snapshot["close_time"]
            rates = self._get_rates(snapshot)
            buffer_rates.append(rates)
            del(buffer_rates[:-self.length_training])

            if (t + 1) % self.period_training == 0:
                self.train(buffer_rates)
                buffer_rates.clear()

            for each_approximation in self.approximations:
                each_approximation.cycle(rates, skip_train=True)

            self._update_market(rates, t)
            if 0. >= max(rates):
                raise ValueError("all assets at zero?!")

            if Timer.time_passed(1000):
                self._update_plot(timestamp, self.approximations)
                self.t_last = t


def main():
    random.seed(235456345)
    pairs = get_pairs_from_filesystem()
    pairs = random.sample(pairs, 5)
    print(pairs)

    no_assets = len(pairs)
    approximations = (
        MultivariatePolynomialRegression(no_assets, 2, no_assets),
        MultivariatePolynomialRecurrentRegression(
            no_assets, 2, no_assets,
            no_memories=1,
            error_memory=lambda output_val, target_val: float(max(enumerate(output_val), key=lambda x: x[1])[0] != max(enumerate(target_val), key=lambda x: x[1])[0]) * 1000.),
        MultivariatePolynomialRecurrentRegression(
            no_assets, 2, no_assets,
            no_memories=2,
            error_memory=lambda output_val, target_val: float(max(enumerate(output_val), key=lambda x: x[1])[0] != max(enumerate(target_val), key=lambda x: x[1])[0]) * 1000.),
        MultivariatePolynomialRecurrentRegression(
            no_assets, 2, no_assets,
            no_memories=3,
            error_memory=lambda output_val, target_val: float(max(enumerate(output_val), key=lambda x: x[1])[0] != max(enumerate(target_val), key=lambda x: x[1])[0]) * 1000.),
    )

    fee = .01
    certainty = 1.001
    e = ExperimentContinual(approximations, pairs, certainty, fee)
    # e = ExperimentPeriodic(approximations, pairs, certainty, fee)
    e.start()

    # todo: implement continual optimal trading as Experiment


if __name__ == "__main__":
    main()
