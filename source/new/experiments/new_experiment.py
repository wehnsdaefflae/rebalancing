import random
import time
from typing import Sequence, Optional, Any, Tuple, Generator

from matplotlib import pyplot

from source.new.data.binance_examples import get_pairs_from_filesystem
from source.new.data.snapshot_generation import merge_generator
from source.new.learning.approximation import Approximation
from source.new.learning.regression import MultivariatePolynomialRegression
from source.new.learning.tools import ratio_generator_multiple, smear


class MovingGraph:
    def __init__(self, names_plots: Sequence[str], size_window: int, alpha_values: Optional[Sequence[float]], interval_ms: int = 1000):
        assert 5000. >= interval_ms >= 0.
        self.names_plots = names_plots
        self.alpha_value = tuple(1. for _ in names_plots) if alpha_values is None else alpha_values

        self.interval_ms = interval_ms

        self.no_plots = len(names_plots)

        self.time = None
        self.values = [0. for _ in names_plots]

        self.time_range = []
        self.plots = tuple([] for _ in names_plots)

        self.size_window = size_window
        self.fig, self.ax = pyplot.subplots()
        # self.ax_error = self.ax.twinx()

        # todo: add second axis, add time axis format

        self.iterations_since_draw = 0

        self.time_last = -1.

    def add_snapshot(self, now: Any, points: Sequence[float]):
        assert len(points) == self.no_plots
        for i, (each_value, each_point) in enumerate(zip(self.values, points)):
            self.values[i] = smear(each_value, each_point, self.iterations_since_draw)
        self.time = now
        self.iterations_since_draw += 1

        time_now = time.time() * 1000.
        if self.time_last < 0. or time_now - self.time_last >= self.interval_ms:
            self.draw()
            self.time_last = time_now

    def draw(self):
        self.ax.clear()

        self.time_range.append(self.time)
        del(self.time_range[:-self.size_window])

        for i, (each_name, each_plot, each_value) in enumerate(zip(self.names_plots, self.plots, self.values)):
            each_plot.append(each_value)
            del(each_plot[:-self.size_window])
            self.ax.plot(self.time_range, each_plot, label=f"{each_name:s}", alpha=self.alpha_value[i])

        val_min = min(min(each_plot) for each_plot in self.plots)
        val_max = max(max(each_plot) for each_plot in self.plots)
        val_d = .2 * (val_max - val_min)
        self.ax.set_ylim([val_min - val_d,  val_max + val_d])

        pyplot.legend()
        pyplot.tight_layout()
        pyplot.pause(.05)

        self.iterations_since_draw = 0


class ApproximationQuantified:
    def __str__(self) -> str:
        raise NotImplementedError()

    def cycle(self, input_value: Sequence[float], target_value: Sequence[float]) -> Sequence[float]:
        raise NotImplementedError()


class Experiment:
    def __int__(self, approximations: Sequence[ApproximationQuantified]):
        self.approximations = approximations
        self.iteration = 0

    def _examples(self) -> Generator[Tuple[Any, Sequence[float], Sequence[float]], None, None]:
        raise NotImplementedError()

    def _apply(self, key: Any, input_value: Sequence[float], target_value: Sequence[float], results: Sequence[Sequence[float]]):
        pass

    def start(self):
        generator_examples = self._examples()
        for key, input_value, target_value in generator_examples:
            results = tuple(
                each_approximation.cycle(input_value, target_value)
                for each_approximation in self.approximations
            )
            self._apply(key, input_value, target_value, results)

            self.iteration += 1


class Investor(ApproximationQuantified):
    def __init__(self, approximation: Approximation[Sequence[float]], no_assets: int, fees: float):
        self.approximation = approximation
        self.asset_current = -1
        self.amount_current = 1.
        self.after_fee = 1. - fees

        self.ratio_generator = ratio_generator_multiple(no_assets)
        next(self.ratio_generator)

        self.error = 1.
        self.ratios_last = None
        self.iteration = 0

    def __str__(self) -> str:
        return str(self.approximation)

    def _learn(self, ratios: Sequence[float]):
        output_value = self.approximation.output(self.ratios_last)
        asset_output, _ = max(enumerate(output_value), key=lambda x: x[1])
        asset_target, _ = max(enumerate(ratios), key=lambda x: x[1])
        self.error = float(asset_output != asset_target)
        self.approximation.fit(self.ratios_last, ratios, self.iteration)
        self.iteration += 1

    def _invest(self, ratios: Sequence[float]):
        output_value = self.approximation.output(ratios)
        asset_output, _ = max(enumerate(output_value), key=lambda x: x[1])

        if asset_output != self.asset_current:
            self.amount_current *= self.after_fee
            self.asset_current = asset_output

        self.amount_current *= ratios[self.asset_current]

    def cycle(self, _: Sequence[float], rates: Sequence[float]) -> Sequence[float]:
        ratios = self.ratio_generator.send(rates)
        if ratios is not None:
            self._invest(ratios)

            if self.ratios_last is not None:
                self._learn(ratios)

            self.ratios_last = ratios

        return self.error, self.amount_current


class ExperimentMarket(Experiment):
    def __init__(self, approximations: Sequence[ApproximationQuantified], no_assets: int, visualize: bool = True):
        super().__int__(approximations)
        self.graph = None
        if visualize:
            names_graphs_amounts = [f"{str(each_approximation):s} amount" for each_approximation in approximations]
            names_graphs_errors = [f"{str(each_approximation):s} error" for each_approximation in approximations]
            name_market_average = ["market"]
            alpha_graphs = tuple(1. for _ in approximations) + tuple(.2 for _ in approximations) + (1.,)
            self.graph = MovingGraph(names_graphs_amounts + names_graphs_errors + name_market_average, 1000, alpha_graphs)
        self.no_assets = no_assets
        self.market_average = 1.

    def _examples(self) -> Generator[Tuple[Any, Sequence[float], Sequence[float]], None, None]:
        pairs = get_pairs_from_filesystem()
        pairs = random.sample(pairs, self.no_assets)

        keys_rates = tuple("-".join(each_pair) + "_close" for each_pair in pairs)

        time_range = 1532491200000, 1577836856000
        interval_minutes = 1

        generator_snapshots = merge_generator(pairs, timestamp_range=time_range, interval_minutes=interval_minutes)

        ratios_rates = ratio_generator_multiple(self.no_assets)
        next(ratios_rates)

        self.ratio_last = None
        for each_snapshot in generator_snapshots:
            if each_snapshot is None:
                continue

            rates = tuple(each_snapshot[key] for key in keys_rates)
            ratio = ratios_rates.send(rates)
            if self.ratio_last is not None:
                yield each_snapshot["close_time"], self.ratio_last, ratio

            self.ratio_last = ratio

    def _apply(self, key: Any, ratios_last: Sequence[float], ratios_now: Sequence[float], results: Sequence[Sequence[float]]):
        self.market_average *= sum(ratios_now) / self.no_assets
        if self.graph is not None:
            errors, amounts = zip(*results)
            self.graph.add_snapshot(key, amounts + errors + (self.market_average, ))


if __name__ == "__main__":
    # todo: add second axis, add time axis format
    # todo: refactor other experiments (mixin classes)
    # todo: compare greedy with dynamic programming (no learning!)
    # todo: reinforcement learning
    # todo: test failure regression
    # todo: normalize output?
    # todo: equidistant sampling

    no_assets_market = 3
    fee = .1
    approximations_market = MultivariatePolynomialRegression(no_assets_market, 3, no_assets_market), MultivariatePolynomialRegression(no_assets_market, 2, no_assets_market)
    m = ExperimentMarket(tuple(Investor(each_approximation, no_assets_market, fee) for each_approximation in approximations_market), no_assets_market)
    m.start()
