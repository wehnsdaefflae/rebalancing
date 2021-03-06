import datetime
import json
from typing import Iterable, Tuple, Type, Collection

from matplotlib import pyplot

from deprecated.data.data_processing import series_generator
from deprecated.tools.scipy_plot import heat_plot
from deprecated.tools.optimizer import StatefulOptimizer, SAMPLE
from source.tools.timer import Timer
from deprecated.tactics.signals.signals import TradingSignal, SymmetricChannelSignal

TIME_SERIES = Iterable[Tuple[datetime.datetime, float]]
PARAMETERS = Tuple[float, ...]


def evaluate_signal(signal: TradingSignal,
                    time_series: TIME_SERIES,
                    asset="asset", base="base",
                    plot: bool = False) -> float:
    time_axis = []
    rate_axis = []
    signal_axis = []
    total_value_axis = []
    value_base, value_asset = 1., 0.

    buys = []
    sells = []

    risk = .5
    tolerance = .5
    trading_factor = .9975
    min_trading_volume_base = .02

    for each_date, each_rate in time_series:
        tendency = signal.get_tendency(each_rate)
        if tendency >= tolerance:
            if 0. < value_base:
                amount = value_base * risk
                if amount >= min_trading_volume_base:
                    buys.append(each_date)
                    value_asset += trading_factor * amount / each_rate
                    value_base -= amount
        elif -tolerance >= tendency:
            if 0. < value_asset:
                amount = value_asset * risk
                if amount >= min_trading_volume_base:
                    sells.append(each_date)
                    value_base += trading_factor * amount * each_rate
                    value_asset -= amount

        time_axis.append(each_date)
        rate_axis.append(each_rate)
        signal_axis.append(tendency)
        total_value_axis.append(value_base + value_asset * each_rate)

    if plot:
        # TODO: align left and right y axis
        pyplot.clf()
        pyplot.close()

        fig, (ax1, ax3) = pyplot.subplots(2, sharex="all")
        ax1.plot(time_axis, rate_axis, label="{:s} rate".format(asset))
        ax1.set_ylabel("{:s} rate".format(asset))
        ax1.legend()

        ax2 = ax1.twinx()
        ax2.plot(time_axis, total_value_axis, label="total {:s} value".format(base), color="orange")
        ax2.set_ylabel("total {:s} value".format(base))
        ax2.legend()

        # signal.plot(time_axis, ax3, axis_label=asset)
        ax3.plot(time_axis, signal_axis, label="{:s} {:s}".format(asset, signal.__class__.__name__))
        ax3.set_ylabel("{:s} {:s}".format(asset, signal.__class__.__name__))
        ax3.axhline(y=tolerance, color="red")
        ax3.axhline(y=-tolerance, color="green")

        ax3.legend()

        for each_buy in buys:
            ax2.axvline(x=each_buy, color="red", alpha=.2)
        for each_sell in sells:
            ax2.axvline(x=each_sell, color="green", alpha=.2)

        pyplot.tight_layout()
        pyplot.show()

    return total_value_axis[-1]                     # against investment
    # return total_value[-1] / other_value[-1]  # against hodling


def sample_signal(signal_class: Type[TradingSignal],
                  time_series: TIME_SERIES,
                  parameter_ranges: Tuple[Tuple[float, float], ...],
                  no_samples: int,
                  plot: bool = False) -> Collection[SAMPLE]:

    sequence = list(time_series)

    def series_eval(parameter: float) -> float:
        signal = signal_class(round(parameter))
        return evaluate_signal(signal, sequence)

    optimizer = StatefulOptimizer(series_eval, parameter_ranges)
    samples = set()

    for i in range(no_samples):
        if Timer.time_passed(2000):
            print("Iteration {:d}/{:d}".format(i, no_samples))
        each_sample = optimizer.next()
        samples.add(each_sample)
        if plot:
            pyplot.plot(each_sample[0], [each_sample[1]], "x")

    print("Best parameters: {:s} (value: {:.4f})".format(str(optimizer.best_parameters), optimizer.best_value))
    if plot:
        axes = sorted(samples, key=lambda _x: _x[0])
        pyplot.plot(*zip(*axes))
        pyplot.show()

    return samples


def optimal_parameter_development(signal_class: Type[TradingSignal],
                                  trail_length: int,
                                  sampling_frequency: int,
                                  parameter_ranges: Tuple[Tuple[float, float], ...],
                                  time_series: TIME_SERIES,
                                  plot: bool = False):
    sequence = list(time_series)
    if trail_length >= len(sequence):
        raise ValueError("Trail length is too long for time series")

    # time_axis = [_x[0] for _x in sequence]
    time_axis = []
    parameter_axis = []
    value_axis = []
    for i in range(len(sequence) - trail_length):
        trail = sequence[i:i+trail_length]
        samples = sample_signal(signal_class, trail, parameter_ranges, sampling_frequency)
        for each_parameter, each_value in samples:
            parameter_axis.append(each_parameter[0])
            value_axis.append(each_value)
            time_axis.append(i)
        print("Finished {:d}/{:d} trails...".format(i, len(sequence) - trail_length))

    if plot:
        heat_plot(time_axis, [_x[0] for _x in sequence], parameter_axis, value_axis)


if __name__ == "__main__":
    with open("../../configs/time_series.json", mode="r") as file:
        config = json.load(file)

    # start_time = "2017-07-27 00:03:00 UTC"
    # end_time = "2018-06-22 23:52:00 UTC"
    start_time = "2017-08-01 00:00:00 UTC"
    end_time = "2017-08-02 00:00:00 UTC"
    interval_minutes = 1

    asset_symbol, base_symbol = "QTUM", "ETH"

    source_path = config["data_dir"] + "{:s}{:s}.csv".format(asset_symbol, base_symbol)
    series_generator = series_generator(source_path,
                                        start_time=start_time,
                                        end_time=end_time,
                                        interval_minutes=interval_minutes)

    evaluate_signal(SymmetricChannelSignal(76), list(series_generator)[:-24*7], asset=asset_symbol, base=base_symbol, plot=True)
    # optimize_signal(SymmetricChannelSignal, series_generator, ((1., 250), ), 2000, plot=True)

    #exit()
    #signal_classes = [HillValleySignal, SymmetricChannelSignal, AsymmetricChannelSignal, InvAsymmetricChannelSignal, RelativeStrengthIndexSignal]

    # one week: 1008 * 10 minutes, 8 * 120 min
    # one day: 144 * 10 minutes

    #optimal_parameter_development(signal_classes[4], 120, 50, ((1., 120.), ), series_generator, plot=True)
    # consider value in plot. close to 1 doesnt mean much

