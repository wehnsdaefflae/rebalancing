import datetime
import json
from typing import Iterable, Tuple, Type

from matplotlib import pyplot

from source.data.data_generation import series_generator
from source.experiments.optimizer.my_optimizer import MyOptimizer
from source.experiments.timer import Timer
from source.tactics.signals.signals import TradingSignal, RelativeStrengthIndexSignal, SymmetricChannelSignal, \
    AsymmetricChannelSignal, HillValleySignal

TIME_SERIES = Iterable[Tuple[datetime.datetime, float]]
PARAMETERS = Tuple[float, ...]


def evaluate_signal(signal: TradingSignal,
                    time_series: TIME_SERIES,
                    asset="asset", base="base",
                    plot: bool = False) -> float:
    time_axis = []
    signal_axis = []
    value_base, value_asset = 1., 0.

    buys = []
    sells = []
    total_base_value = []
    all_asset_value = []
    amount_asset = -1.
    tolerance = .1
    trading_factor = .9975
    min_trading_volume_base = .02

    for each_date, each_rate in time_series:
        tendency = signal.get_tendency(each_rate)
        if tendency >= tolerance:
            if 0. < value_base:
                amount = value_base
                if amount >= min_trading_volume_base:
                    buys.append(each_date)
                    value_asset += trading_factor * amount / each_rate
                    value_base = 0.
        elif -tolerance >= tendency:
            if 0. < value_asset:
                amount = value_asset * each_rate
                if amount >= min_trading_volume_base:
                    sells.append(each_date)
                    value_base += trading_factor * amount
                    value_asset = 0.

        if amount_asset < 0.:
            # only on first iteration: change initial base asset into cur
            amount_asset = 1. / each_rate

        all_asset_value.append(amount_asset * each_rate)
        time_axis.append(each_date)
        signal_axis.append(tendency)
        total_base_value.append(value_base + value_asset * each_rate)

    if plot:
        pyplot.clf()
        pyplot.close()

        fig, (ax1, ax2, ax3) = pyplot.subplots(3, sharex="all")
        signal.plot(time_axis, ax1, axis_label=asset)
        ax2.plot(time_axis, signal_axis)
        ax2.set_ylabel("signal")
        ax3.plot(time_axis, all_asset_value, label="{:s} value".format(asset))
        ax3.plot(time_axis, total_base_value, label="total value")
        ax3.set_ylabel("total {:s} value".format(base))
        ax3.legend()

        for each_buy in buys:
            ax1.axvline(x=each_buy, color="red", alpha=.2)
            ax2.axvline(x=each_buy, color="red", alpha=.2)
            ax3.axvline(x=each_buy, color="red", alpha=.2)
        for each_sell in sells:
            ax1.axvline(x=each_sell, color="green", alpha=.2)
            ax2.axvline(x=each_sell, color="green", alpha=.2)
            ax3.axvline(x=each_sell, color="green", alpha=.2)

        pyplot.tight_layout()
        pyplot.show()

    return total_base_value[-1]                      # against investment
    # return total_value[-1] / other_value[-1]  # against hodling


def optimize_signal(signal_class: Type[TradingSignal],
                    time_series: TIME_SERIES,
                    plot: bool = True) -> Tuple[PARAMETERS, float]:

    def series_eval(parameter: float) -> float:
        signal = signal_class(round(parameter))
        return evaluate_signal(signal, time_series)

    optimizer = MyOptimizer(series_eval, ((1., 1000.),))
    axes = []

    samples = 50

    for i in range(samples):
        if Timer.time_passed(2000):
            print("Iteration {:d}/{:d}".format(i, samples))
        c = optimizer.next()
        each_value = series_eval(*c)
        point = c[0], each_value
        axes.append(point)
        if plot:
            pyplot.plot(c, [each_value], "x")

    print("Best parameters: {:s} (value: {:.4f})".format(str(optimizer.best_parameters), optimizer.best_value))
    if plot:
        axes = sorted(axes, key=lambda _x: _x[0])
        pyplot.plot(*zip(*axes))
        pyplot.show()
    return optimizer.best_parameters, optimizer.best_value


def optimal_parameter_development(signal_class: Type[TradingSignal], trail_length: int, time_series: TIME_SERIES):
    sequence = list(time_series)
    if trail_length >= len(sequence):
        raise ValueError("Trail length is too long for time series")

    time_axis = [_x[0] for _x in sequence]

    optimal_parameter_axis = []
    for i in range(len(sequence) - trail_length):
        max_parameters, max_value = optimize_signal(signal_class, sequence[i:i+trail_length], plot=False)
        optimal_parameter_axis.append(max_parameters[0])
        print("Finished {:d}/{:d} trails...".format(i, len(sequence) - trail_length))

    pyplot.plot(time_axis[trail_length:], optimal_parameter_axis)
    pyplot.show()


if __name__ == "__main__":
    with open("../../configs/time_series.json", mode="r") as file:
        config = json.load(file)

    # start_time = "2017-07-27 00:03:00 UTC"
    # end_time = "2018-06-22 23:52:00 UTC"
    start_time = "2017-07-27 00:03:00 UTC"
    end_time = "2017-08-27 23:52:00 UTC"
    interval_minutes = 120

    asset_symbol, base_symbol = "QTUM", "ETH"

    source_path = config["data_dir"] + "{:s}{:s}.csv".format(asset_symbol, base_symbol)
    series_generator = series_generator(source_path,
                                        start_time=start_time,
                                        end_time=end_time,
                                        interval_minutes=interval_minutes)

    evaluate_signal(SymmetricChannelSignal(50), series_generator, asset=asset_symbol, base=base_symbol, plot=True)

    exit()

    # signal = FakeSignal([_x[1] for _x in DEBUG_SERIES(cur, config_path="../../../configs/config.json")])
    signal_classes = [HillValleySignal, SymmetricChannelSignal, AsymmetricChannelSignal, RelativeStrengthIndexSignal]

    # optimize_signal(None, list(data_generator))
    # evaluate_parameter(3250, data_generator, plot=True)
    optimal_parameter_development(signal_classes[0], 8, list(series_generator))
    # one week: 1008 * 10 minutes, 8 * 120 min
