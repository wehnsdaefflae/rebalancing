import datetime
import os
import random
from typing import Callable

from matplotlib import pyplot
# random.seed(2353534234563)
from source.applications.trading_bots import PORTFOLIO_INFO


def absolute_brownian(initial=1., factor=1., relative_bias=0.):  # constant equiprobable change
    a = initial
    while True:
        yield a
        if 0. < a:
            rnd_value = 2. * factor * random.random() - factor + relative_bias * factor
            a = max(a + rnd_value / 100., .0)


def relative_brownian(initial=1., factor=.2, min_limit=.05):  # relative normally distributed change
    a = initial
    while True:
        yield a
        if 0. < a:
            rnd_value = max(a * (1. + random.gauss(0.1, factor / 4.)), .0)
            a = rnd_value if min_limit * initial < rnd_value else 0.


def random_generator(symbols, relative=False):
    i = 0
    pairs = [absolute_brownian() if relative else absolute_brownian() for _ in symbols]
    while True:
        rands = {"time": i}
        rands.update({x: next(r_) for x, r_ in zip(symbols, pairs)})
        yield rands


def data_generator(file_path, select=None, date_start=None, date_end=None):
    with open(file_path, mode="r") as file:
        first_line = file.readline()
        cells = first_line[:-1].split("\t")
        indices = sorted(cells.index(x) for x in select)
        symbols = [cells[x] for x in indices]
        for each_line in file:
            cells = each_line[:-1].split("\t")
            date = datetime.datetime.strptime(cells[0], "%Y-%m-%d")
            rates = [float(cells[x]) for x in indices]
            if (date_start is None or date >= date_start) and (date_end is None or date_end >= date):
                row = {"time": date}
                row.update({x: r_ for x, r_ in zip(symbols, rates)})
                yield row


def redistribute_assets(assets, rates, ratios, buffer_cur, trading):
    total_value = sum(assets.get(x, .0) * rates.get(x, .0) for x in assets)
    buffer_rate = rates.get(buffer_cur, .0)

    for each_asset in assets:
        asset_rate = rates.get(each_asset, 0.)
        if each_asset == buffer_cur or 0. >= buffer_rate:
            continue
        elif 0. >= asset_rate:
            ratios[each_asset] = 0.
            s = sum(ratios.values())
            for x in ratios:
                ratios[x] = ratios.get(x, 0.) / s
            continue

        current_value = assets.get(each_asset, 0.) * asset_rate
        target_value = total_value * ratios.get(each_asset, 0.)
        difference_value = current_value - target_value
        difference_buffer = 0. if 0. >= buffer_rate else difference_value / buffer_rate

        if difference_buffer >= .001 * trading:  # min btc = 0.001
            delta = 0. if 0 >= asset_rate else difference_value * trading / asset_rate
            assets[each_asset] = assets.get(each_asset, 0.) - delta
            assets[buffer_cur] = assets.get(buffer_cur, 0.) + difference_buffer

    for each_asset in assets:
        asset_rate = rates.get(each_asset, 0.)
        if 0. >= asset_rate or 0. >= buffer_rate:
            continue

        current_value = assets.get(each_asset, 0.) * rates.get(each_asset, 0.)
        target_value = total_value * ratios.get(each_asset, 0.)
        difference_value = target_value - current_value
        difference_buffer = 0. if 0. >= buffer_rate else difference_value / buffer_rate

        if difference_buffer >= .001 * trading:
            delta = 0. if 0. >= asset_rate else difference_value / asset_rate
            assets[buffer_cur] = assets.get(buffer_cur, 0.) - difference_buffer * trading
            assets[each_asset] = assets.get(each_asset, 0.) + delta


def rebalancing(generator, generator_parameters, ratios, buffer_cur, initial_assets, rebalancing_interval=500, trading_cost=.0025, duration=-1, plot="yes"):
    data_source = generator(**generator_parameters)
    ratio_sum = sum(ratios.values())
    ratios = {x: y / ratio_sum for x, y in ratios.items()}

    trading_inv = 1. + trading_cost

    assets = {x: initial_assets.get(x, 0.) for x in set(initial_assets.keys()) | {buffer_cur}}
    values = {x: [] for x in set(ratios.keys()) | {buffer_cur}}

    hodl_assets = {x: initial_assets.get(x, 0.) for x in set(initial_assets.keys()) | {buffer_cur}}
    hodl_values = {x: [] for x in set(ratios.keys()) | {buffer_cur}}

    v_lines = []
    X = []
    for i, rates in enumerate(data_source):
        if 0 < duration <= i:
            break

        if i < 1:
            redistribute_assets(hodl_assets, rates, ratios, buffer_cur, trading_inv)

        if i % rebalancing_interval == 0:
            each_values = {x: rates.get(x, 0.) * assets.get(x, 0.) for x in assets}
            each_hodl_values = {x: rates.get(x, 0.) * hodl_assets.get(x, 0.) for x in assets}
            X.append(i)
            for v in assets:
                values[v].append(each_values[v])
                hodl_values[v].append(each_hodl_values[v])

            redistribute_assets(assets, rates, ratios, buffer_cur, trading_inv)
            v_lines.append(i)

        each_values = {x: rates.get(x, 0.) * assets.get(x, 0.) for x in assets}
        each_hodl_values = {x: rates.get(x, 0.) * hodl_assets.get(x, 0.) for x in assets}
        X.append(i)
        for v in assets:
            values[v].append(each_values[v])
            hodl_values[v].append(each_hodl_values[v])

    r_, h_ = sum(x[-1] for x in values.values()), sum(x[-1] for x in hodl_values.values())
    print("{:5.2f}%".format(0. if 0. >= h_ else r_ * 100. / h_))
    if plot == "yes":
        plot_strategy_comparison(X, hodl_values, values, v_lines)
    elif plot[-1] == "/":
        no_files = len([x for x in os.listdir(plot) if os.path.isfile(plot + x)])
        plot_strategy_comparison(X, hodl_values, values, v_lines, target_path=plot + "{:03.0f}_{:06d}.png".format(100. * r_ / h_, no_files))
    elif plot != "no":
        plot_strategy_comparison(X, hodl_values, values, v_lines, target_path=plot)

    return r_, h_


def plot_strategy_comparison(time, hodl_values, values, v_lines, target_path=None):
    max_value = max(sum(x[-1] for x in values.values()), sum(x[-1] for x in hodl_values.values()))

    symbols = sorted(values.keys())

    f, (ax1, ax2) = pyplot.subplots(2, 1, sharey="all", sharex="all")
    for d in v_lines:
        ax1.axvline(x=d, linewidth=1, color="green", alpha=.2)

    re_stack = [values[x] for x in symbols]
    ax1.stackplot(time, *re_stack, labels=symbols)
    ax1.set_ylabel("value")
    ax1.set_xlabel("time")
    ax1.axhline(y=max_value)
    ax1.set_title("rebalancing")

    hd_stack = [hodl_values[x] for x in symbols]
    ax2.stackplot(time, *hd_stack, labels=symbols)
    ax2.set_ylabel("value")
    ax2.set_xlabel("time")
    ax2.axhline(y=max_value)
    ax2.set_title("hodling")
    ax2.legend()
    if target_path is None:
        pyplot.show()
    else:
        pyplot.savefig(target_path)
    pyplot.close()


def strategy_evaluation(parameters, iterations=1000):
    p = "../results/"
    r_data = []
    h_data = []
    iterate = range(iterations)
    for i in iterate:
        print("{:04.1f}% finished".format(i * 100. / iterations))
        rebalance_eval, hodl_eval = rebalancing(**parameters, plot=p)
        r_data.append(rebalance_eval)
        h_data.append(hodl_eval)

    for i in range(1, len(r_data)):
        r_data[i] += r_data[i - 1]
    for i in range(1, len(h_data)):
        h_data[i] += h_data[i - 1]

    pyplot.plot(iterate, r_data, label="rebalancing")
    pyplot.plot(iterate, h_data, label="hodling")
    pyplot.title(", ".join(["{}: {}".format(k, v) for k, v in parameters.items()]))
    print("performance: {:05.1f}%".format(r_data[-1]*100. / h_data[-1]))
    pyplot.legend()
    pyplot.savefig(p + "total.png")


if __name__ == "__main__":
    data_parameters = {"symbols": ["{:03d}".format(x) for x in range(5)], "relative": False}
    data = random_generator

    """
    data_parameters = {"file_path": "../data/stock/all_closes.csv",
                       "select": {"ge", "mmm", "pg", "utx", "xom"},
                       "date_start": datetime.datetime.strptime("2000-01-02", "%Y-%m-%d")}

    """
    """
    data_parameters = {"file_path": "../data/stock/all_closes.csv",
                       "select": {"BTC", "ETH", "DASH", "LTC"},
                       "date_start": datetime.datetime.strptime("2017-05-04", "%Y-%m-%d")}
    """

    # data = data_generator

    setting = {"generator": data,
               "generator_parameters": data_parameters,
               "ratios": {"000": 1., "001": 1., "002": 1., "003": 1., "004": 1.},
               "buffer_cur": "000",
               "initial_assets": {"000": 100., "001": 0., "002": 0., "003": 0., "004": 0.},
               "rebalancing_interval": 7, "trading_cost": .0025, "duration": 10000}
    # strategy_evaluation(setting, iterations=1000)
    r, h = rebalancing(**setting)  # , plot="../results/1.png")

