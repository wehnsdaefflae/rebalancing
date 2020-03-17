import glob
import os
from typing import Sequence, Tuple

from matplotlib import pyplot, colors

from source.data.merge_csv import merge_generator
from source.tools.timer import Timer


def template():
    # https://stackoverflow.com/questions/7684475/plotting-labeled-intervals-in-matplotlib-gnuplot
    from matplotlib import pyplot
    from matplotlib.dates import DateFormatter, MinuteLocator, SecondLocator
    import numpy as np
    import datetime as dt

    ### The example data
    a=StringIO("""a 10:15:22 10:15:30 OK
    b 10:15:23 10:15:28 OK
    c 10:16:00 10:17:10 FAILED
    b 10:16:30 10:16:50 OK
    """)

    #Converts str into a datetime object.
    conv = lambda s: dt.datetime.strptime(s, '%H:%M:%S')

    #Use numpy to read the data in.
    data = np.genfromtxt(a, converters={1: conv, 2: conv},
                         names=['caption', 'start', 'stop', 'state'], dtype=None)
    cap, start, stop = data['caption'], data['start'], data['stop']

    #Check the status, because we paint all lines with the same color
    #together
    is_ok = (data['state'] == 'OK')
    not_ok = np.logical_not(is_ok)

    #Get unique captions and there indices and the inverse mapping
    captions, unique_idx, caption_inv = np.unique(cap, 1, 1)

    #Build y values from the number of unique captions.
    y = (caption_inv + 1) / float(len(captions) + 1)

    #Plot function
    def timelines(y, xstart, xstop, color='b'):
        """Plot timelines at y from xstart to xstop with given color."""
        pyplot.hlines(y, xstart, xstop, color, lw=4)
        pyplot.vlines(xstart, y+0.03, y-0.03, color, lw=2)
        pyplot.vlines(xstop, y+0.03, y-0.03, color, lw=2)

    #Plot ok tl black
    timelines(y[is_ok], start[is_ok], stop[is_ok], 'k')
    #Plot fail tl red
    timelines(y[not_ok], start[not_ok], stop[not_ok], 'r')

    #Setup the plot
    ax = pyplot.gca()
    ax.xaxis_date()
    myFmt = DateFormatter('%H:%M:%S')
    ax.xaxis.set_major_formatter(myFmt)
    ax.xaxis.set_major_locator(SecondLocator(interval=20)) # used to be SecondLocator(0, interval=20)

    #To adjust the xlimits a timedelta is needed.
    delta = (stop.max() - start.min())/10

    pyplot.yticks(y[unique_idx], captions)
    pyplot.ylim(0,1)
    pyplot.xlim(start.min()-delta, stop.max()+delta)
    pyplot.xlabel('Time')
    pyplot.show()


def get_intervals(names_assets: Sequence[str]) -> Tuple[Sequence[Tuple[int, int]], ...]:
    pairs_asset = tuple((x[:-3], x[-3:]) for x in names_assets)
    no_assets = len(pairs_asset)

    intervals = tuple([] for _ in range(no_assets))
    state_interval_open = [False for _ in intervals]

    last_timestamp = -1
    generator = merge_generator(pairs=pairs_asset, header=("close_time", "close"))
    for t, data in enumerate(generator):
        for i, each_data in enumerate(data):
            each_timestamp = each_data[0]
            each_rate = each_data[1]
            intervals_all = intervals[i]
            if state_interval_open[i] and each_rate < 0.:
                interval_last = intervals_all[-1]
                interval_last.append(each_timestamp)
                state_interval_open[i] = False

            elif state_interval_open[i] and each_rate >= 0.:
                # print()
                pass

            elif not state_interval_open[i] and each_rate < 0.:
                # print()
                pass

            elif not state_interval_open[i] and each_rate >= 0.:
                interval_last = [each_timestamp]
                intervals_all.append(interval_last)
                state_interval_open[i] = True

            last_timestamp = each_timestamp

        if Timer.time_passed(2000):
            print(f"finished processing {t:d} data points...")

    for intervals_all in intervals:
        for interval_each in intervals_all:
            if len(interval_each) == 1:
                interval_each.append(last_timestamp
                                     )
    return intervals


def plot_intervals(intervals: Sequence[Sequence[Tuple[int, int]]], names: Sequence[str]):
    assert len(intervals) == len(names)
    colors_all = list(colors.CSS4_COLORS.keys())
    fig, ax = pyplot.subplots()

    labelled = [False for _ in names]

    def add_interval(start: int, end: int, index_asset: int):
        c = colors_all[index_asset]
        c = "black"
        if labelled[index_asset]:
            ax.hlines(i + 1, start, end, c, lw=4)
        else:
            ax.hlines(i + 1, start, end, c, lw=4, label=names[index_asset])
            labelled[index_asset] = True
        ax.vlines(start, i + 1. + .03, i + 1. - .03, c, lw=2)
        ax.vlines(end, i + 1 + .03, i + 1 - .03, c, lw=2)

    for i, intervals_all in enumerate(intervals):
        for interval_each in intervals_all:
            add_interval(interval_each[0], interval_each[1], i)

    ax.ticklabel_format(style="plain")

    pyplot.yticks(range(1, len(names) + 1), names)
    # pyplot.legend()
    pyplot.show()


def main():
    directory_data = "../../data/"
    directory_csv = directory_data + "binance/"

    files = glob.glob(directory_csv + "*.csv")
    names_assets = sorted(os.path.splitext(os.path.basename(x))[0] for x in files)[:3]

    intervals = get_intervals(names_assets)
    plot_intervals(intervals, names_assets)


if __name__ == "__main__":
    main()
