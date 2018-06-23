import datetime

import dtw
import numpy
from matplotlib import pyplot

from source.main import absolute_brownian

#x_raw = [0, 0, 1, 1, 2, 4, 2, 1, 2, 0]
#y_raw = [1, 1, 1, 2, 2, 2, 2, 3, 2, 0]

#g = absolute_brownian(initial=1., factor=2., relative_bias=.1)
#_a = [next(g) for _ in range(11)]
#x_raw = _a[:10]
#y_raw = [_x - .1 for _x in _a][1:]

#x_raw = [float(_x) == 5 for _x in range(10)]
#y_raw = [float(_x) == 8 for _x in range(10)]
from source.my_dtw import get_series

start_date = datetime.datetime(2018, 5, 20, 0, 0, 0, tzinfo=datetime.timezone.utc)
# end_date = datetime.datetime(2018, 6, 20, 0, 0, 0, tzinfo=datetime.timezone.utc)
end_date = datetime.datetime(2018, 5, 27, 0, 0, 0, tzinfo=datetime.timezone.utc)

timestamp_start, timestamp_end = int(start_date.timestamp()), int(end_date.timestamp())
fp_a = "../data/binance/20May2018-20Jun2018-1m/{}.csv".format("ADAETH")
fp_b = "../data/binance/20May2018-20Jun2018-1m/{}.csv".format("ADXETH")
x_raw = get_series(fp_a, range_start=timestamp_start, range_end=timestamp_end)[10000:]
y_raw = get_series(fp_b, range_start=timestamp_start, range_end=timestamp_end)[:-10000]

x = numpy.array(x_raw).reshape(-1, 1)
y = numpy.array(y_raw).reshape(-1, 1)

pyplot.plot(x)
pyplot.plot(y)
pyplot.show()

dist, cost, acc, path = dtw.dtw(x, y, dist=numpy.linalg.norm)

print('Minimum distance found:', dist)

pyplot.imshow(acc.T, origin='lower', interpolation='nearest')
pyplot.plot(path[0], path[1], 'w')
pyplot.xlim((-0.5, acc.shape[0]-0.5))
pyplot.ylim((-0.5, acc.shape[1]-0.5))
pyplot.show()

print(path)


def my_custom_norm(a, b):
    return (a * a) + (b * b)


dist, cost, acc, path = dtw.dtw(x, y, dist=my_custom_norm)
