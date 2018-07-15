"""
Comparison of griddata and tricontour for an unstructured triangular grid.
"""
from matplotlib import tri, pyplot
import numpy
import matplotlib.mlab as mlab


def heat_plot(x, y, z):
    ngridx = 200
    ngridy = 200

    xi = numpy.linspace(-2.1, 2.1, ngridx)
    yi = numpy.linspace(-2.1, 2.1, ngridy)
    zi = mlab.griddata(numpy.ndarray(x), y, z, xi, yi, interp='linear')

    # tricontour.
    triang = tri.Triangulation(x, y)
    pyplot.tricontour(x, y, z, 15, linewidths=0.5, colors='k')
    pyplot.tricontourf(x, y, z, 15,
                       norm=pyplot.Normalize(vmax=abs(zi).max(), vmin=-abs(zi).max()))
    pyplot.colorbar()
    pyplot.plot(x, y, 'ko', ms=3)
    pyplot.xlim(-2, 2)
    pyplot.ylim(-2, 2)
    pyplot.title('tricontour')

    pyplot.subplots_adjust(hspace=0.5)

    pyplot.show()


if __name__ == "__main__":
    numpy.random.seed(0)
    _npts = 200

    _x = numpy.random.uniform(-2, 2, _npts)
    _y = numpy.random.uniform(-2, 2, _npts)
    _z = _x * numpy.exp(-_x ** 2 - _y ** 2)

    heat_plot(_x, _y, _z)
