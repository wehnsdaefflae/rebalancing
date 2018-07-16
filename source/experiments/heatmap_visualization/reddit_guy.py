# https://www.reddit.com/r/Python/comments/87lbao/matplotlib_griddata_deprecation_help/

from numpy.random import uniform
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import numpy as np


def heat_plot(x, y, z):
    # mlab.griddata accepts the xi and yi 1d arrays above with different lengths.
    # scipy's griddata requires a full grid array. This is created with the mgrid function.
    print("\n".join(["({:.5f}, {:.5f}, {:.5f})".format(*each_p) for each_p in zip(x, y, z)]))
    xi, yi = np.mgrid[min(x):max(x):200j, min(y):(max(y)):200j]
    # grid the data.
    # points = np.vstack((x,y)).T
    zi = griddata((x, y), z, (xi, yi), method='cubic')
    # contour the gridded data, plotting dots at the nonuniform data points.
    plt.contour(xi, yi, zi, 15, linewidths=0.5, colors='k')
    plt.contourf(xi, yi, zi, 15)
    plt.colorbar()  # draw colorbar
    # plot data points.
    # plt.scatter(x, y, marker='o', s=5, zorder=10)
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.title('griddata test')
    plt.show()


if __name__ == "__main__":
    # make up data.
    npts = 200
    _x = uniform(-2, 2, npts)      # 200 random points between -2 to 2
    _y = uniform(-2, 2, npts)
    _z = _x * np.exp(-_x**2 - _y**2)
    heat_plot(_x, _y, _z)
