# https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.griddata.html
from matplotlib import cm, colors
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import numpy as np
import yt


class MidpointNormalize(colors.Normalize):
    """
        http://chris35wills.github.io/matplotlib_diverging_colorbar/
        Normalise the colorbar so that diverging bars work there way either side from a prescribed midpoint value)
        e.g. im=ax1.imshow(array, norm=MidpointNormalize(midpoint=0.,vmin=-100, vmax=100))
    """

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))


def heat_plot(X_in, times, Y_in, values_in):
    # https://stackoverflow.com/questions/45787354/remove-boxes-around-imshow-when-sharing-x-axis

    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.griddata.html
    _X = np.array(X_in)
    _Y = np.array(Y_in)
    values = np.array(values_in)

    s = len(set(_X))
    grid_x, grid_y = np.meshgrid(np.linspace(min(_X), max(_X), s), np.linspace(min(_Y), max(_Y), max(_Y) - min(_Y)))

    methods = ["nearest", "linear", "cubic"]

    _p = np.array(list(zip(_X, _Y)))

    grid_z = griddata(_p, values, (grid_x, grid_y), method=methods[1])

    plt.imshow(grid_z, origin='lower', extent=(0, s, min(_Y), max(_Y)), aspect="auto", cmap="Pastels",
               norm=MidpointNormalize(midpoint=1.00001, vmin=min(values), vmax=max(values)))

    #plt.contour(_X, _Y, values, 15, linewidths=0.5, colors='k')
    #plt.contourf(_X, _Y, values, 15)

    max_values = dict()
    for each_x, each_y, each_z in zip(_X, _Y, values):
        old_v = max_values.get(each_x)
        if old_v is None or old_v[1] < each_z:
            max_values[each_x] = each_y, each_z

    max_x, max_y = zip(*[(k, v[0]) for k, v in max_values.items()])
    # plt.plot(max_x, max_y, color="black")

    plt.colorbar()
    #plt.plot(_X, _Y, 'k.', markersize=".5", alpha=.2)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    points = np.random.rand(1000, 2)
    X, Y = points[:, 0], points[:, 1]

    def func(x, y):
        return x*(1-x)*np.cos(4*np.pi*x) * np.sin(4*np.pi*y**2)**2

    Z = func(X, Y)
    heat_plot(X, Y, Z)

