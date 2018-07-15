import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt


def heat_plot(_X, _Y, _T, _f):
    # Choose npts random point from the discrete domain of our model function
    npts = 400
    px = np.random.choice(x, npts)
    py = np.random.choice(y, npts)

    fig, ax = plt.subplots(1)
    # Plot the model function and the randomly selected sample points
    # ax.contourf(_X, _Y, _T)

    Ti = griddata((px, py), _f(px, py), (_X, _Y), method='cubic')
    ax.contourf(_X, _Y, Ti)
    ax.set_title('method = {}'.format("cubic"))
    ax.scatter(px, py, c='k', alpha=0.2, marker='.')

    plt.show()


if __name__ == "__main__":
    x = np.linspace(-1,1,100)
    y =  np.linspace(-1,1,100)

    X, Y = np.meshgrid(x, y)

    def f(x, y):
        s = np.hypot(x, y)
        phi = np.arctan2(y, x)
        tau = s + s*(1-s)/5 * np.sin(6*phi)
        return 5*(1-tau) + tau
    T = f(X, Y)

    heat_plot(X, Y, T, f)
