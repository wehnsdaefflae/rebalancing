import numpy

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as LA


# https://sourceforge.net/p/geoms2/wiki/Kriging/

def heat_plot(_x, _y, _z):

    def SK(x, y, v, variogram, grid):
        cov_angulos = np.zeros((x.shape[0], x.shape[0]))
        cov_distancias = np.zeros((x.shape[0], x.shape[0]))
        K = np.zeros((x.shape[0], x.shape[0]))
        for i in range(x.shape[0] - 1):
            cov_angulos[i,i:]=np.arctan2((y[i:]-y[i]),(x[i:]-x[i]))
            cov_distancias[i,i:]=np.sqrt((x[i:]-x[i])**2+(y[i:]-y[i])**2)
        for i in range(x.shape[0]):
            for j in range(x.shape[0]):
                if cov_distancias[i,j]!=0:
                    amp=np.sqrt((variogram[1]*np.cos(cov_angulos[i,j]))**2+(variogram[0]*np.sin(cov_angulos[i,j]))**2)
                    K[i,j]=v[:].var()*(1-np.e**(-3*cov_distancias[i,j]/amp))
        K = K + K.T

        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                distancias = np.sqrt((i-x[:])**2+(j-y[:])**2)
                angulos = np.arctan2(i-y[:],j-x[:])
                amplitudes = np.sqrt((variogram[1]*np.cos(angulos[:]))**2+(variogram[0]*np.sin(angulos[:]))**2)
                M = v[:].var()*(1-np.e**(-3*distancias[:]/amplitudes[:]))
                W = LA.solve(K,M)
                grid[i,j] = np.sum(W*(v[:]-v[:].mean()))+v[:].mean()
        return grid

    Grid = np.zeros((100, 100),dtype='float32') # float32 gives us a lot precision
    Grid = SK(numpy.array(_x), numpy.array(_y), numpy.array(_z), (50, 30), Grid)
    plt.imshow(Grid.T,origin='lower', interpolation='nearest', cmap='jet')
    plt.scatter(_x, _y, c=_z, cmap='jet', s=120)
    plt.xlim(0, Grid.shape[0])
    plt.ylim(0, Grid.shape[1])
    plt.grid()
    plt.show()


if __name__ == "__main__":
    np.random.seed(123433789) # GIVING A SEED NUMBER FOR THE EXPERIENCE TO BE REPRODUCIBLE
    X = np.random.randint(0, 100, 10)
    Y = np.random.randint(0, 100, 10)  # CREATE POINT SET.
    V = np.random.randint(0, 10, 10)   # THIS IS MY VARIABLE
    heat_plot(X, Y, V)
