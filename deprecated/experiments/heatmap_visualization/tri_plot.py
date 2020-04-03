"""
Interpolation from triangular grid to quad grid.
"""
from matplotlib import pyplot
from matplotlib import tri
import numpy as np

# Create triangulation.
x = np.asarray([0, 1, 2, 3, 0.5, 1.5, 2.5, 1, 2, 1.5])
y = np.asarray([0, 0, 0, 0, 1.0, 1.0, 1.0, 2, 2, 3.0])
triangles = [[0, 1, 4], [1, 2, 5], [2, 3, 6], [1, 5, 4], [2, 6, 5], [4, 5, 7],
             [5, 6, 8], [5, 8, 7], [7, 8, 9]]
triang = tri.Triangulation(x, y, triangles)

# Interpolate to regularly-spaced quad grid.
z = np.cos(1.5*x)*np.cos(1.5*y)
xi, yi = np.meshgrid(np.linspace(0, 3, 20), np.linspace(0, 3, 20))

interp_lin = tri.LinearTriInterpolator(triang, z)
zi_lin = interp_lin(xi, yi)

interp_cubic_geom = tri.CubicTriInterpolator(triang, z, kind='geom')
zi_cubic_geom = interp_cubic_geom(xi, yi)

interp_cubic_min_E = tri.CubicTriInterpolator(triang, z, kind='min_E')
zi_cubic_min_E = interp_cubic_min_E(xi, yi)


# Plot the triangulation.
pyplot.subplot(221)
pyplot.tricontourf(triang, z)
pyplot.triplot(triang, 'ko-')
pyplot.title('Triangular grid')

# Plot linear interpolation to quad grid.
pyplot.subplot(222)
pyplot.contourf(xi, yi, zi_lin)
pyplot.plot(xi, yi, 'k-', lw=0.5, alpha=0.5)
pyplot.plot(xi.T, yi.T, 'k-', lw=0.5, alpha=0.5)
pyplot.title("Linear interpolation")

# Plot cubic interpolation to quad grid, kind=geom
pyplot.subplot(223)
pyplot.contourf(xi, yi, zi_cubic_geom)
pyplot.plot(xi, yi, 'k-', lw=0.5, alpha=0.5)
pyplot.plot(xi.T, yi.T, 'k-', lw=0.5, alpha=0.5)
pyplot.title("Cubic interpolation,\nkind='geom'")

# Plot cubic interpolation to quad grid, kind=min_E
pyplot.subplot(224)
pyplot.contourf(xi, yi, zi_cubic_min_E)
pyplot.plot(xi, yi, 'k-', lw=0.5, alpha=0.5)
pyplot.plot(xi.T, yi.T, 'k-', lw=0.5, alpha=0.5)
pyplot.title("Cubic interpolation,\nkind='min_E'")

pyplot.tight_layout()
pyplot.show()
