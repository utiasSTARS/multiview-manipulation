import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import chi2


def plot_confidence_ellipsoid_from_points(ax, points, percentile, color, alpha=0.2):
    """ Plot a confidence ellipsoid from a set of 3d points.

    points should be (3, num_pts) array, percentile is [0, 1], indicating proportion of points distribution
    that should be within bounds of ellipse.


    Uses code/ideas from several places, including:

    https://stackoverflow.com/questions/7819498/plotting-ellipsoid-with-matplotlib (mostly from 2nd answer here, with edit in comment)
    https://www.mathworks.com/matlabcentral/fileexchange/4705-error_ellipse (chi square percentile magnitude)
    http://www.cs.utah.edu/~tch/CS6640F2020/resources/How%20to%20draw%20a%20covariance%20error%20ellipse.pdf
    https://people.richland.edu/james/lecture/m170/tbl-chi.html
    """

    cov_mat = np.cov(points)
    center = points.mean(axis=1)

    # find the rotation matrix and radii of the axes
    U, s, rotation = linalg.svd(cov_mat)
    radii = np.sqrt(s)

    # multiply sizes by desired percentile from chi square dist
    chi_dist = chi2(3)
    mag = np.sqrt(chi_dist.ppf(percentile))
    radii = radii * mag

    u = np.linspace(0.0, 2.0 * np.pi, 100)
    v = np.linspace(0.0, np.pi, 100)
    x = radii[0] * np.outer(np.cos(u), np.sin(v))
    y = radii[1] * np.outer(np.sin(u), np.sin(v))
    z = radii[2] * np.outer(np.ones_like(u), np.cos(v))
    for i in range(len(x)):
        for j in range(len(x)):
            [x[i, j], y[i, j], z[i, j]] = np.dot([x[i, j], y[i, j], z[i, j]], rotation) + center

    # plot
    # ax.scatter(points[0], points[1], points[2])
    ax.plot_wireframe(x, y, z, rstride=4, cstride=4, color=color, alpha=alpha)