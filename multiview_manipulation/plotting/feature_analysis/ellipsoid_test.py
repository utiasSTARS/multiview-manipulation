import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import chi2


x, y = np.random.normal(size=(2, 100))
z = x + y + np.random.normal(size=(1, 100))
points = np.array([x, y, z.squeeze()])

A = np.cov(points)

center = points.mean(axis=1)

# your ellispsoid and center in matrix form
# A = np.array([[10,0,0],[0,2,0],[0,0,2]])

# A = np.array([[ 0.00009, -0.00024, -0.00014],
#        [-0.00024,  0.00095,  0.00068],
#        [-0.00014,  0.00068,  0.00063]])

# center = [0,0,0]

# find the rotation matrix and radii of the axes
U, s, rotation = linalg.svd(A)
radii = np.sqrt(s)

# multiply sizes by desired percentile from chi square dist
chi_dist = chi2(3)
mag = np.sqrt(chi_dist.ppf(.95))
radii = radii * mag

# todo no scaling based on confidence interval yet -- test this but with points and confidence intervals
# see https://www.mathworks.com/matlabcentral/fileexchange/4705-error_ellipse

# now carry on with EOL's answer
u = np.linspace(0.0, 2.0 * np.pi, 100)
v = np.linspace(0.0, np.pi, 100)
x = radii[0] * np.outer(np.cos(u), np.sin(v))
y = radii[1] * np.outer(np.sin(u), np.sin(v))
z = radii[2] * np.outer(np.ones_like(u), np.cos(v))
for i in range(len(x)):
    for j in range(len(x)):
        [x[i,j],y[i,j],z[i,j]] = np.dot([x[i,j],y[i,j],z[i,j]], rotation) + center

# plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(points[0], points[1], points[2])

ax.plot_wireframe(x, y, z,  rstride=1, cstride=1, color='b', alpha=0.2)

# lim = 2
# ax.set_xlim([-lim, lim])
# ax.set_ylim([-lim, lim])
# ax.set_zlim([-lim, lim])

# with points
p = points
ax.set_xlim([p[0].min() - .1, p[0].max() + .1])
ax.set_ylim([p[1].min() - .1, p[1].max() + .1])
ax.set_zlim([p[2].min() - .1, p[2].max() + .1])

plt.show()
plt.close(fig)
del fig