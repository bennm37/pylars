import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def data_for_cylinder_along_z(center_x, center_y, radius, height_z):
    z = np.linspace(0, height_z, 100)
    theta = np.linspace(0, 2 * np.pi, 100)
    theta_grid, z_grid = np.meshgrid(theta, z)
    x_grid = radius * np.cos(theta_grid) + center_x
    y_grid = radius * np.sin(theta_grid) + center_y
    return x_grid, y_grid, z_grid


fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
airforceblue = [0.36, 0.54, 0.66, 0.5]
np.random.seed(2)
centers = np.random.random((10, 2)) * 0.9 + 0.05
radii = np.random.random(10) * 0.1
for center, radius in zip(centers, radii):
    Xc, Yc, Zc = data_for_cylinder_along_z(center[0], center[1], radius, 0.5)
    ax.plot_surface(
        Xc,
        Yc,
        Zc,
        alpha=1,
        color=airforceblue,
        rstride=1,
        cstride=1,
    )

X, Y = np.meshgrid(np.linspace(0, 1, 300), np.linspace(0, 1, 300))
Z = np.ones_like(X) * 0.5
colors = np.zeros((300, 300, 4))
for center, radius in zip(centers, radii):
    mask = (X - center[0]) ** 2 + (Y - center[1]) ** 2 < radius**2
    colors[mask, :] = airforceblue
ax.plot_surface(X, Y, Z, alpha=0.5, facecolors=colors, cstride=1, rstride=1)
ax.axis("equal")
ax.axis("off")
plt.show()
