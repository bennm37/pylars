import numpy as np
import matplotlib.pyplot as plt

# get trapezium rule
from scipy.integrate import trapz

type = "couette"
n = 1000
stride = 10
fs = 50
centroid = np.array([0.3, 0.5])
t = np.linspace(0, 1, n)
R = 0.5
x = centroid[0] + R * np.cos(2 * np.pi * t)
y = centroid[1] + R * np.sin(2 * np.pi * t)
X, Y = np.meshgrid(np.linspace(-1, 1, 20), np.linspace(-1, 1, 20), indexing="ij")
if type == "poiseuille":
    U = 1 - Y**2
    V = np.zeros_like(Y)
    stress = np.array([[2 * x, -2 * y], [-2 * y, 2 * x]])
if type == "couette":
    U = 1 + Y
    V = np.zeros_like(Y)
    z = np.zeros_like(x)
    o = np.ones_like(x)
    stress = np.array([[z, o], [o, z]])
norm = np.array([np.cos(2 * np.pi * t), np.sin(2 * np.pi * t)]).T
stress = np.moveaxis(stress, 2, 0)
norm_stress = np.array([stress[i] @ norm[i] for i in range(n)])
dt = 2 * np.pi * R / n
norm_torque = np.array([np.cross(norm[i], norm_stress[i]) for i in range(n)])
force = trapz(norm_stress, axis=0, dx=dt)
torque = trapz(norm_torque, axis=0, dx=dt)
print("force = ", force)
print("torque = ", torque)
fig, ax = plt.subplots(1, 2)
ax[0].plot(x, y)
ax[0].set_aspect("equal")
ax[0].quiver(
    x[::stride],
    y[::stride],
    norm_stress[::stride, 0],
    norm_stress[::stride, 1],
    color="b",
    label="normal stress",
)
ax[0].quiver(X, Y, U, V, color="k", label="flow", scale=30)
comp_force = force[0] + 1j * force[1]
ax[0].quiver(
    centroid[0],
    centroid[1],
    force[0],
    force[1],
    color="g",
    label=f"force = {comp_force:.1}",
)
ax[0].legend(loc="center", bbox_to_anchor=(0.5, 1.2, 0.0, 0.0))
ax[1].quiver(X, Y, U, V, color="k", label="flow", scale=30)
ax[1].set_aspect("equal")
scat = ax[1].scatter(x, y, c=norm_torque, s=1, label="point torque")
plt.colorbar(scat)
ax[1].quiver(
    centroid[0] + R,
    centroid[1],
    0,
    torque,
    color="r",
    scale=5,
    label=f"torque = {torque:.1}",
)
ax[1].legend(loc="center", bbox_to_anchor=(0.5, 1.2, 0.0, 0.0))
plt.show()
