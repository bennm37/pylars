from pylars import Domain, Solver
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.io import loadmat, savemat
from pylars.numerics import va_orthogonalise


# create a square domain
start = time.perf_counter()
corners = [1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j]
dom = Domain(
    corners,
    num_edge_points=300,
    num_poles=24,
    length_scale=1.5 * np.sqrt(2),
)
sol = Solver(dom, 24, least_squares="pinv")
# sol.add_boundary_condition("0", "psi[0]", 0)
# sol.add_boundary_condition("0", "u[0]", 1)
# sol.add_boundary_condition("2", "psi[2]", 0)
# sol.add_boundary_condition("2", "u[2]", 0)
# sol.add_boundary_condition("1", "u[1]", 0)
# sol.add_boundary_condition("1", "v[1]", 0)
# sol.add_boundary_condition("3", "u[3]", 0)
# sol.add_boundary_condition("3", "v[3]", 0)
sol.add_boundary_condition("0", "psi[0]", 0)
sol.add_boundary_condition("0", "u[0]", 1)
sol.add_boundary_condition("2", "psi[2]", 0)
sol.add_boundary_condition("2", "u[2]", 0)
sol.add_boundary_condition("1", "psi[1]", 0)
sol.add_boundary_condition("1", "v[1]", 0)
sol.add_boundary_condition("3", "psi[3]", 0)
sol.add_boundary_condition("3", "v[3]", 0)

psi, uv, p, omega = sol.solve()
end = time.perf_counter()
print("Time taken: ", end - start, "s")
residual = np.max(np.abs(sol.A @ sol.coefficients - sol.b))
print(f"Residual: {residual:.15e}")


x = np.linspace(-1, 1, 100)
X, Y = np.meshgrid(x, x)
Z = X + 1j * Y
psi_100_100 = psi(Z.flatten()).reshape(100, 100)
uv_100_100 = uv(Z.flatten()).reshape(100, 100)
# plot the velocity magnitude
fig, ax = plt.subplots()
# interpolate using bilinear interpolation
speed = np.abs(uv_100_100)
ax.pcolormesh(X, Y, np.abs(uv_100_100), cmap="jet")
psi_min, psi_max = psi_100_100.min(), psi_100_100.max()
levels = np.linspace(psi_min, psi_max, 20)
ax.contour(X, Y, psi_100_100, colors="k", levels=20)
ax.set_aspect("equal")
psi_ratio = psi_min / psi_max
fac = np.max([psi_ratio, 1 / psi_ratio])

# if np.sign(fac) == -1:
#     levels_1 = levels[::2] * fac * -1
#     ax.contour(X, Y, psi_100_100, levels=levels_1, colors="y", linewidths=1)
# if np.abs(fac) > 1e4:
#     levels_2 = levels_1 * fac
#     ax.contour(X, Y, psi_100_100, levels=levels_2, colors="w", linewidths=1)
# if np.abs(fac) > 1e6:
#     levels_3 = levels_2 * fac
#     ax.contour(X, Y, psi_100_100, levels=levels_3, colors="y", linewidths=1)
levels_moffat = np.linspace(1e-7, 5e-6, 10)
ax.contour(X, Y, psi_100_100, colors="y", levels=levels_moffat, linewidths=0.5)

plt.show()
