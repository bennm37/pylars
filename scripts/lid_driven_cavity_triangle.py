from pyls import Domain, Solver
import numpy as np
import matplotlib.pyplot as plt

# create a square domain
corners = [1 + 1j, -1 + 1j, -1j]
dom = Domain(corners, num_boundary_points=300, num_poles=24)
sol = Solver(dom, 24)
sol.add_boundary_condition("0", "psi(0)", 0)
sol.add_boundary_condition("0", "u(0)", 1)
sol.add_boundary_condition("2", "psi(2)", 0)
sol.add_boundary_condition("2", "u(2)", 0)
sol.add_boundary_condition("1", "u(1)", 0)
sol.add_boundary_condition("1", "v(1)", 0)
psi, uv, p, omega = sol.solve()

residual = np.max(np.abs(sol.A @ sol.coefficients - sol.b))
print(f"Residual: {residual:.15e}")

a = a
# x = np.linspace(-1, 1, 100)
# X, Y = np.meshgrid(x, x)
# Z = X + 1j * Y
# psi_100_100 = psi(Z.flatten()).reshape(100, 100)
# uv_100_100 = uv(Z.flatten()).reshape(100, 100)
# # plot the velocity magnitude
# fig, ax = plt.subplots()
# # interpolate using bilinear interpolation
# speed = np.abs(uv_100_100)
# ax.pcolormesh(X, Y, np.abs(uv_100_100), cmap="jet")
# ax.contour(X, Y, psi_100_100, colors="k", levels=20)
# ax.set_aspect("equal")
# plt.show()
