from pyls import Domain, Solver
import numpy as np
import matplotlib.pyplot as plt

# create a square domain
corners = [1 + 1j, -1 + 1j, -1, 0, -1j, 1 - 1j]
dom = Domain(corners, num_boundary_points=300, num_poles=24)
dom.show()
sol = Solver(dom, 24)
# 1 is the inlet, 4 is the outlet, 2,3,5 are the walls
# inlet
sol.add_boundary_condition("1", "u(1)-1", 0)
sol.add_boundary_condition("1", "v(1)", 0)
# outlet
sol.add_boundary_condition("4", "p(4)", 0)
sol.add_boundary_condition("4", "v(4)", 0)
# walls no slip no penetration
sol.add_boundary_condition("0", "u(0)", 0)
sol.add_boundary_condition("0", "v(0)", 0)
sol.add_boundary_condition("2", "u(2)", 0)
sol.add_boundary_condition("2", "v(2)", 0)
sol.add_boundary_condition("3", "u(3)", 0)
sol.add_boundary_condition("3", "v(3)", 0)
psi, uv, p, omega = sol.solve(check=False)

residual = np.max(np.abs(sol.A @ sol.coefficients - sol.b))
print(f"Residual: {residual:.15e}")
"flip(u(0))"

x = np.linspace(-1, 1, 100)
X, Y = np.meshgrid(x, x)
Z = X + 1j * Y
Z[np.logical_not(dom.mask_contains(Z))] = np.nan
psi_100_100 = psi(Z.flatten()).reshape(100, 100)
uv_100_100 = uv(Z.flatten()).reshape(100, 100)
# plot the velocity magnitude
fig, ax = plt.subplots()
# interpolate using bilinear interpolation
speed = np.abs(uv_100_100)
pc = ax.pcolormesh(X, Y, np.abs(uv_100_100), cmap="jet")
plt.colorbar(pc)
ax.contour(X, Y, psi_100_100, colors="k", levels=20)
ax.set_aspect("equal")
plt.show()
