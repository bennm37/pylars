from pyls import Domain, Solver
import numpy as np
import matplotlib.pyplot as plt

# create a square domain
corners = [1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j]
dom = Domain(corners, num_boundary_points=300, num_poles=0)
sol = Solver(dom, 24)
sol.add_boundary_condition("0", "u(0)", 0)
sol.add_boundary_condition("0", "psi(0)", 1 / 3)
sol.add_boundary_condition("2", "u(2)", 0)
sol.add_boundary_condition("2", "psi(2)", "0*y-1/3")
# inlet
y = np.linspace(-1, 1, 300)
sol.add_boundary_condition("1", "u(1)-u(3)[::-1]", 0)
sol.add_boundary_condition("1", "v(1)-v(3)[::-1]", 0)
# sol.add_boundary_condition("3", "p(3)", 1)
# sol.add_boundary_condition("3", "v(3)", 0)
psi, uv, p, omega = sol.solve(check=False)

residual = np.max(np.abs(sol.A @ sol.coefficients - sol.b))
print(f"Residual: {residual:.15e}")
"flip(u(0))"

x = np.linspace(-1, 1, 100)
X, Y = np.meshgrid(x, x)
Z = X + 1j * Y
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
