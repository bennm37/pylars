"""Solve periodic Couette flow using the lighting stokes method."""
from pyls import Domain, Solver, Analysis
import numpy as np
import matplotlib.pyplot as plt

# create a square domain
corners = [1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j]
dom = Domain(corners, num_boundary_points=300, num_poles=0, spacing="linear")
sol = Solver(dom, 24)
# periodic Couette flow with no-slip boundary conditions
sol.add_boundary_condition("0", "u(0)", 1)
sol.add_boundary_condition("0", "v(0)", 0)
sol.add_boundary_condition("2", "u(2)", 0)
sol.add_boundary_condition("2", "v(2)", 0)
sol.add_boundary_condition("1", "p(1)", 0)
sol.add_boundary_condition("1", "v(1)", 0)
sol.add_boundary_condition("3", "p(3)", 0)
sol.add_boundary_condition("3", "v(3)", 0)
psi, uv, p, omega = sol.solve(check=False, weight=False)
residual = np.max(np.abs(sol.A @ sol.coefficients - sol.b))
print(f"Residual: {residual:.15e}")
analyse = Analysis(dom, sol)
fig, ax = analyse.plot()
y = np.linspace(-1, 1, 100)
ax.plot(uv(1j * y - 1).real - 1, y, color="black")
y = np.linspace(-1, 0.9, 10)
x = -np.ones_like(y)
u = uv(x + 1j * y).real
v = uv(x + 1j * y).imag
ax.quiver(x, y, u, v, color="black", scale=2)
ax.axis("off")
plt.savefig("media/couette_flow.pdf", bbox_inches="tight")
plt.show()
