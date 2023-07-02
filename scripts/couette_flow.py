"""Solve periodic Couette flow using the lighting stokes method."""
from pyls import Domain, Solver, Analysis
import numpy as np
import matplotlib.pyplot as plt

# create a square domain
corners = [1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j]
dom = Domain(corners, num_boundary_points=300, num_poles=0)
sol = Solver(dom, 24)
# periodic Couette flow with no-slip boundary conditions
sol.add_boundary_condition("0", "u(0)", 1)
sol.add_boundary_condition("0", "v(0)", 0)
sol.add_boundary_condition("2", "u(2)", 0)
sol.add_boundary_condition("2", "v(2)", 0)
sol.add_boundary_condition("1", "u(1)-u(3)[::-1]", 0)
sol.add_boundary_condition("1", "v(1)-v(3)[::-1]", 0)
psi, uv, p, omega = sol.solve(check=False)
residual = np.max(np.abs(sol.A @ sol.coefficients - sol.b))
print(f"Residual: {residual:.15e}")
analyse = Analysis(dom, sol)
analyse.plot()
plt.show()
