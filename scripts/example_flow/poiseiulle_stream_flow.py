"""Solve Poiseiulle flow with stream function boundary conditions."""
from pylars import Problem, Solver, Analysis
import numpy as np
import matplotlib.pyplot as plt

# create a square domain
corners = [1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j]
prob.add_exterior_polygon(
    corners, num_edge_points=300, num_poles=0, spacing="linear"
)
sol = Solver(dom, 24)
sol.add_boundary_condition("0", "u(0)", 0)
sol.add_boundary_condition("0", "psi(0)", 2 / 3)
sol.add_boundary_condition("2", "u(2)", 0)
sol.add_boundary_condition("2", "psi(2)", -2 / 3)
sol.add_boundary_condition("1", "u(1)-u(3)[::-1]", 0)
sol.add_boundary_condition("1", "v(1)-v(3)[::-1]", 0)
psi, uv, p, omega = sol.solve(check=False, weight=False)
residual = np.max(np.abs(sol.A @ sol.coefficients - sol.b))
print(f"Residual: {residual:.15e}")
a = Analysis(dom, sol)
fig, ax = a.plot()
plt.show()
