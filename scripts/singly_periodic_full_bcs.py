"""Solve Poiseiulle flow with stream function boundary conditions."""
from pyls import Domain, Solver, Analysis
import numpy as np
import matplotlib.pyplot as plt

# create a square domain
corners = [1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j]
dom = Domain(corners, num_boundary_points=100, num_poles=0, spacing="linear")
sol = Solver(dom, 24)
sol.add_boundary_condition("0", "u(0)", 0)
sol.add_boundary_condition("0", "psi(0)", 1)
sol.add_boundary_condition("2", "u(2)", 0)
sol.add_boundary_condition("2", "psi(2)", -1)
# sol.add_boundary_condition("1", "u(1)-u(3)[::-1]", 0)
sol.add_boundary_condition("1", "psi(1)", "-np.sin(3*np.pi*y/2)")
# sol.add_boundary_condition("1", "psi(1)", "-y**3/2+3*y/2")
sol.add_boundary_condition("3", "v(1)-v(3)[::-1]", 0)
sol.add_boundary_condition("3", "psi(3)", "-np.sin(3*np.pi*y/2)")
# sol.add_boundary_condition("3", "psi(3)", "-y**3/2+3*y/2")
psi, uv, p, omega = sol.solve(check=False, weight=False)
residual = np.max(np.abs(sol.A @ sol.coefficients - sol.b))
print(f"Residual: {residual:.15e}")
a = Analysis(dom, sol)
fig, ax = a.plot()
plt.show()
# plot errors along each edge
fig, ax = a.plot_stream_boundary()
plt.show()
