"""Flow round a corner"""
from pyls import Domain, Solver, Analysis
import numpy as np
import matplotlib.pyplot as plt

# create a square domain
corners = [1 + 1j, -1 + 1j, -1, 0, -1j, 1 - 1j]
dom = Domain(corners, num_boundary_points=500, num_poles=30)
# dom.show()
sol = Solver(dom, 24)
# 1 is the inlet, 4 is the outlet, 0,2,3,5 are the walls
# inlet
sol.add_boundary_condition("1", "u(1)", "4*y*(1-y)")
sol.add_boundary_condition("1", "v(1)", 0)
# outlet
sol.add_boundary_condition("4", "p(4)", 0)
sol.add_boundary_condition("4", "u(4)", 0)
# walls no slip no penetration
sol.add_boundary_condition("0", "u(0)", 0)
sol.add_boundary_condition("0", "v(0)", 0)
sol.add_boundary_condition("2", "u(2)", 0)
sol.add_boundary_condition("2", "v(2)", 0)
sol.add_boundary_condition("3", "u(3)", 0)
sol.add_boundary_condition("3", "v(3)", 0)
sol.add_boundary_condition("5", "u(5)", 0)
sol.add_boundary_condition("5", "v(5)", 0)
psi, uv, p, omega = sol.solve(check=False)
residual = np.max(np.abs(sol.A @ sol.coefficients - sol.b))
print(f"Residual: {residual:.15e}")
a = Analysis(dom, sol)
fig, ax = a.plot(resolution=300)
max = a.psi_values[~np.isnan(a.psi_values)].max()
moffat_levels = max + np.linspace(-1.6e-5, 0, 10)
ax.contour(
    a.X, a.Y, a.psi_values, levels=moffat_levels, colors="y", linewidths=0.5
)
plt.show()
