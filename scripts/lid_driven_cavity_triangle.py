from pyls import Domain, Solver, Analysis
import numpy as np
import matplotlib.pyplot as plt

# create a square domain
corners = [1 + 1j, -1 + 1j, -5j]
dom = Domain(corners, num_boundary_points=300, num_poles=49)
sol = Solver(dom, 24)
sol.add_boundary_condition("0", "psi(0)", 0)
sol.add_boundary_condition("0", "u(0)", 1)
sol.add_boundary_condition("2", "psi(2)", 0)
sol.add_boundary_condition("2", "u(2)", 0)
sol.add_boundary_condition("1", "psi(1)", 0)
sol.add_boundary_condition("1", "u(1)", 0)
psi, uv, p, omega = sol.solve()

residual = np.max(np.abs(sol.A @ sol.coefficients - sol.b))
print(f"Residual: {residual:.15e}")

a = Analysis(dom, sol)
fig, ax = a.plot(resolution=100)
ax.axis("off")
values = a.psi_values[~np.isnan(a.psi_values)].flatten()
max = values.max()
min = values.min()
moffat_levels = max + np.linspace(-3e-4, 0, 10)
ax.contour(
    a.X, a.Y, a.psi_values, levels=moffat_levels, colors="y", linewidths=0.5
)
moffat_levels_2 = max + np.linspace(-3e-3, -3e-4, 10)
# moffat_levels_2 = min + np.linspace(0, 1e-4, 10)
ax.contour(
    a.X, a.Y, a.psi_values, levels=moffat_levels_2, colors="w", linewidths=0.5
)
plt.show()
