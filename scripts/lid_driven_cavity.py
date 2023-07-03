"""Solve the lid driven cavity problem from the paper."""
from pyls import Domain, Solver, Analysis
import numpy as np
import matplotlib.pyplot as plt
import time

# create a square domain
start = time.perf_counter()
corners = [1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j]
dom = Domain(
    corners, num_boundary_points=300, num_poles=24, L=1.5 * np.sqrt(2)
)
sol = Solver(dom, 24)
sol.add_boundary_condition("0", "psi(0)", 0)
sol.add_boundary_condition("0", "u(0)", 1)
sol.add_boundary_condition("2", "psi(2)", 0)
sol.add_boundary_condition("2", "u(2)", 0)
sol.add_boundary_condition("1", "psi(1)", 0)
sol.add_boundary_condition("1", "v(1)", 0)
sol.add_boundary_condition("3", "psi(3)", 0)
sol.add_boundary_condition("3", "v(3)", 0)
psi, uv, p, omega = sol.solve()
end = time.perf_counter()
print("Time taken: ", end - start, "s")
residual = np.max(np.abs(sol.A @ sol.coefficients - sol.b))
print(f"Residual: {residual:.15e}")

a = Analysis(dom, sol)
fig, ax = a.plot(resolution=300)
max = a.psi_values[~np.isnan(a.psi_values)].max()
levels_moffat = max + np.linspace(-6e-6, 0, 10)
ax.contour(
    a.X, a.Y, a.psi_values, colors="y", levels=levels_moffat, linewidths=0.3
)
ax.axis("off")
ax.scatter(dom.poles[:, -18:].real, dom.poles[:, -18:].imag, s=1, c="r")
plt.savefig("media/lid_driven_cavity.pdf")
plt.show()
