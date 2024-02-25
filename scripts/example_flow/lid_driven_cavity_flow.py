"""Solve the lid driven cavity problem from the paper."""
from pylars import Problem, Solver, Analysis
import numpy as np
import matplotlib.pyplot as plt
import time

# create a square domain
start = time.perf_counter()
corners = [1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j]
prob = Problem()
prob.add_exterior_polygon(
    corners,
    num_edge_points=300,
    num_poles=24,
    length_scale=1.5 * np.sqrt(2),
    sigma=4,
    deg_poly=24,
)
prob.add_boundary_condition("0", "psi[0]", 0)
prob.add_boundary_condition("0", "u[0]", 1)
prob.add_boundary_condition("2", "psi[2]", 0)
prob.add_boundary_condition("2", "u[2]", 0)
prob.add_boundary_condition("1", "psi[1]", 0)
prob.add_boundary_condition("1", "v[1]", 0)
prob.add_boundary_condition("3", "psi[3]", 0)
prob.add_boundary_condition("3", "v[3]", 0)
solver = Solver(prob)
sol = solver.solve()
end = time.perf_counter()
print("Time taken: ", end - start, "s")
print(f"Error: {solver.max_error}")

a = Analysis(sol)
fig, ax = a.plot(resolution=300)
max = a.psi_values[~np.isnan(a.psi_values)].max()
levels_moffat = max + np.linspace(-6e-6, 0, 10)
ax.contour(
    a.X, a.Y, a.psi_values, colors="y", levels=levels_moffat, linewidths=0.3
)
ax.axis("off")
# ax.scatter(
#     prob.domain.poles[:, -18:].real,
#     prob.domain.poles[:, -18:].imag,
#     s=1,
#     c="r",
# )
# plt.savefig("media/lid_driven_cavity.pdf")
plt.show()
