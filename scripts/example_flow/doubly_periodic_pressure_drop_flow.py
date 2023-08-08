"""Solve poiseuille flow with stream function boundary conditions."""
from pylars import Problem, Solver, Analysis
import numpy as np
import matplotlib.pyplot as plt

# create a square domain
corners = [1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j]
prob = Problem()
prob.add_exterior_polygon(
    corners, num_edge_points=300, deg_poly=50, num_poles=0, spacing="linear"
)
prob.add_boundary_condition("0", "u[0]-u[2][::1]", 0)
prob.add_boundary_condition("0", "psi[0]-psi[2][::-1]", 1)
prob.add_boundary_condition("2", "p[0]-p[2][::-1]", 0)
prob.add_boundary_condition("2", "e12[0]-e12[2][::-1]", 0)
p_drop = 0
prob.add_boundary_condition("1", "u[1]-u[3][::-1]", 0)
prob.add_boundary_condition("1", "v[1]-v[3][::-1]", 0)
prob.add_boundary_condition("3", "p[1]-p[3][::-1]", p_drop)
prob.add_boundary_condition("3", "e12[1]-e12[3][::-1]", 0)
solver = Solver(prob)
sol = solver.solve(check=False, weight=False, normalize=True)
residual = np.max(np.abs(solver.A @ solver.coefficients - solver.b))
print(f"Residual: {residual:.15e}")
sol.problem.domain._update_polygon(buffer=1e-5)
an = Analysis(sol)
fig, ax = an.plot(interior_patch=True, resolution=200)
plt.show()

# continuity checks
dom = sol.problem.domain
points_0 = dom.boundary_points[dom.indices["0"]]
points_1 = dom.boundary_points[dom.indices["1"]]
plt.plot(
    points_0.real,
    np.abs(sol.eij(points_0)[:, 0, 1] - sol.eij(points_0 - 2j)[:, 0, 1]),
    label="e12 tb",
)
plt.plot(
    points_1.imag,
    np.abs(sol.eij(points_1)[:, 0, 1] - sol.eij(points_1 + 2)[:, 0, 1]),
    label="e12 lr",
)
plt.plot(
    points_0.real,
    np.abs(sol.p(points_0) - sol.p(points_0 - 2j)),
    label="p tb",
)
plt.plot(
    points_1.imag,
    np.abs(sol.p(points_1) - sol.p(points_1 + 2) - p_drop),
    label="p lr",
)

plt.legend()
# plt.plot(points.imag, sol.p(points+2), label="p right")
plt.show()
