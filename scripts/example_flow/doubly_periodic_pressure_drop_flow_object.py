"""Solve Poiseiulle flow with stream function boundary conditions."""
from pylars import Problem, Solver, Analysis
import numpy as np
import matplotlib.pyplot as plt

# create a square domain
shift = 0.0 + 0.0j
corners = np.array([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j])
corners += shift
prob = Problem()
prob.add_exterior_polygon(
    corners, num_edge_points=50, deg_poly=50, num_poles=0, spacing="linear"
)
centroid = shift + 0.5 - 0.5j
R = 0.2j
prob.add_interior_curve(
    lambda t: centroid + R * np.exp(2j * np.pi * t),
    num_points=50,
    deg_laurent=10,
    centroid=centroid,
)
centroid2 = shift + 0.5j
R = 0.2j
prob.add_interior_curve(
    lambda t: centroid2 + R * np.exp(2j * np.pi * t),
    num_points=50,
    deg_laurent=10,
    centroid=centroid2,
)
prob.add_point(shift + -1 - 1j)
# prob.domain.plot()
# plt.show()

# top and bottom periodicity
p_drop = 2.0
prob.add_boundary_condition("0", "u[0]-u[2][::-1]", 0)
prob.add_boundary_condition("0", "v[0]-v[2][::-1]", 0)
prob.add_boundary_condition("2", "p[0]-p[2][::-1]", 0)
prob.add_boundary_condition("2", "e12[0]-e12[2][::-1]", 0)

prob.add_boundary_condition("1", "psi[1]-psi[3][::-1]", 0)
prob.add_boundary_condition("1", "v[1]-v[3][::-1]", 0)
prob.add_boundary_condition("3", "p[1]-p[3][::-1]", p_drop)
prob.add_boundary_condition("3", "e12[1]-e12[3][::-1]", 0)
prob.add_boundary_condition("4", "u[4]", 0)
prob.add_boundary_condition("4", "v[4]", 0)
prob.add_boundary_condition("5", "u[5]", 0)
prob.add_boundary_condition("5", "v[5]", 0)
prob.add_boundary_condition("6", "p[6]", 0)
prob.add_boundary_condition("6", "psi[6]", 0)

solver = Solver(prob)
sol = solver.solve(check=False, weight=False, normalize=False)
residual = np.max(np.abs(solver.A @ solver.coefficients - solver.b))
# relatieve_
# residual = np.max(
#     np.abs(solver.A @ solver.coefficients - solver.b)
#     / (np.abs(solver.b) + 1e-8)
# )
print(f"Residual: {residual:.15e}")
# sol.problem.domain._update_polygon(buffer=1e-5)
sol.problem.domain.enlarge_holes(1.1)
an = Analysis(sol)
fig, ax = an.plot(interior_patch=True, resolution=200, epsilon=0.01)
plt.savefig("media/doubly_periodic_pressure_drop_flow_object.pdf")

# continuity checks
dom = sol.problem.domain
points_0 = dom.boundary_points[dom.indices["0"]]
points_1 = dom.boundary_points[dom.indices["1"]]
fig, ax = plt.subplots()
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
