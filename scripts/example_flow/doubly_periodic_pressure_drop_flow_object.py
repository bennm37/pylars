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
    corners, num_edge_points=300, deg_poly=50, num_poles=0, spacing="linear"
)
centroid = shift + 0.5 - 0.5j
R = 0.2j
prob.add_interior_curve(
    lambda t: centroid + R * np.exp(2j * np.pi * t),
    num_points=300,
    deg_laurent=30,
    centroid=centroid,
)
centroid2 = shift + 0.5j
R = 0.2j
prob.add_interior_curve(
    lambda t: centroid2 + R * np.exp(2j * np.pi * t),
    num_points=300,
    deg_laurent=30,
    centroid=centroid2,
)

prob.add_point(shift + -1 - 1j)
# prob.domain.plot()
# plt.show()

# top and bottom periodicity
prob.add_boundary_condition("0", "u[0]-u[2][::1]", 0)
prob.add_boundary_condition("0", "v[0]-v[2][::-1]", 0)
prob.add_boundary_condition("2", "p[0]-p[2][::-1]", 0)
prob.add_boundary_condition("2", "e12[0]-e12[2]", 0)

prob.add_boundary_condition("1", "psi[1]-psi[3][::-1]", 0)
prob.add_boundary_condition("1", "v[1]-v[3][::-1]", 0)
prob.add_boundary_condition("3", "p[1]-p[3][::-1]", 2)
prob.add_boundary_condition("3", "e12[1]-e12[3][::-1]", 0)
prob.add_boundary_condition("4", "u[4]", 0)
prob.add_boundary_condition("4", "v[4]", 0)
prob.add_boundary_condition("5", "u[5]", 0)
prob.add_boundary_condition("5", "v[5]", 0)
prob.add_boundary_condition("6", "p[6]", 2)
prob.add_boundary_condition("6", "psi[6]", 0)

solver = Solver(prob)
sol = solver.solve(check=False, weight=False, normalize=False)
residual = np.max(np.abs(solver.A @ solver.coefficients - solver.b))
# residual = np.max(
#     np.abs(solver.A @ solver.coefficients - solver.b)
#     / (np.abs(solver.b) + 1e-8)
# )
print(f"Residual: {residual:.15e}")
# sol.problem.domain._update_polygon(buffer=1e-5)
sol.problem.domain.enlarge_holes(1.1)
an = Analysis(sol)
fig, ax = an.plot(interior_patch=True, resolution=200, epsilon=0.01)
plt.show()

# continuity checks
dom = sol.problem.domain
points_0 = dom.boundary_points[dom.indices["0"]]
points_1 = dom.boundary_points[dom.indices["1"]]
plt.plot(sol.p(points_0))
plt.show()
