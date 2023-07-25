"""Flow a domain with a circular interior curve.""" ""
from pylars import Problem, Solver, Analysis
import matplotlib.pyplot as plt
import numpy as np

prob = Problem()
corners = [-1 - 1j, 1 - 1j, 1 + 1j, -1 + 1j]
prob.add_exterior_polygon(
    corners,
    num_edge_points=600,
    num_poles=0,
    deg_poly=20,
    spacing="linear",
)
prob.add_interior_curve(
    lambda t: 0.1 * np.exp(2j * np.pi * t),
    num_points=100,
    deg_laurent=20,
    centroid=0.0 + 0.0j,
)
# prob.domain.plot()
# plt.show()
prob.add_point(-1.0 - 1.0j)
prob.add_boundary_condition("0", "u[0]", 0)
prob.add_boundary_condition("0", "psi[0]", 0)
prob.add_boundary_condition("2", "u[2]", 0)
prob.add_boundary_condition("2", "v[2]", 0)
prob.add_boundary_condition("1", "u[3]-u[1][::-1]", 0)
prob.add_boundary_condition("1", "v[3]-v[1][::-1]", 0)
prob.add_boundary_condition("3", "e12[3]-e12[1][::-1]", 0)
prob.add_boundary_condition("3", "p[3]-p[1][::-1]", 10)
prob.add_boundary_condition("4", "u[4]", 0.2)
prob.add_boundary_condition("4", "v[4]", 0)
prob.add_boundary_condition("5", "p[5]", 0)
# prob.add_boundary_condition("5", "psi[5]", 0)

solver = Solver(prob)
sol = solver.solve(check=False, normalize=False)
an = Analysis(sol)
fig, ax = an.plot(resolution=200, interior_patch=True, levels=100)
plt.show()
