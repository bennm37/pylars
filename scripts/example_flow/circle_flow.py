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
    lambda t: 0.5 * np.exp(2j * np.pi * t),
    num_points=100,
    deg_laurent=20,
    centroid=0.0 + 0.0j,
)
prob.add_boundary_condition("0", "psi[0]", 1)
prob.add_boundary_condition("0", "u[0]", 0)
prob.add_boundary_condition("2", "psi[2]", 0)
prob.add_boundary_condition("2", "u[2]", 0)
prob.add_boundary_condition("1", "u[1]-u[3][::-1]", 0)
prob.add_boundary_condition("1", "v[1]-v[3][::-1]", 0)
prob.add_boundary_condition("4", "u[4]", 0)
prob.add_boundary_condition("4", "v[4]", 0)

solver = Solver(prob)
sol = solver.solve(check=False, normalize=False)
an = Analysis(prob, sol)
fig, ax = an.plot(resolution=100, interior_patch=True, buffer=1e-2)
plt.savefig("media/circle_flow.pdf")
plt.show()
