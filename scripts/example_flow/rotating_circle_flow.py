"""Flow a domain with a circular interior curve.""" ""
from pylars import Problem, Solver, Analysis
import matplotlib.pyplot as plt
import numpy as np

prob = Problem()
corners = [-2 - 1j, 2 - 1j, 2 + 1j, -2 + 1j]
prob.add_exterior_polygon(
    corners,
    num_edge_points=600,
    num_poles=0,
    deg_poly=50,
    spacing="linear",
)
centroid = 0.5 + 0.0j
circle = lambda t: centroid + 0.5 * np.exp(2j * np.pi * t)  # noqa: E731
circle_deriv = lambda t: 1j * np.pi * np.exp(2j * np.pi * t)  # noqa: E731
num_points = 100
prob.add_interior_curve(
    circle,
    num_points=num_points,
    deg_laurent=20,
    centroid=centroid,
)
# prob.domain.plot()
# plt.show()
prob.add_boundary_condition("0", "u[0]", 1)
prob.add_boundary_condition("0", "v[0]", 0)
prob.add_boundary_condition("2", "u[2]", 0)
prob.add_boundary_condition("2", "v[2]", 0)
prob.add_boundary_condition("1", "u[1]-u[3][::-1]", 0)
prob.add_boundary_condition("1", "v[1]-v[3][::-1]", 0)
prob.add_boundary_condition("3", "p[1]-p[3][::-1]", 1)
prob.add_boundary_condition("3", "e12[1]-e12[3][::-1]", 0)
# prob.add_boundary_condition("0", "v[0]", 0)
# prob.add_boundary_condition("0", "u[0]", 1)
# prob.add_boundary_condition("2", "v[2]", 0)
# prob.add_boundary_condition("2", "u[2]", 0)
# prob.add_boundary_condition("1", "p[1]", 1)
# prob.add_boundary_condition("1", "v[1]", 0)
# prob.add_boundary_condition("3", "p[3]", 0)
# prob.add_boundary_condition("3", "v[3]", 0)
rot_speed = -0.0 / np.pi
trans_speed = -0.1
prob.add_boundary_condition(
    "4",
    "u[4]",
    trans_speed + rot_speed * circle_deriv(np.linspace(0, 1, num_points)).real,
)
prob.add_boundary_condition(
    "4", "v[4]", rot_speed * circle_deriv(np.linspace(0, 1, num_points)).imag
)

solver = Solver(prob)
sol = solver.solve(check=False, normalize=False)
an = Analysis(sol)
print(f"residusal = {np.abs(solver.A @ solver.coefficients - solver.b).max()}")
fig, ax = an.plot(resolution=200, interior_patch=True, quiver=True)
# plt.savefig("media/rotating_flow.pdf")
plt.show()
