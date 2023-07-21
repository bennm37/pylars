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
    deg_poly=20,
    spacing="linear",
)
circle = lambda t: 0.5 * np.exp(2j * np.pi * t)  # noqa: E731
circle_deriv = lambda t: 1j * np.pi * np.exp(2j * np.pi * t)  # noqa: E731
num_points = 100
prob.add_interior_curve(
    circle,
    num_points=num_points,
    deg_laurent=20,
    centroid=0.0 + 0.0j,
)
prob.domain.plot()
plt.show()
prob.add_boundary_condition("0", "psi[0]", 0)
prob.add_boundary_condition("0", "u[0]", 0)
prob.add_boundary_condition("2", "psi[2]", 1)
prob.add_boundary_condition("2", "u[2]", 0)
prob.add_boundary_condition("1", "u[1]-u[3][::-1]", 0)
prob.add_boundary_condition("1", "v[1]-v[3][::-1]", 0)
rot_speed = 1.0 / np.pi
trans_speed = 0.0
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
fig, ax = an.plot(resolution=100, interior_patch=True, buffer=1e-2)
# plt.savefig("media/rotating_flow.pdf")
plt.show()
