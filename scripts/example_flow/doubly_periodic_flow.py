"""Using new doubly periodic flow BCs."""
from pylars import Problem, Solver, Analysis
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad


def check_flow(problem, solution):
    """Check if the flow is doubly periodic."""
    dom = problem.domain
    psi, uv, p, omega, eij = solution.functions
    boundary_top = dom.boundary_points[dom.indices["0"]]
    boundary_bottom = dom.boundary_points[dom.indices["2"]]
    boundary_left = dom.boundary_points[dom.indices["1"]]
    boundary_right = dom.boundary_points[dom.indices["3"]]
    assert np.allclose(uv(boundary_top), uv(boundary_bottom[::-1]))
    assert np.allclose(uv(boundary_left), uv(boundary_right[::-1]))
    print("Flow is doubly periodic")
    #  print psi, u and v at the corners
    print("U at corners: ", uv(corners).real)
    print("V at corners: ", uv(corners).imag)
    print("Psi at corners: ", psi(corners))
    left_flux = quad(lambda y: uv(-1 + 1j * y).real, -1, 1)[0]
    right_flux = quad(lambda y: uv(1 + 1j * y).real, -1, 1)[0]
    top_flux = quad(lambda x: uv(x + 1j).imag, -1, 1)[0]
    bottom_flux = quad(lambda x: uv(x - 1j).imag, -1, 1)[0]
    print(f"Left flux is {left_flux}")
    print(f"Right flux is {right_flux}")
    print(f"Top flux is {top_flux}")
    print(f"Bottom flux is {bottom_flux}")


prob = Problem()
corners = [1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j]
prob.add_exterior_polygon(
    corners, num_edge_points=300, num_poles=0, deg_poly=40, spacing="linear"
)
# prob.domain.plot()
# plt.show()
R = 0.1
prob.add_interior_curve(
    lambda t: -0.2 + 0.4j + R * np.exp(2j * np.pi * t),
    num_points=300,
    deg_laurent=60,
)
prob.add_interior_curve(
    lambda t: 0.2 - 0.4j + R * np.exp(2j * np.pi * t),
    num_points=300,
    deg_laurent=60,
)
horizontal_flux = 1
vertical_flux = 0
prob.add_boundary_condition("0", "u[0]-u[2][::-1]", 0)
prob.add_boundary_condition("0", "psi[0]-psi[2][::-1]", horizontal_flux)
prob.add_boundary_condition("1", "v[1]-v[3][::-1]", 0)
prob.add_boundary_condition("1", "psi[1]-psi[3][::-1]", vertical_flux)
prob.add_boundary_condition("4", "u[4]", 0)
prob.add_boundary_condition("4", "v[4]", 0)
prob.add_boundary_condition("5", "u[5]", 0)
prob.add_boundary_condition("5", "v[5]", 0)
solver = Solver(prob)
sol = solver.solve(check=False, weight=False, normalize=False)
print(
    f"Residual is {np.linalg.norm(solver.A @ solver.coefficients - solver.b)}"
)
an = Analysis(prob, sol)
check_flow(prob, sol)
sol.stress_discrete(1 + 1j)
fig, ax = an.plot(resolution=100)
# fig, ax =an.plot_periodic(1.0, 1.0, gapa=vertical_flux, gapb=horizontal_flux)
plt.show()
