"""Flow a domain with a circular interior curve.""" ""
from pylars import Problem, Solver, Analysis
import matplotlib.pyplot as plt
import numpy as np


def get_ellipse(a, b, theta):
    """Return a parametric function for an ellipse."""
    ellipse = lambda t: (
        a * np.cos(2 * np.pi * t) + b * 1j * np.sin(2 * np.pi * t)
    ) * np.exp(1j * theta)
    return ellipse


if __name__ == "__main__":
    prob = Problem()
    corners = [-1 - 1j, 1 - 1j, 1 + 1j, -1 + 1j]
    prob.add_exterior_polygon(
        corners,
        num_edge_points=500,
        num_poles=0,
        deg_poly=100,
        spacing="linear",
    )
    a, b = 0.5, 0.18
    theta = np.pi / 2
    prob.add_interior_curve(
        get_ellipse(a, b, theta),
        num_points=300,
        deg_laurent=0,
        centroid=0.0 + 0.0j,
        aaa=True,
        aaa_mmax=None,
    )
    prob.add_point(-1 - 1j)
    prob.domain.plot()
    plt.tight_layout()
    plt.show()

    prob.add_boundary_condition("0", "u[0]", 0)
    prob.add_boundary_condition("0", "v[0]", 0)
    prob.add_boundary_condition("2", "u[2]", 0)
    prob.add_boundary_condition("2", "v[2]", 0)
    prob.add_boundary_condition("1", "u[1]-u[3][::-1]", 0)
    prob.add_boundary_condition("1", "v[1]-v[3][::-1]", 0)
    prob.add_boundary_condition("3", "p[1]-p[3][::-1]", 2 / 0.09)
    prob.add_boundary_condition("3", "e12[1]-e12[3][::-1]", 0)
    prob.add_boundary_condition("4", "u[4]", 0)
    prob.add_boundary_condition("4", "v[4]", 0)
    prob.add_boundary_condition("5", "p[5]", 0)
    prob.add_boundary_condition("5", "psi[5]", 0)

    solver = Solver(prob, verbose=True)
    sol = solver.solve(check=False, normalize=False, weight=False)
    print(
        f"Residual: {np.abs(solver.A @ solver.coefficients - solver.b).max()}"
    )
    an = Analysis(sol)
    fig, ax = an.plot(resolution=200, interior_patch=True, enlarge_patch=1.0)
    plt.savefig("media/circle_flow.pdf")
    plt.show()
