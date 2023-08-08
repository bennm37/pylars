"""Flow a domain with a circular interior curve.""" ""
from pylars import Problem, Solver, Analysis
import matplotlib.pyplot as plt
import numpy as np
from circle_flow_2 import generate_normal_circles

if __name__ == "__main__":
    prob = Problem()
    corners = [-1 - 1j, 1 - 1j, 1 + 1j, -1 + 1j]
    prob.add_exterior_polygon(
        corners,
        num_edge_points=500,
        num_poles=0,
        deg_poly=75,
        spacing="linear",
    )
    # centroids = [0.4 + 0.5j, 0.5 - 0.6j, -0.5 - 0.5j, -0.5 + 0.5j]
    # np.random.seed(0)
    # n_circles = 10
    # centroids, radii = generate_normal_circles(n_circles, 0.03, 0.01)
    n_circles = 2
    centroids = [(-0.1 + 0.5j), (0.3 + 0.2j)]
    radii = [0.2, 0.2]
    print("Circles generated")
    for centroid, radius in zip(centroids, radii):
        prob.add_interior_curve(
            lambda t: centroid + radius * np.exp(2j * np.pi * t),
            num_points=250,
            deg_laurent=20,
            centroid=centroid,
            mirror_laurents=True,
        )

    prob.add_boundary_condition("0", "u[0]", 0)
    # prob.add_boundary_condition("0", "psi[0]", -2 / 3)
    prob.add_boundary_condition("0", "v[0]", 0)
    prob.add_boundary_condition("2", "u[2]", 0)
    prob.add_boundary_condition("2", "v[2]", 0)
    # prob.add_boundary_condition("2", "psi[2]", 2 / 3)
    prob.add_boundary_condition("1", "u[1]", "1-y**2")
    prob.add_boundary_condition("1", "v[1]", 0)
    prob.add_boundary_condition("3", "p[3]", 0)
    prob.add_boundary_condition("3", "v[3]", 0)
    interiors = [str(i) for i in range(4, 4 + n_circles)]
    for interior in interiors:
        prob.add_boundary_condition(f"{interior}", f"u[{interior}]", 0)
        prob.add_boundary_condition(f"{interior}", f"v[{interior}]", 0)

    solver = Solver(prob, verbose=True)
    sol = solver.solve(check=False, normalize=False, weight=False)
    an = Analysis(sol)
    # sol.problem.domain.enlarge_holes(1.0)
    print(
        f"Residual: {np.abs(solver.A @ solver.coefficients - solver.b).max()}"
    )
    fig, ax = an.plot(resolution=200, interior_patch=True, enlarge_patch=1.1)
    plt.show()
