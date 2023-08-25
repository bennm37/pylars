"""Simulate flow past log normally distributed circles."""
from pylars import Problem, Solver, Analysis
from pylars.domain import generate_rv_circles
from scipy.stats import gamma
import matplotlib.pyplot as plt
import numpy as np


def run_case(centers, radii, bound, p_drop):
    """Run doubly periodic flow past the circles."""
    corners = [
        bound + bound * 1j,
        -bound + bound * 1j,
        -bound - bound * 1j,
        bound - bound * 1j,
    ]
    prob = Problem()
    prob.add_exterior_polygon(
        corners,
        num_edge_points=100 * n_circles,
        num_poles=0,
        deg_poly=50,
        spacing="linear",
    )

    for centroid, radius in zip(centroids, radii):
        prob.add_interior_curve(
            lambda t: centroid + radius * np.exp(2j * np.pi * t),
            num_points=125,
            deg_laurent=30,
            centroid=centroid,
            mirror_laurents=True,
            mirror_tol=bound / 2,
        )
    prob.domain.plot(set_lims=False)
    plt.show()

    p_drop = 0.25
    prob.add_boundary_condition("0", "u[0]-u[2][::-1]", 0)
    prob.add_boundary_condition("0", "v[0]-v[2][::-1]", 0)
    prob.add_boundary_condition("2", "p[0]-p[2][::-1]", 0)
    prob.add_boundary_condition("2", "e12[0]-e12[2][::-1]", 0)

    prob.add_boundary_condition("1", "u[1]-u[3][::-1]", 0)
    prob.add_boundary_condition("1", "v[1]-v[3][::-1]", 0)
    prob.add_boundary_condition("3", "p[1]-p[3][::-1]", p_drop)
    prob.add_boundary_condition("3", "e12[1]-e12[3][::-1]", 0)
    interiors = [str(i) for i in range(4, 4 + n_circles)]
    for interior in interiors:
        prob.add_boundary_condition(f"{interior}", f"u[{interior}]", 0)
        prob.add_boundary_condition(f"{interior}", f"v[{interior}]", 0)

    solver = Solver(prob, verbose=True)
    sol = solver.solve(check=False, normalize=False, weight=False)
    print(
        f"Residual: {np.abs(solver.A @ solver.coefficients - solver.b).max()}"
    )
    return sol


def analayse_case(sol, length=2, p_drop=0.25):
    """Analyse the solution."""
    an = Analysis(sol)
    fig, ax = an.plot(resolution=100, interior_patch=True, enlarge_patch=1.01)
    plt.show()
    peremeability = an.get_permeability("3", length, length, p_drop)
    return peremeability


def plot_pdf(rv, rv_args):
    """Plot the PDF of the random variable."""
    x = np.linspace(rv.ppf(0.001, **rv_args), rv.ppf(0.999, **rv_args), 100)
    y = rv.pdf(x=x, **rv_args)
    plt.plot(x, y)
    plt.show()


if __name__ == "__main__":
    length = 10
    porosity = 0.95
    p_drop = 30
    rv = gamma.rvs
    rv_args = {"a": 5.0, "scale": 0.05, "loc": 0.0}
    # plot_pdf(gamma, rv_args)
    np.random.seed(1)
    centroids, radii = generate_rv_circles(
        porosity=porosity, rv=rv, rv_args=rv_args, length=length, min_dist=0.05
    )
    n_circles = len(centroids)
    print(f"Number of circles: {n_circles}")
    print(f"Porosity: {1 - np.sum(np.pi * radii**2) / length**2}")
    bound = length / 2
    sol = run_case(centroids, radii, bound, p_drop)
    sol.dimensionalize(U=1e-6, L=1e-6, mu=1e-3)
    permeability, wss_distribution = analayse_case(sol, length, p_drop)
    print(f"Permeability: {permeability}")
