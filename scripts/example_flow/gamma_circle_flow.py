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
    prob.add_exterior_polygon(
        corners,
        num_edge_points=200 * n_circles,
        num_poles=0,
        deg_poly=75,
        spacing="linear",
    )

    for centroid, radius in zip(centroids, radii):
        prob.add_interior_curve(
            lambda t: centroid + radius * np.exp(2j * np.pi * t),
            num_points=150,
            deg_laurent=40,
            centroid=centroid,
            mirror_laurents=True,
            mirror_tol=bound / 2,
        )
    prob.domain.plot(set_lims=False)
    plt.show()

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
    curve = lambda t: length + 2j * length * t - 1j * length
    curve_deriv = lambda t: 2j * np.ones_like(t) * length
    permeability = an.get_permeability(
        curve=curve, curve_deriv=curve_deriv, delta_x=length, delta_p=p_drop
    )
    print(f"Permeability: {permeability}")
    fig, ax = an.plot(
        resolution=100, interior_patch=True, enlarge_patch=1.01, epsilon=0
    )
    plt.show()
    return permeability


def plot_pdf(rv, rv_args):
    """Plot the PDF of the random variable."""
    x = np.linspace(rv.ppf(0.001, **rv_args), rv.ppf(0.999, **rv_args), 100)
    y = rv.pdf(x=x, **rv_args)
    plt.plot(x, y)
    plt.show()


if __name__ == "__main__":
    prob = Problem()
    length = 2
    porosity = 0.95
    p_drop = 100
    rv = gamma.rvs
    rv_args = {"a": 5.0, "scale": 0.05, "loc": 0.0}
    # plot_pdf(gamma, rv_args)
    np.random.seed(1)
    # centroids, radii = generate_rv_circles(
    #     porosity=porosity, rv=rv, rv_args=rv_args, length=length, min_dist=0.05
    # )
    centroids, radii = np.array([0 + 0j]), np.array([0.357])
    n_circles = len(centroids)
    print(f"Number of circles: {n_circles}")
    print(f"Porosity: {1 - np.sum(np.pi * radii**2) / length**2}")
    bound = length / 2
    sol = run_case(centroids, radii, bound, p_drop)
    an = Analysis(sol)
    permeability = an.get_permeability(
        lambda t: 1 + 2j * t - 1j, lambda t: 2j * np.ones_like(t), 2, p_drop
    )
    print(f"Permeability: {permeability}")
    an.plot()
    plt.show()
    U, L, mu = 1e-6, 1e-10, 1e-3
    dim_p_drop = p_drop * U * mu / L
    dim_length = length * L
    dim_sol = sol.dimensionalize(U=U, L=L, mu=mu)
    x = np.linspace(-1e-6, 1e-6, 100)
    y = np.linspace(-1e-6, 1e-6, 100)
    X, Y = np.meshgrid(x, y)
    # permeability = analayse_case(dim_sol, dim_length, dim_p_drop)
