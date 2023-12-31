"""Solve poiseuille flow with stream function boundary conditions."""
from pylars import Problem, Solver, Analysis
import numpy as np
import matplotlib.pyplot as plt
import time


def generate_normal_circles(n_circles, mean, std):
    """Generate non-overlapping circles."""
    L = 2 - 1.5 * (mean + 5 * std)
    radii = np.array(np.random.normal(mean, std, 1))
    centroids = np.array(
        L * np.random.rand(1) - L / 2 + 1j * (L * np.random.rand(1) - L / 2)
    )
    n_current = 1
    radius = np.random.normal(mean, std, 1)
    while n_current < n_circles:
        centroid = (
            L * np.random.rand(1)
            - L / 2
            + 1j * (L * np.random.rand(1) - L / 2)
        )
        if np.min(np.abs(centroid - centroids) / (radii + radius)) > 1.5:
            centroids = np.append(centroids, centroid)
            radii = np.append(radii, radius)
            n_current += 1
            radius = np.random.normal(mean, std, 1)
    return centroids, radii


def generate_circle_grid(sqrt_n_circles, radius):
    # create an evenly spaced grid of circles
    centroids = np.linspace(-1, 1, sqrt_n_circles + 2)[1:-1]
    centroids = centroids[:, np.newaxis] + 1j * centroids[np.newaxis, :]
    centroids = centroids.flatten()
    centroids += (np.random.rand(len(centroids)) - 0.5) * 0.5
    radii = np.ones_like(centroids).astype(np.float64) * radius
    return centroids, radii


def get_permeability():
    # create a square domain
    prob = Problem()
    corners = np.array([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j])
    prob.add_exterior_polygon(
        corners=corners,
        num_edge_points=150,
        deg_poly=30,
        num_poles=0,
        spacing="linear",
    )
    # centroids, radii = generate_circle_grid(n_circles, 0.05)
    # centroids, radii = generate_normal_circles(n_circles, 0.03, 0.00)
    R = np.sqrt(0.4 / np.pi)
    centroids = np.array([0.0 + 0.0j])
    radii = np.array([1]) * R
    # centroids = np.array([0.5 + 0.3j, 0.5 - 0.5j, -0.5 - 0.3j, -0.8 + 0.5j])
    # radii = np.array([1, 1, 1, 1]) * R
    n_circles = len(centroids)
    print("Circles generated")
    for centroid, radius in zip(centroids, radii):
        prob.add_interior_curve(
            lambda t: centroid + radius * np.exp(2j * np.pi * t),
            num_points=200,
            deg_laurent=10,
            centroid=centroid,
            mirror_laurents=False,
            mirror_tol=1.0,
        )
    prob.add_point(-1 - 1j)

    # top and bottom periodicity
    p_drop = 2.0
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
    prob.add_boundary_condition(f"{4+n_circles}", f"p[{4+n_circles}]", 0)
    prob.add_boundary_condition(f"{4+n_circles}", f"psi[{4+n_circles}]", 0)
    # prob.add_boundary_condition(f"{4+n_circles}", f"p[{4+n_circles}]", 2)
    # prob.add_boundary_condition(f"{4+n_circles}", f"psi[{4+n_circles}]", 0)

    solver = Solver(prob, verbose=True)
    print("Solving the problem")
    sol = solver.solve(check=False, weight=False, normalize=False)
    print(f" Error is {solver.max_error}")
    an = Analysis(sol)
    prob.domain.plot()
    plt.show()
    an.plot(resolution=300, interior_patch=True, quiver=True)
    plt.show()
    curve = lambda t: 1 + 2j * t - 1j
    curve_deriv = lambda t: 2j * np.ones_like(t)
    return an.get_permeability(curve, curve_deriv, delta_x=2, delta_p=p_drop)


if __name__ == "__main__":
    permeability = get_permeability()
    print(f"Permeability is ", permeability)
