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


def get_relative_error(dom, func, drop_lr=0, drop_tb=0, atol=1e-16):
    top = dom.boundary_points[dom.indices["0"]]
    left = dom.boundary_points[dom.indices["1"]]
    bot = top - 2j
    right = left + 2
    rel_max_error_lr = np.max(
        np.abs((drop_lr - (func(left) - func(right))) / (func(right) + atol))
    )
    rel_max_error_tb = np.max(
        np.abs((drop_tb - (func(top) - func(bot))) / (func(top) + atol))
    )
    rel_max_error = np.max((rel_max_error_lr, rel_max_error_tb))
    return rel_max_error


def get_residual_time():
    # create a square domain
    shift = 0.0 + 0.0j
    corners = np.array([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j])
    corners += shift
    prob = Problem()
    prob.add_periodic_domain(
        length=2,
        height=2,
        num_edge_points=150,
        deg_poly=100,
        num_poles=0,
        spacing="linear",
    )
    centroid = shift + 0.5 - 0.5j
    n_circles = 30
    np.random.seed(0)
    centroids, radii = generate_normal_circles(n_circles, 0.03, 0.00)
    print("Circles generated")
    for centroid, radius in zip(centroids, radii):
        prob.add_periodic_curve(
            lambda t: centroid + radius * np.exp(2j * np.pi * t),
            num_points=80,
            deg_laurent=20,
            centroid=centroid,
            image_laurents=True,
            mirror_laurents=False,
            mirror_tol=1.0,
        )
    prob.add_point(shift + -1 - 1j)

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
    prob.add_boundary_condition(f"{4+n_circles}", f"p[{4+n_circles}]", 2)
    prob.add_boundary_condition(f"{4+n_circles}", f"psi[{4+n_circles}]", 0)

    start = time.perf_counter()
    solver = Solver(prob, verbose=True)
    print("Solving the problem")
    sol = solver.solve(check=False, weight=False, normalize=False)
    abs_residual = np.max(np.abs(solver.A @ solver.coefficients - solver.b))
    end = time.perf_counter()
    dom = sol.problem.domain
    ATOL = 0.0
    e12 = lambda z: sol.eij(z)[0, 1]
    rel_eij = get_relative_error(dom, e12, atol=ATOL)
    rel_p = get_relative_error(dom, sol.p, drop_lr=p_drop, atol=ATOL)
    rel_uv = get_relative_error(dom, sol.uv, atol=ATOL)
    rel_max_error = np.max((rel_eij, rel_p, rel_uv))
    time_taken = end - start
    print(f"Time taken: {time_taken:.2f}s")
    print(f"Absolute Residual: {abs_residual:.15e}")
    print(f"Relative Maximum Error: {rel_max_error:.15e}")
    return time_taken, abs_residual, rel_max_error


if __name__ == "__main__":
    time_taken, abs_residual, rel_residual = get_residual_time()
