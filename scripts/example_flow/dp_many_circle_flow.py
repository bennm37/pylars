"""Solve poiseuille flow with stream function boundary conditions."""
from pylars import Problem, Solver, Analysis
import numpy as np
import matplotlib.pyplot as plt
from circle_flow_2 import generate_normal_circles
import time

# create a square domain
# k = 100
# deg_poly = k
# num_edge_points = 4 * k
# deg_laurent = int(0.2 * k)
# num_points = int(0.5 * k)
num_edge_points = 350
deg_poly = 50
num_points = 200
deg_laurent = 15
print(f"deg_poly: {deg_poly}, deg_laurent: {deg_laurent}")
corners = np.array([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j])
prob = Problem()
prob.add_periodic_domain(
    length=2,
    height=2,
    num_edge_points=num_edge_points,
    deg_poly=deg_poly,
    num_poles=0,
    spacing="linear",
)
n_circles = 30
np.random.seed(4)
shift = -0.0
centroids, radii = generate_normal_circles(n_circles, 0.05, 0.00)
# centroids = [0.80356341 + 0.36518719j, 0.336139 - 0.48264786j]
# centroids = [(-0.5 + 0.5j), (0.5 - 0.49j)]
# radii = [0.2, 0.2]
print("Circles generated")
for centroid, radius in zip(centroids, radii):
    centroid += shift
    prob.add_periodic_curve(
        lambda t: centroid + radius * np.exp(2j * np.pi * t),
        num_points=num_points,
        deg_laurent=deg_laurent,
        centroid=centroid,
        image_laurents=True,
        image_tol=0.3,
        mirror_laurents=False,
        mirror_tol=0.5,
    )
prob.add_point(-1 - 1j)
prob.domain.plot(set_lims=False)
plt.show()

# top and bottom periodicity
p_drop = 60
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
    prob.add_boundary_condition(f"{interior}", f"u[{interior}]", 0.0)
    prob.add_boundary_condition(f"{interior}", f"v[{interior}]", 0)
prob.add_boundary_condition(f"{4 + n_circles}", f"p[{4 + n_circles}]", 0)
prob.add_boundary_condition(f"{4 + n_circles}", f"psi[{4 + n_circles}]", 0)
start = time.perf_counter()
solver = Solver(prob, verbose=True)
print("Solving the problem")
sol = solver.solve(check=False, weight=False, normalize=False)
abs_residual = np.max(np.abs(solver.A @ solver.coefficients - solver.b))
rel_residual = np.max(
    np.abs(solver.A @ solver.coefficients - solver.b) / np.abs(solver.b)
)
end = time.perf_counter()
print(f"Time taken: {end-start:.2f}s")
print(f"Absolute Residual: {abs_residual:.15e}")
# sol.problem.domain._update_polygon(buffer=1e-5)
an = Analysis(sol)
fig, ax = an.plot_periodic(interior_patch=True, enlarge_patch=1.0)
plt.show()

# continuity checks
dom = sol.problem.domain
points_0 = dom.boundary_points[dom.indices["0"]]
points_1 = dom.boundary_points[dom.indices["1"]]
fig, ax = plt.subplots()
plt.plot(
    points_0.real,
    np.abs(sol.eij(points_0)[:, 0, 1] - sol.eij(points_0 - 2j)[:, 0, 1]),
    label="e12 tb",
)
plt.plot(
    points_1.imag,
    np.abs(sol.eij(points_1)[:, 0, 1] - sol.eij(points_1 + 2)[:, 0, 1]),
    label="e12 lr",
)
plt.plot(
    points_0.real,
    np.abs(sol.eij(points_0)[:, 0, 0] - sol.eij(points_0 - 2j)[:, 0, 0]),
    label="e11 tb",
)
plt.plot(
    points_1.imag,
    np.abs(sol.eij(points_1)[:, 0, 0] - sol.eij(points_1 + 2)[:, 0, 0]),
    label="e11 lr",
)
plt.plot(
    points_0.real,
    np.abs(sol.p(points_0) - sol.p(points_0 - 2j)),
    label="p tb",
)
plt.plot(
    points_1.imag,
    np.abs(sol.p(points_1) - sol.p(points_1 + 2) - p_drop),
    label="p lr",
)

plt.legend()
plt.show()
