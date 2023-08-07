"""Solve Poiseiulle flow with stream function boundary conditions."""
from pylars import Problem, Solver, Analysis
import numpy as np
import matplotlib.pyplot as plt
from circle_flow_2 import generate_normal_circles
import time

# create a square domain
shift = 0.0 + 0.0j
corners = np.array([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j])
corners += shift
prob = Problem()
prob.add_periodic_domain(
    length=2,
    height=2,
    num_edge_points=400,
    deg_poly=50,
    num_poles=0,
    spacing="linear",
)
centroid = shift + 0.5 - 0.5j
n_circles = 10
np.random.seed(14)
centroids, radii = generate_normal_circles(n_circles, 0.03, 0.00)
print("Circles generated")
for centroid, radius in zip(centroids, radii):
    prob.add_periodic_curve(
        lambda t: centroid + radius * np.exp(2j * np.pi * t),
        num_points=200,
        deg_laurent=10,
        centroid=centroid,
        image_laurents=True,
        image_tol=0.1,
        mirror_laurents=False,
        mirror_tol=0.1,
    )
prob.add_point(shift + -1 - 1j)
prob.domain.plot(set_lims=False)
plt.show()

# top and bottom periodicity
p_drop = 2.0
prob.add_boundary_condition("0", "u[0]-u[2][::-1]", 0)
prob.add_boundary_condition("0", "v[0]-v[2][::-1]", 0)
prob.add_boundary_condition("2", "p[0]-p[2][::-1]", 0)
prob.add_boundary_condition("2", "e12[0]-e12[2][::-1]", 0)

prob.add_boundary_condition("1", "psi[1]-psi[3][::-1]", 0)
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
rel_residual = np.max(
    np.abs(solver.A @ solver.coefficients - solver.b) / np.abs(solver.b)
)
end = time.perf_counter()
print(f"Time taken: {end-start:.2f}s")
# relatieve_
# residual = np.max(
#     np.abs(solver.A @ solver.coefficients - solver.b)
#     / (np.abs(solver.b) + 1e-8)
# )
print(f"Absolute Residual: {abs_residual:.15e}")
# sol.problem.domain._update_polygon(buffer=1e-5)
sol.problem.domain.enlarge_holes(1.1)
an = Analysis(sol)
fig, ax = an.plot_periodic(interior_patch=True, quiver=True)
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
    np.abs(sol.p(points_0) - sol.p(points_0 - 2j)),
    label="p tb",
)
plt.plot(
    points_1.imag,
    np.abs(sol.p(points_1) - sol.p(points_1 + 2) - p_drop),
    label="p lr",
)

plt.legend()
# plt.plot(points.imag, sol.p(points+2), label="p right")
plt.show()
