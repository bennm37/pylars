"""Compare stress data to COMSOL."""
import pandas as pd
import matplotlib.pyplot as plt
import time
from pylars import Problem, Solver, Analysis
import numpy as np

plt.style.use("ggplot")


def load_comsol_data(center=0.2):
    """Load COMSOL data."""
    mesh_sizes = [
        "coarse",
        "normal",
        "fine",
        "finer",
        "extra_fine",
        "extremely_fine",
    ]
    data = {}
    for size in mesh_sizes:
        data[size] = [
            pd.read_csv(
                f"data/stress_data/COMSOL_mesh_{size}_center_{center:.1f}_stress_x.txt"
            )
        ]
        data[size].append(
            pd.read_csv(
                f"data/stress_data/COMSOL_mesh_{size}_center_{center:.1f}_stress_y.txt"
            )
        )
    return data


def get_pylars_solution(center=0.2j):
    start = time.perf_counter()
    prob = Problem()
    corners = [-1 - 1j, 1 - 1j, 1 + 1j, -1 + 1j]
    prob.add_exterior_polygon(
        corners,
        num_edge_points=500,
        num_poles=0,
        deg_poly=40,
        spacing="linear",
    )
    prob.add_interior_curve(
        lambda t: center + 0.5 * np.exp(2j * np.pi * t),
        num_points=300,
        deg_laurent=30,
        centroid=center,
        mirror_laurents=True,
        mirror_tol=2,
    )
    prob.domain.plot(set_lims=False)
    plt.show()
    prob.add_point(-1 + 1j)
    # prob.domain.plot()
    # plt.show()
    prob.add_boundary_condition("0", "u[0]", 0)
    prob.add_boundary_condition("0", "v[0]", 0)
    prob.add_boundary_condition("2", "u[2]", 0)
    prob.add_boundary_condition("2", "v[2]", 0)
    prob.add_boundary_condition("3", "u[3]-u[1][::-1]", 0)
    prob.add_boundary_condition("3", "v[3]-v[1][::-1]", 0)
    prob.add_boundary_condition("1", "p[3]-p[1][::-1]", 1)
    prob.add_boundary_condition("1", "e12[3]-e12[1][::-1]", 0)
    prob.add_boundary_condition("4", "u[4]", 0)
    prob.add_boundary_condition("4", "v[4]", 0)
    prob.add_boundary_condition("5", "p[5]", 0)
    prob.add_boundary_condition("5", "psi[5]", 0)
    solver = Solver(prob)
    sol = solver.solve(check=False, normalize=False, weight=False)
    print("Error: ", solver.max_error)
    end = time.perf_counter()
    print(str(end - start) + " seconds")
    return sol


center = 0.2j
sol = get_pylars_solution(center=center)

# calculate normal stress data
circle = lambda t: center + 0.5 * np.exp(t * 1j)
normal = lambda t: -np.exp(t * 1j)
theta = np.linspace(-np.pi, np.pi, 300)
z = circle(theta)
stress_tensor = sol.stress_goursat(z)
normals = normal(np.array(theta))
normals = np.array([normals.real, normals.imag]).T

# plotting normals
an = Analysis(sol)
fig, ax = an.plot(
    resolution=201,
    interior_patch=True,
    enlarge_patch=1.02,
    streamline_type="starting_points",
    n_streamlines=20,
)
ax.quiver(
    z.real,
    z.imag,
    normals[:, 0],
    normals[:, 1],
    color="k",
    zorder=10,
)
plt.show()


traction = np.array([S @ n for n, S in zip(normals, stress_tensor)])
pylars_data = traction
fig, ax = plt.subplots()
ax.plot(theta, pylars_data[:, 0], label="pylars x")
ax.plot(theta, pylars_data[:, 1], label="pylars y")
plt.show()
