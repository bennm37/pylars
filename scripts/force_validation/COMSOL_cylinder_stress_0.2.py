"""Compare stress data to COMSOL."""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import time
from pylars import Problem, Solver, Analysis
import numpy as np

# plt.style.use("ggplot")
# plt.rcParams.update({"text.usetex": True, "font.family": "Computer Modern"})
# matplotlib.rcParams["mathtext.fontset"] = "custom"
# matplotlib.rcParams["mathtext.rm"] = "Bitstream Vera Sans"
# matplotlib.rcParams["mathtext.it"] = "Bitstream Vera Sans:italic"
# matplotlib.rcParams["mathtext.bf"] = "Bitstream Vera Sans:bold"


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
            np.loadtxt(
                f"data/periodic_stress_data/COMSOL_mesh_{size}_center_{center:.1f}_stress_x.txt",
                delimiter=",",
            )
        ]
        data[size].append(
            np.loadtxt(
                f"data/periodic_stress_data/COMSOL_mesh_{size}_center_{center:.1f}_stress_y.txt",
                delimiter=",",
            )
        )
    return data


def get_pylars_solution(center=0.2j):
    start = time.perf_counter()
    prob = Problem()
    corners = [-1 - 1j, 1 - 1j, 1 + 1j, -1 + 1j]
    prob.add_exterior_polygon(
        corners,
        num_edge_points=400,
        num_poles=0,
        deg_poly=40,
        spacing="linear",
    )
    prob.add_interior_curve(
        lambda t: center + 0.5 * np.exp(2j * np.pi * t),
        num_points=200,
        deg_laurent=30,
        centroid=center,
    )
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
    prob.add_boundary_condition("5", "psi[5]", 100)
    solver = Solver(prob)
    sol = solver.solve(check=False, normalize=False, weight=False)
    print("Error: ", solver.max_error)
    end = time.perf_counter()
    print(str(end - start) + " seconds")
    return sol


if __name__ == "__main__":
    center = 0.2j
    data = load_comsol_data(center=center.imag)
    sol = get_pylars_solution(center=center)
    an = Analysis(sol)
    fig, ax = an.plot(
        resolution=201,
        interior_patch=True,
        enlarge_patch=1.02,
        streamline_type="starting_points",
        n_streamlines=20,
    )
    plt.show()

    # calculate traction data
    mesh_sizes = data.keys()
    circle = lambda t: center + 0.5 * np.exp(t * 1j)
    normal = lambda t: -np.exp(t * 1j)
    pylars_data = {}
    # mesh_sizes = ["extremely_fine"]
    for size in mesh_sizes:
        theta = data[size][0][:, 0]
        z = circle(np.array(theta))
        stress_tensor = sol.stress_goursat(z)
        normals = normal(np.array(theta))
        normals = np.array([normals.real, normals.imag]).T
        surface_stress = np.array(
            [S @ n for n, S in zip(normals, stress_tensor)]
        )
        pylars_data[size] = surface_stress
    size = "extremely_fine"
    # for size in mesh_sizes:
    for i in range(1):
        fig, ax = plt.subplots(2, 2)
        fig.set_size_inches(10, 5)
        sample = 1
        theta = np.array(data[size][0][:, 0])
        # theta = np.linspace(0, 2 * np.pi, 100)
        stress_x = np.array(data[size][0][:, 1])
        stress_y = np.array(data[size][1][:, 1])
        pylars_stress_x = pylars_data[size][:, 0]
        pylars_stress_y = pylars_data[size][:, 1]
        ax[0, 0].plot(theta[::sample], stress_x[::sample], label="$T_x^C$")
        ax[0, 0].plot(theta[::sample], stress_y[::sample], label="$T_y^C$")
        ax[0, 0].plot(
            theta,
            pylars_stress_x,
            c="k",
            linestyle="--",
            label="$T_x^P$",
        )
        ax[0, 0].plot(
            theta,
            pylars_stress_y,
            linestyle="-.",
            c="k",
            label="$T_y^P$",
        )
        ax[0, 0].set(
            xlabel=" $\Theta$ ",
            ylabel="Traction",
            title="Traction on Cylinder",
        )
        ax[0, 0].legend(loc="upper right")
        ax[0, 1].plot(theta[::sample], stress_x[::sample], label="$T_x^C$")
        ax[0, 1].plot(theta[::sample], stress_y[::sample], label="$T_y^C$")
        ax[0, 1].plot(
            theta,
            pylars_stress_x,
            c="k",
            linestyle="--",
            label="$T_x^P$",
        )
        ax[0, 1].plot(
            theta,
            pylars_stress_y,
            linestyle="-.",
            c="k",
            label="RS $T_y^P$",
        )
        ax[0, 1].set(
            xlabel=" $\Theta$ ",
            ylabel="Traction",
            xlim=(-1.8, -1.3),
            ylim=(-0.45, -0.3),
            title="Traction on Cylinder",
        )
        ax[0, 1].legend(loc="upper right")
        sample = 10
        ax[1, 0].semilogy(
            theta[::sample],
            np.abs(stress_x[::sample] - pylars_stress_x[::sample]),
            label="$T_x$",
        )
        ax[1, 0].semilogy(
            theta[::sample],
            np.abs(stress_y[::sample] - pylars_stress_y[::sample]),
            label="$T_y$",
        )
        ax[1, 0].set(
            xlabel=" $\Theta$ ",
            ylabel="Difference",
            title="Difference in Traction",
        )
        ax[1, 0].legend()
        # plt.savefig("media/COMSOL_vs_PyLARS_stress.pdf")
        # plot convergence of the max error
        max_errors_x = [
            np.max(
                np.abs(np.array(data[size][0][:, 1]) - pylars_data[size][:, 0])
            )
            for size in mesh_sizes
        ]
        max_errors_y = [
            np.max(
                np.abs(np.array(data[size][1][:, 1]) - pylars_data[size][:, 1])
            )
            for size in mesh_sizes
        ]
        ax[1, 1].plot(max_errors_x, label="$T_x$")
        ax[1, 1].plot(max_errors_y, label="$T_y$")
        ax[1, 1].set_xlabel("Mesh Size")
        ax[1, 1].set_ylabel("Max Difference")
        ax[1, 1].set_xticks(range(len(mesh_sizes)))
        labels = [f"{size.split('_')[0]}" for size in mesh_sizes]
        ax[1, 1].set_xticklabels(labels)
        ax[1, 1].set_title("Convergence of Max Difference")
        ax[1, 1].legend()
        plt.tight_layout()
        plt.savefig(f"media/COMSOL_vs_PyLARS_stress_{size}.pdf")
        plt.show()
