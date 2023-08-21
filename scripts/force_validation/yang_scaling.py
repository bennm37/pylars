"""Flow a domain with a circular interior curve.""" ""
from pylars import Problem, Solver, Analysis
from pylars.simulation import Mover
import matplotlib.pyplot as plt
import numpy as np


def get_force_torque(
    velocity,
    angular_velocity,
    centroid=0.0 + 0.0j,
    radius=0.5,
    deg_laurent=50,
    inlet_profile="1-y**2",
    resolution=600,
    colorbar=False,
    l=1,
    plot=False,
):
    prob = Problem()
    s = 8
    corners = [
        s * l + 1j * l,
        -s * l + 1j * l,
        -s * l - 1j * l,
        s * l - 1j * l,
    ]
    prob.add_exterior_polygon(
        corners=corners,
        num_edge_points=1000,
        num_poles=0,
        deg_poly=50,
        spacing="linear",
    )
    circle = lambda t: centroid + radius * np.exp(2j * np.pi * t)  # noqa: E731
    circle_deriv = (
        lambda t: 2j * radius * np.pi * np.exp(2j * np.pi * t)
    )  # noqa: E731
    num_points = 500
    mover = Mover(
        circle,
        circle_deriv,
        centroid,
        velocity=velocity,
        angular_velocity=angular_velocity,
    )
    prob.add_mover(
        mover,
        num_points=num_points,
        deg_laurent=deg_laurent,
        mirror_laurents=True,
        mirror_tol=10,
    )
    prob.add_boundary_condition("0", "u[0]", 0)
    prob.add_boundary_condition("0", "v[0]", 0)
    prob.add_boundary_condition("2", "u[2]", 0)
    prob.add_boundary_condition("2", "v[2]", 0)
    prob.add_boundary_condition("1", "u[1]", 0)
    prob.add_boundary_condition("1", "v[1]", 0)
    prob.add_boundary_condition("3", "u[3]", 0)
    prob.add_boundary_condition("3", "v[3]", 0)
    solver = Solver(prob)
    sol = solver.solve(normalize=False, weight=False)
    print(
        f"residual = {np.abs(solver.A @ solver.coefficients - solver.b).max()}"
    )
    if plot:
        prob.domain.plot(set_lims=False)
        plt.show()
        an = Analysis(sol)
        fig, ax = an.plot(
            resolution=resolution,
            interior_patch=True,
            quiver=False,
            streamline_type="linear",
            colorbar=colorbar,
            enlarge_patch=1.0,
            n_streamlines=20,
            imshow=True,
        )
        plt.tight_layout()
        plt.show()
    F = sol.force(mover.curve, mover.deriv)
    T = sol.torque(mover.curve, mover.deriv, mover.centroid)
    return F, T


def get_data(filename="yang_scaling.npz", n_samples=10):
    l = 1
    omega = 1
    Rs = np.array([0.29, 0.6, 0.8, 0.9])
    tol = 1e-2
    centroids = 0.0 + 1j * np.linspace(0, l - Rs - tol, n_samples).T
    force_results = np.zeros(centroids.shape, dtype=np.complex128)
    torque_results = np.zeros(centroids.shape, dtype=np.float64)
    for i, R in enumerate(Rs):
        print(f"Starting {R = }")
        cases = [
            {
                "velocity": 0.0,
                "angular_velocity": omega,
                "centroid": centroid,
                "radius": R,
            }
            for centroid in centroids[i]
        ]
        for j, case in enumerate(cases):
            if j % 5 == 0:
                print(f"case {j} of {len(cases)}")
            F, T = get_force_torque(**case)
            force_results[i, j] = F
            torque_results[i, j] = T
    np.savez(
        filename,
        force_results=force_results,
        torque_results=torque_results,
        centroids=centroids,
        Rs=Rs,
        omega=omega,
        l=l,
    )


def plot_data(
    filename="data/yang_scaling.npz", plotname="media/yang_scaling.pdf"
):
    data = np.load(filename)
    force_results = data["force_results"]
    torque_results = data["torque_results"]
    centroids = data["centroids"]
    Rs = data["Rs"]
    l = data["l"]
    omega = data["omega"]
    ks = Rs / l
    # generate scaling solution data
    CF = -1.14
    CT = -np.pi * np.sqrt(2)
    eps_bs = np.array(
        [(l - R + centroids[i].imag) / R for i, R in enumerate(Rs)]
    )
    eps_ts = np.array(
        [(l - R - centroids[i].imag) / R for i, R in enumerate(Rs)]
    )
    force_scaling = np.array(
        [
            CF
            * 24
            * omega
            * R
            * (eps_bs[i] - eps_ts[i])
            / (eps_bs[i] ** (5 / 2) + eps_ts[i] ** (5 / 2))
            for i, R in enumerate(Rs)
        ]
    )
    torque_scaling = np.array(
        [
            2
            * CT
            * omega
            * R**2
            * (1 / (eps_bs[i] ** 0.5) + 1 / (eps_ts[i] ** 0.5))
            for i, R in enumerate(Rs)
        ]
    )
    plt.style.use("ggplot")
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    scaling_labels = [None, None, None, "scaling"]
    for i, k in enumerate(ks):
        e_o_emax = centroids[i].imag / (l - Rs[i])
        ax[0].plot(
            e_o_emax,
            -((1 / k - 1) ** 1.5) * force_results[i].real / (omega * Rs[i]),
            "--.",
            label=f"$k={k}$",
        )
        ax[0].plot(
            e_o_emax,
            -((1 / k - 1) ** 1.5) * force_scaling[i] / (omega * Rs[i]),
            color="k",
            label=scaling_labels[i],
        )
    ax[0].set_xlabel("$e/e_{max}$")
    ax[0].set_ylabel("$(1/k-1)^{3/2}) F_x/(\mu\omega R)$")
    ax[0].legend()
    for i, k in enumerate(ks):
        e_o_emax = centroids[i].imag / (l - Rs[i])
        ax[1].plot(
            e_o_emax,
            -((1 / k - 1) ** 0.5)
            * torque_results[i].real
            / (4 * np.pi * omega * Rs[i] ** 2),
            "--.",
            label=f"$k={k}$",
        )
        ax[1].plot(
            e_o_emax,
            -((1 / k - 1) ** 0.5)
            * torque_scaling[i]
            / (4 * np.pi * omega * Rs[i] ** 2),
            color="k",
            label=scaling_labels[i],
        )
    ax[1].set_xlabel("$e/e_{max}$")
    ax[1].set_ylabel("$(1/k-1)^{3/2}) F_x/(\mu\omega R)$")
    ax[1].legend()
    fig.suptitle(
        "Comparison of Force and Torque to Scaling Solution from Yang et. al."
    )
    plt.savefig(plotname)
    plt.show()


if __name__ == "__main__":
    # get_force_torque(
    #     **{
    #         "velocity": 0.0,
    #         "angular_velocity": 1.0,
    #         "centroid": 0.09j,
    #         "radius": 0.9,
    #         "plot": True,
    #         "resolution": 400,
    #         "colorbar": True,
    #     }
    # )
    # get_data(
    #     filename="data/yang_scaling.npz",
    #     n_samples=20,
    # )
    plot_data(
        filename="data/yang_scaling.npz", plotname="media/yang_scaling.pdf"
    )
