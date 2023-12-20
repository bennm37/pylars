"""Flow a domain with a circular interior curve.""" ""
from pylars import Problem, Solver, Analysis
from pylars.simulation import Mover
from pylars.numerics import split_laurent
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams.update(
    {
        "pgf.texsystem": "pdflatex",
        "font.family": "serif",
        "text.usetex": True,
        "pgf.rcfonts": False,
    }
)


def get_force_torque_stokeslet(
    velocity,
    angular_velocity,
    centroid=0.0 + 0.0j,
    radius=0.5,
    deg_laurent=30,
    inlet_profile="1-y**2",
    resolution=600,
    colorbar=False,
    l=1,
    plot=False,
):
    """Calculate the force and torque on a circular interior curve."""
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
        num_edge_points=400,
        num_poles=0,
        deg_poly=30,
        spacing="linear",
    )
    circle = lambda t: centroid + radius * np.exp(2j * np.pi * t)  # noqa: E731
    circle_deriv = (
        lambda t: 2j * radius * np.pi * np.exp(2j * np.pi * t)
    )  # noqa: E731
    num_points = 300
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
    print(f"Error: {solver.max_error}")
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
    coeff = solver.coefficients
    interior_laurents = solver.domain.interior_laurents
    cf, cg, clf, clg = split_laurent(coeff, interior_laurents)
    F_stokeslet = -8 * np.pi * clf[0, 0]
    T_rotlet = 4 * np.pi * np.imag(clg[0, 0] + np.conj(centroid) * radius**3)
    return F, T, F_stokeslet, T_rotlet


if __name__ == "__main__":
    n_rad, n_cen = 3, 3
    radii = np.linspace(0.1, 0.4, n_rad)
    centroids = np.linspace(0, 0.1, n_cen) * 1j
    forces = np.zeros((n_rad, n_cen), dtype=np.complex128)
    torques = np.zeros((n_rad, n_cen), dtype=np.complex128)
    forces_stokeslet = np.zeros((n_rad, n_cen), dtype=np.complex128)
    torques_rotlet = np.zeros((n_rad, n_cen), dtype=np.complex128)
    for i, centroid in enumerate(centroids):
        for j, radius in enumerate(radii):
            F, T, F_S, T_R = get_force_torque_stokeslet(
                **{
                    "velocity": 0.0,
                    "angular_velocity": 1.0,
                    "centroid": centroid,
                    "radius": radius,
                    "plot": False,
                    "resolution": 400,
                    "colorbar": True,
                }
            )
            forces[i, j] = F
            torques[i, j] = T
            forces_stokeslet[i, j] = F_S
            torques_rotlet[i, j] = T_R
    plt.plot(forces.flatten().real - forces_stokeslet.flatten().real)
    plt.plot(torques.flatten() - torques_rotlet.flatten())
    plt.legend(["F", "F_S", "T", "T_R"])
    plt.show()
