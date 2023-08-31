"""Flow a domain with a circular interior curve.""" ""
from pylars import Problem, Solver, Analysis
from pylars.simulation import Mover
import matplotlib.pyplot as plt
import numpy as np


def get_roating_circle_torque(
    v_x, v_y, v_theta, centroid=0.0 + 0.1j, radius=0.1
):
    """Flow a domain with a circular interior curve."""
    prob = Problem()
    prob.add_periodic_domain(
        length=2,
        height=2,
        num_edge_points=600,
        num_poles=0,
        deg_poly=75,
        spacing="linear",
    )

    num_points = 100
    angle = 0.0
    circle = lambda t: centroid + radius * np.exp(2j * np.pi * t)  # noqa: E731
    circle_deriv = lambda t: 1j * np.pi * np.exp(2j * np.pi * t)  # noqa: E731
    mover = Mover(
        circle,
        circle_deriv,
        centroid,
        angle,
        velocity=v_x + v_y,
        angular_velocity=v_theta,
    )
    prob.add_boundary_condition("0", "u[0]", 0)
    prob.add_boundary_condition("0", "v[0]", 0)
    prob.add_boundary_condition("2", "u[2]", 0)
    prob.add_boundary_condition("2", "v[2]", 0)
    prob.add_boundary_condition("1", "u[1]-u[3][::-1]", 0)
    prob.add_boundary_condition("1", "v[1]-v[3][::-1]", 0)
    prob.add_boundary_condition("3", "p[1]-p[3][::-1]", 2)
    prob.add_boundary_condition("3", "e12[1]-e12[3][::-1]", 0)
    prob.add_mover(mover, num_points=num_points, mirror_laurents=True)

    solver = Solver(prob)
    sol = solver.solve(check=False, normalize=False)
    print(f"Error: {solver.max_error}")
    force = sol.force(circle, circle_deriv)
    torque = sol.torque(circle, circle_deriv, centroid)
    return force, torque, sol


def plot_torques():
    """Plot torques for different angular velocities."""
    v_x, v_y = 0.0, 0.0
    n_samples = 10
    v_thetas = np.linspace(-1, 1, n_samples)
    forces = np.zeros(n_samples, dtype=np.complex128)
    torques = np.zeros(n_samples, dtype=np.complex128)
    for i, v_theta in enumerate(v_thetas):
        force, torque = get_roating_circle_torque(v_x, v_y, v_theta)
        forces[i] = force
        torques[i] = torque
        print(f"force = {force}")
        print(f"torque = {torque}")
    fig, ax = plt.subplots()
    # ax.plot(v_thetas, forces, label="force")
    ax.plot(v_thetas, torques, label="torque")
    plt.show()


if __name__ == "__main__":
    v_x, v_y = 0.0952517530741671, -5.55653764294123e-09
    v_theta = -3.063423608102873
    # v_x, v_y = 0.0, 0.0
    # v_theta = -0.1
    # v_x, v_y = -0.3737360515871475, 0.0
    # v_theta = -0.7534464129151148
    centroid = 0.0 + 0.85j
    R = 0.1
    force, torque, sol = get_roating_circle_torque(
        v_x, v_y, v_theta, centroid, R
    )
    print(f"force = {force}")
    print(f"torque = {torque}")
    an = Analysis(sol)
    fig, ax = an.plot(
        resolution=200, interior_patch=True, quiver=True, n_streamlines=100
    )
    ax.quiver(
        centroid.real,
        centroid.imag,
        force.real,
        force.imag,
        color="r",
        zorder=4,
    )
    ax.quiver(
        centroid.real + R,
        centroid.imag,
        0,
        torque,
        color="r",
        zorder=4,
    )
    plt.show()
