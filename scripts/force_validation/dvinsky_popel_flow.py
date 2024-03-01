"""Flow a domain with a circular interior curve.""" ""
from pylars import Problem, Solver, Analysis
from pylars.simulation import Mover
import matplotlib.pyplot as plt
import numpy as np


def run(
    velocity,
    angular_velocity,
    inlet_profile,
    top_velocity=0,
    filename=None,
    resolution=600,
    colorbar=False,
):
    prob = Problem()
    corners = [2 + 1j, -2 + 1j, -2, 2]
    prob.add_exterior_polygon(
        corners=corners,
        num_edge_points=600,
        num_poles=0,
        deg_poly=75,
        spacing="linear",
    )

    centroid = -0.0 + 0.6j
    radius = 0.1
    circle = lambda t: centroid + radius * np.exp(2j * np.pi * t)  # noqa: E731
    circle_deriv = lambda t: 1j * np.pi * np.exp(2j * np.pi * t)  # noqa: E731
    num_points = 100
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
        deg_laurent=20,
    )
    prob.domain.plot(set_lims=False)
    plt.show()
    prob.add_boundary_condition("0", "u[0]", top_velocity)
    prob.add_boundary_condition("0", "v[0]", 0)
    prob.add_boundary_condition("2", "u[2]", 0)
    prob.add_boundary_condition("2", "v[2]", 0)
    prob.add_boundary_condition("1", "u[1]", inlet_profile)
    prob.add_boundary_condition("1", "v[1]", 0)
    prob.add_boundary_condition("3", "p[3]", 0)
    prob.add_boundary_condition("3", "v[3]", 0)
    solver = Solver(prob)
    sol = solver.solve(normalize=False)
    an = Analysis(sol)
    print(f"residusal = {np.abs(solver.A @ solver.coefficients - solver.b).max()}")
    fig, ax = an.plot(
        resolution=resolution,
        interior_patch=True,
        quiver=False,
        streamline_type="linear",
        colorbar=colorbar,
        enlarge_patch=1.1,
        n_streamlines=20,
        imshow=True,
    )
    plt.tight_layout()
    if filename is not None:
        an.save_pgf(filename)
    plt.show()


if __name__ == "__main__":
    cases = [
        {
            "filename": "dvinsky_popel_flow_poiseuille",
            "velocity": 0.0,
            "angular_velocity": 20,
            "inlet_profile": "6*y*(1-y)",
            "resolution": 200,
        },
        {
            "filename": "dvinsky_popel_flow_couette",
            "velocity": 0.0,
            "angular_velocity": 100,
            "top_velocity": 1,
            "inlet_profile": "y",
            "resolution": 200,
        },
    ]
    for case in cases:
        run(**case)
