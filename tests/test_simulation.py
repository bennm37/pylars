"""Test the QuasiSteadySimulation class."""


def test_import_simulation():
    """Test that the QuasiSteadySimulation class can be imported."""
    from pylars.simulation import Simulation

    assert Simulation is not None


def test_mover_simulation():
    from pylars import Problem, Solution
    from pylars.simulation import LowDensityMoverSimulation, Mover
    import numpy as np
    import matplotlib.pyplot as plt

    init_prob = Problem()
    corners = [-1 - 1j, 1 - 1j, 1 + 1j, -1 + 1j]
    init_prob.add_exterior_polygon(
        corners,
        num_edge_points=600,
        num_poles=0,
        deg_poly=20,
        spacing="linear",
    )
    init_prob.add_point(-1.0 - 1.0j)
    init_prob.add_boundary_condition("0", "u[0]", 0)
    init_prob.add_boundary_condition("0", "v[0]", 0)
    init_prob.add_boundary_condition("2", "u[2]", 0)
    init_prob.add_boundary_condition("2", "v[2]", 0)
    init_prob.add_boundary_condition("1", "u[1]-u[3][::-1]", 0)
    init_prob.add_boundary_condition("1", "v[1]-v[3][::-1]", 0)
    init_prob.add_boundary_condition("3", "p[1]-p[3][::-1]", -1)
    init_prob.add_boundary_condition("3", "e12[1]-e12[3][::-1]", 0)
    init_prob.add_boundary_condition("4", "p[4]", 0)
    init_prob.add_boundary_condition("4", "psi[4]", 0)

    centroid = 0.0 + 0.01j
    angle = 0.0
    velocity = 0.0 + 0.0j
    angular_velocity = 0.0
    R = 0.1
    curve = lambda t: centroid + R * np.exp(2j * np.pi * t)
    deriv = lambda t: R * 2j * np.pi * np.exp(2j * np.pi * t)
    cell = Mover(
        curve=curve,
        deriv=deriv,
        centroid=centroid,
        angle=angle,
        velocity=velocity,
        angular_velocity=angular_velocity,
    )
    movers = [cell]
    ldms = LowDensityMoverSimulation(init_prob, movers)
    results = ldms.run(0, 0.4, 0.1)
    solutions = results["solution_data"]
    mover_data = results["mover_data"]
    for sol in solutions:
        assert isinstance(sol, Solution)
    position_data = mover_data["positions"]
    angle_data = mover_data["angles"]
    velocity_data = mover_data["velocities"]
    angular_velocity_data = mover_data["angular_velocities"]
    assert isinstance(position_data, np.ndarray)
    assert position_data.shape == (4, 1)
    assert position_data.dtype == np.complex128
    assert isinstance(angle_data, np.ndarray)
    assert angle_data.shape == (4, 1)
    assert angle_data.dtype == np.float64
    assert isinstance(velocity_data, np.ndarray)
    assert velocity_data.shape == (4, 1)
    assert velocity_data.dtype == np.complex128
    assert isinstance(angular_velocity_data, np.ndarray)
    assert angular_velocity_data.shape == (4, 1)
    assert angular_velocity_data.dtype == np.float64


if __name__ == "__main__":
    test_import_simulation()
    test_mover_simulation()
