"""Test the QuasiSteadySimulation class."""


def test_import_simulation():
    """Test that the QuasiSteadySimulation class can be imported."""
    from pylars import QuasiSteadySimulation

    assert QuasiSteadySimulation is not None
    qss = QuasiSteadySimulation()
    assert qss is not None


def test_simulation():
    """Test the QuasiSteadySimulation class."""
    from pylars import Problem, Solution, Analysis
    from pylars.simulation import QuasiSteadySimulation, Mover
    import numpy as np
    import matplotlib.pyplot as plt

    init_prob = Problem()
    corners = [-1 - 1j, 1 - 1j, 1 + 1j, -1 + 1j]
    init_prob.add_exterior_polygon(
        corners,
        num_edge_points=50,
        num_poles=0,
        deg_poly=20,
        spacing="linear",
    )
    init_prob.add_boundary_condition("0", "psi[0]", 1)
    init_prob.add_boundary_condition("0", "u[0]", 0)
    init_prob.add_boundary_condition("2", "psi[2]", 0)
    init_prob.add_boundary_condition("2", "u[2]", 0)
    init_prob.add_boundary_condition("1", "u[1]-u[3][::-1]", 0)
    init_prob.add_boundary_condition("1", "v[1]-v[3][::-1]", 0)
    init_prob.add_boundary_condition("4", "u[4]", 0)
    init_prob.add_boundary_condition("4", "v[4]", 0)

    centroid = 0.0 + 0.0j
    angle = 0.0
    velocity = 0.0 + 0.0j
    angular_velocity = 0.0
    curve = lambda t: 0.5 * np.exp(2j * np.pi * t)
    deriv = lambda t: 1j * np.pi * np.exp(2j * np.pi * t)
    cell = Mover(
        curve=curve,
        deriv=deriv,
        centroid=centroid,
        angle=angle,
        velocity=velocity,
        angular_velocity=angular_velocity,
    )

    def update(prob, sol, movers, t, dt):
        """Take in the base problem, preivous solution, and mover and return a new problem."""  # noqa E501
        new_prob = prob.copy()
        if sol is None:
            for mover in movers:
                new_prob.add_mover(mover)
            return new_prob
        else:
            for mover in movers:
                force = sol.force(mover.curve, mover.deriv)
                torque = sol.torque(mover.curve, mover.deriv, mover.centroid)
                acceleration = -force / mover.mass
                angular_acceleration = torque / mover.moi
                mover.velocity += acceleration * dt
                mover.angular_velocity += angular_acceleration * dt
                mover.translate(mover.velocity * dt)
                mover.rotate(mover.angular_velocity * dt)
                new_prob.add_mover(mover)
            return new_prob

    qss = QuasiSteadySimulation(init_prob)
    qss.add_mover(cell)
    qss.add_update_rule(update)
    solutions, mover_data = qss.run(start=0, end=1, dt=0.1)
    assert solutions is not None
    for result in solutions:
        assert isinstance(result, Solution)
    an = Analysis(solutions, type="simulation")
    fig, ax, anim = an.animate()
    plt.show()


def test_mover_simulation():
    from pylars import Problem, Analysis
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
    # init_prob.add_boundary_condition("0", "u[0]-u[2][::-1]", 0)
    # init_prob.add_boundary_condition("0", "v[0]-v[2][::-1]", 0)
    # init_prob.add_boundary_condition("2", "p[0]-p[2][::-1]", 0)
    # init_prob.add_boundary_condition("2", "e12[0]-e12[2][::-1]", 0)

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


    centroid = 0.0 + 0.0j
    angle = 0.0
    velocity = 0.0 + 0.0j
    angular_velocity = 0.0
    R = 0.05
    curve = lambda t: R * np.exp(2j * np.pi * t)
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
    results = ldms.run(0, 0.5, 0.1)
    solutions = results["solution_data"]
    for solution in solutions:
        an = Analysis(solution)
        fig, ax = an.plot(interior_patch=True)
        plt.show()
    print(results)
    plt.plot(results["mover_data"]["positions"].real)
    plt.plot(results["mover_data"]["velocities"].real)
    plt.plot(results["mover_data"]["angles"])
    plt.plot(results["mover_data"]["angular_velocities"])
    plt.show()


if __name__ == "__main__":
    # test_import_simulation()
    # test_simulation()
    test_mover_simulation()
