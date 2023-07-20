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
    from pylars.simulation import QuasiSteadySimulation
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
    init_prob.add_interior_curve(
        lambda t: 0.5 * np.exp(2j * np.pi * t),
        num_points=100,
        deg_laurent=20,
        centroid=0.0 + 0.0j,
    )
    init_prob.add_boundary_condition("0", "psi[0]", 1)
    init_prob.add_boundary_condition("0", "u[0]", 0)
    init_prob.add_boundary_condition("2", "psi[2]", 0)
    init_prob.add_boundary_condition("2", "u[2]", 0)
    init_prob.add_boundary_condition("1", "u[1]-u[3][::-1]", 0)
    init_prob.add_boundary_condition("1", "v[1]-v[3][::-1]", 0)
    init_prob.add_boundary_condition("4", "u[4]", 0)
    init_prob.add_boundary_condition("4", "v[4]", 0)

    def update(prob, sol, t, dt):
        return prob

    qss = QuasiSteadySimulation(init_prob)
    qss.add_update_rule(update)
    results = qss.run(start=0, end=1, dt=0.1)
    assert results is not None
    for result in results:
        assert isinstance(result, Solution)
    an = Analysis(results, type="simulation")
    fig, ax, anim = an.animate()
    plt.show()
