"""Test generating curved domains."""


def test_circle_in_circle():
    from pylars import Problem, CurvedDomain
    import numpy as np
    import matplotlib.pyplot as plt

    prob = Problem()
    outer_r = 1
    outer_circle = lambda t: outer_r * np.exp(2j * np.pi * t)
    prob.add_curved_domain(outer_circle, num_edge_points=500)
    center = 0.5
    inner_r = 0.2
    inner_circle = lambda t: center + inner_r * np.exp(2j * np.pi * t)
    prob.add_interior_curve(inner_circle, num_points=300)
    prob.domain.plot(set_lims=False)
    assert isinstance(prob.domain, CurvedDomain)
    assert np.allclose(
        prob.domain.exterior_points,
        outer_circle(np.linspace(0, 1, 500).reshape(-1, 1)),
    )
    assert np.allclose(
        prob.domain.boundary_points[prob.domain.indices["1"]],
        inner_circle(np.linspace(0, 1, 300)).reshape(-1, 1),
    )


def test_solve_circle_in_circle():
    from pylars import Problem, Solver, Analysis
    from pylars.simulation import Mover
    import numpy as np
    import matplotlib.pyplot as plt

    prob = Problem()
    outer_r = 1
    outer_circle = lambda t: outer_r * np.exp(2j * np.pi * t)
    prob.add_curved_domain(outer_circle, num_edge_points=500)
    inner_center = 0.5
    inner_r = 0.2
    inner_circle = lambda t: inner_center + inner_r * np.exp(2j * np.pi * t)
    inner_deriv = lambda t: 2j * np.pi * inner_r * np.exp(2j * np.pi * t)
    mover = Mover(
        inner_circle,
        inner_deriv,
        centroid=inner_center,
        velocity=1.0,
        angular_velocity=1,
    )
    prob.add_mover(mover, num_points=200)
    prob.add_boundary_condition("0", "u[0]", 0)
    prob.add_boundary_condition("0", "v[0]", 0)
    solver = Solver(prob)
    sol = solver.solve()
    analysis = Analysis(sol)
    analysis.plot(resolution=301)
    # TODO compare to williams results here


if __name__ == "__main__":
    test_circle_in_circle()
    test_solve_circle_in_circle()
