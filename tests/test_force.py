"""Test the force and torque methods of Solution."""


def test_poiseuille_force():
    """Test force calculation using the solution class."""
    from pylars import Problem, Solver
    import numpy as np

    # TODO check signs
    # create a square domain
    corners = [2 + 1j, -2 + 1j, -2 - 1j, 2 - 1j]
    prob = Problem()
    prob.add_exterior_polygon(
        corners,
        num_edge_points=300,
        num_poles=0,
        deg_poly=24,
        spacing="linear",
    )
    p_0 = 5
    prob.add_boundary_condition("0", "u[0]", 0)
    prob.add_boundary_condition("0", "v[0]", 0)
    prob.add_boundary_condition("2", "u[2]", 0)
    prob.add_boundary_condition("2", "v[2]", 0)
    # parabolic inlet
    prob.add_boundary_condition("1", "p[1]", p_0 + 4)
    prob.add_boundary_condition("1", "v[1]", 0)
    # outlet
    prob.add_boundary_condition("3", "p[3]", p_0 - 4)
    prob.add_boundary_condition("3", "v[3]", 0)
    solver = Solver(prob)
    sol = solver.solve(weight=False, normalize=False)

    psi_answer = lambda z: z.imag * (1 - z.imag**2 / 3)  # noqa E731
    uv_answer = lambda z: 1 - z.imag**2  # noqa E731
    p_answer = lambda z: p_0 - 2 * z.real  # noqa E731
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-1, 1, 100)
    X, Y = np.meshgrid(x, y, indexing="ij")
    Z = X + 1j * Y
    ATOL, RTOL = 1e-10, 1e-3
    assert np.allclose(
        sol.uv(Z).reshape(100, 100), uv_answer(Z), atol=ATOL, rtol=RTOL
    )
    assert np.allclose(
        sol.p(Z).reshape(100, 100), p_answer(Z), atol=ATOL, rtol=RTOL
    )
    poiseuille_force = np.array([8 + 4 * p_0 * 1j])
    curve = lambda t: 4 * t - 2 + 1j  # noqa E731
    deriv = lambda t: 4  # noqa E731
    force = sol.force(curve, deriv)
    assert np.allclose(force, poiseuille_force, atol=ATOL, rtol=RTOL)
    centorids = np.arange(0, 0.5, 0.1) * 1j
    for centroid in centorids:
        circle = lambda t: centroid + 0.5 * np.exp(2j * np.pi * t)  # noqa E731
        circle_deriv = (
            lambda t: 1j * np.pi * np.exp(2j * np.pi * t)
        )  # noqa E731
        torque = sol.torque(circle, circle_deriv, centroid=centroid)
        assert np.isclose(torque, -centroid.imag * np.pi, atol=ATOL, rtol=RTOL)


def test_goursat_force():
    """Test force calculation using the solution class."""
    pass


if __name__ == "__main__":
    test_poiseuille_force()
    test_goursat_force()
