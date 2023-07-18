"""Test the different stress calculation methods."""


def test_poiseuille_stress():
    """Test stress calculation using the solution class."""
    from pylars import Problem, Solver
    import numpy as np

    # create a square domain
    corners = [1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j]
    prob = Problem()
    prob.add_exterior_polygon(
        corners,
        num_edge_points=300,
        num_poles=0,
        deg_poly=24,
        spacing="linear",
    )
    prob.add_boundary_condition("0", "u[0]", 0)
    prob.add_boundary_condition("0", "v[0]", 0)
    prob.add_boundary_condition("2", "u[2]", 0)
    prob.add_boundary_condition("2", "v[2]", 0)
    # parabolic inlet
    prob.add_boundary_condition("1", "p[1]", 2)
    prob.add_boundary_condition("1", "v[1]", 0)
    # outlet
    prob.add_boundary_condition("3", "p[3]", -2)
    prob.add_boundary_condition("3", "v[3]", 0)
    solver = Solver(prob)
    sol = solver.solve(weight=False, normalize=False)

    psi_answer = lambda z: z.imag * (1 - z.imag**2 / 3)  # noqa E731
    uv_answer = lambda z: 1 - z.imag**2  # noqa E731
    p_answer = lambda z: -2 * z.real  # noqa E731
    x = np.linspace(-1, 1, 100)
    X, Y = np.meshgrid(x, x)
    Z = X + 1j * Y
    ATOL, RTOL = 1e-10, 1e-3
    assert np.allclose(
        sol.uv(Z).reshape(100, 100), uv_answer(Z), atol=ATOL, rtol=RTOL
    )
    assert np.allclose(
        sol.p(Z).reshape(100, 100), p_answer(Z), atol=ATOL, rtol=RTOL
    )

    def poiseuille_stress(z):
        zshape = z.shape
        z = z.reshape(-1)
        isotropic = np.array([-np.eye(2) * (-2 * x) for x in z.real])
        tr = np.array([[0, 1], [0, 0]])
        bl = np.array([[0, 0], [1, 0]])
        deviatoric = np.array([tr * (-2 * y) + bl * (-2 * y) for y in z.imag])
        stress = isotropic + deviatoric
        return stress.reshape(zshape + (2, 2))

    stress_discrete = sol.stress_discrete(Z)
    assert np.allclose(
        stress_discrete, poiseuille_stress(Z), atol=ATOL, rtol=RTOL
    )
    stress_goursat = sol.stress_goursat(Z)
    assert np.allclose(
        stress_goursat, poiseuille_stress(Z), atol=ATOL, rtol=RTOL
    )


def test_couette_stress():
    """Test stress calculation using the solution class."""
    from pylars import Problem, Solver
    import numpy as np

    # create a square domain
    corners = [1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j]
    prob = Problem()
    prob.add_exterior_polygon(
        corners, num_edge_points=300, num_poles=0, spacing="linear"
    )
    # periodic Couette flow with no-slip boundary conditions
    prob.add_boundary_condition("0", "u[0]", 1)
    prob.add_boundary_condition("0", "v[0]", 0)
    prob.add_boundary_condition("2", "u[2]", 0)
    prob.add_boundary_condition("2", "v[2]", 0)
    prob.add_boundary_condition("1", "p[1]", 0)
    prob.add_boundary_condition("1", "v[1]", 0)
    prob.add_boundary_condition("3", "p[3]", 0)
    prob.add_boundary_condition("3", "v[3]", 0)
    solver = Solver(prob)
    sol = solver.solve(check=False, weight=False)
    x = np.linspace(-1, 1, 100)
    y = x.copy()
    X, Y = np.meshgrid(x, x)
    Z = X + 1j * Y
    ATOL, RTOL = 1e-9, 1e-5
    assert np.allclose(sol.p(1j * y), 0, atol=ATOL, rtol=RTOL)
    assert np.allclose(
        sol.uv(1j * y).real.reshape(100), (y + 1) / 2, atol=ATOL, rtol=RTOL
    )
    assert np.allclose(sol.uv(1j * y).imag, 0, atol=ATOL, rtol=RTOL)

    def couette_stress(z):
        zshape = z.shape
        z = z.reshape(-1)
        # no pressure
        isotropic = np.array([np.eye(2) * 0 for x in z.real])
        # constant shear
        tr = np.array([[0, 1], [0, 0]])
        bl = np.array([[0, 0], [1, 0]])
        deviatoric = np.array([tr * (1 / 2) + bl * (1 / 2) for y in z.imag])
        stress = isotropic + deviatoric
        return stress.reshape(zshape + (2, 2))

    stress_discrete = sol.stress_discrete(Z)
    assert np.allclose(
        stress_discrete, couette_stress(Z), atol=ATOL, rtol=RTOL
    )
    stress_goursat = sol.stress_goursat(Z)
    assert np.allclose(stress_goursat, couette_stress(Z), atol=ATOL, rtol=RTOL)


def test_poiseuille_force():
    """Test force calculation using the solution class."""
    from pylars import Problem, Solver
    import numpy as np

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


def test_discrete_vs_goursat_circle_stress():
    """Flow a domain with a circular interior curve.""" ""
    from pylars import Problem, Solver
    import numpy as np

    prob = Problem()
    corners = [-1 - 1j, 1 - 1j, 1 + 1j, -1 + 1j]
    prob.add_exterior_polygon(
        corners,
        num_edge_points=600,
        num_poles=0,
        deg_poly=20,
        spacing="linear",
    )
    circle = lambda t: 0.5 * np.exp(2j * np.pi * t)  # noqa: E731
    prob.add_interior_curve(
        lambda t: 0.5 * np.exp(2j * np.pi * t),
        num_points=100,
        deg_laurent=20,
        centroid=0.0 + 0.0j,
    )
    prob.add_boundary_condition("0", "psi[0]", 1)
    prob.add_boundary_condition("0", "u[0]", 0)
    prob.add_boundary_condition("2", "psi[2]", 0)
    prob.add_boundary_condition("2", "u[2]", 0)
    prob.add_boundary_condition("1", "u[1]-u[3][::-1]", 0)
    prob.add_boundary_condition("1", "v[1]-v[3][::-1]", 0)
    prob.add_boundary_condition("4", "u[4]", 0)
    prob.add_boundary_condition("4", "v[4]", 0)
    solver = Solver(prob)
    sol = solver.solve(check=False, normalize=False)
    z = circle(np.linspace(0, 1, 200))
    stress_discrete = sol.stress_discrete(z, dx=1e-5)
    stress_goursat = sol.stress_goursat(z)
    ATOL, RTOL = 1e-9, 2
    assert np.allclose(stress_discrete, stress_goursat, atol=ATOL, rtol=RTOL)


def test_goursat_force():
    """Test force calculation using the solution class."""
    pass


if __name__ == "__main__":
    # test_poiseuille_stress()
    # test_couette_stress()
    # test_poiseuille_force()
    test_discrete_vs_goursat_circle_stress()
