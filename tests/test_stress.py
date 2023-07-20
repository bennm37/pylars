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


def test_comsol_cylinder_stress():
    """Compare stress data to COMSOL."""
    import pandas as pd
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
    prob.add_interior_curve(
        lambda t: 0.5 * np.exp(2j * np.pi * t),
        num_points=100,
        deg_laurent=20,
        centroid=0.0 + 0.0j,
    )
    prob.add_boundary_condition("0", "u[0]", 0)
    prob.add_boundary_condition("0", "v[0]", 0)
    prob.add_boundary_condition("2", "u[2]", 0)
    prob.add_boundary_condition("2", "v[2]", 0)
    prob.add_boundary_condition("3", "p[3]", 1)
    prob.add_boundary_condition("3", "v[3]", 0)
    prob.add_boundary_condition("1", "p[1]", -1)
    prob.add_boundary_condition("1", "v[1]", 0)
    prob.add_boundary_condition("4", "u[4]", 0)
    prob.add_boundary_condition("4", "v[4]", 0)
    solver = Solver(prob)
    sol = solver.solve(check=False, normalize=False)
    # calculate normal stress data
    stress_x_df = pd.read_csv("tests/data/COMSOL_cylinder_stress_x.csv")
    theta = stress_x_df["theta"]
    circle = lambda t: 0.5 * np.exp(t * 1j)  # noqa E731
    normal = lambda t: -np.exp(t * 1j)  # noqa E731
    z = circle(np.array(theta))
    stress = sol.stress_goursat(z)
    normals = normal(np.array(theta))
    normals = np.array([normals.real, normals.imag]).T
    # normal_stress = np.einsum('ij,ljk->ij', normals, stress)
    normal_stress = np.array([S @ n for n, S in zip(normals, stress)])

    # normal stress data from COMSOL
    stress_x_df = pd.read_csv("tests/data/COMSOL_cylinder_stress_x.csv")
    stress_y_df = pd.read_csv("tests/data/COMSOL_cylinder_stress_y.csv")
    theta = stress_x_df["theta"]
    stress_x = stress_x_df["stress_x"]
    stress_y = stress_y_df["stress_y"]
    # qualitative agreement
    assert np.allclose(normal_stress[:, 0], stress_x, atol=2e-2, rtol=1e-3)
    assert np.allclose(normal_stress[:, 1], stress_y, atol=2e-2, rtol=1e-3)


if __name__ == "__main__":
    test_poiseuille_stress()
    test_couette_stress()
    test_discrete_vs_goursat_circle_stress()
    test_comsol_cylinder_stress()
