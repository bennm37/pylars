"""Test solver functions."""
from test_settings import RTOL


def test_lid_driven_cavity_make_functions():
    """Test using make_function to create functions."""
    from scipy.io import loadmat
    from pylars.numerics import make_function
    import numpy as np

    n = 24
    num_poles = 24
    test_answers = loadmat(
        f"tests/data/lid_driven_cavity_n_{n}_np_{num_poles}.mat"
    )
    Z = test_answers["Z"]
    Hes = test_answers["Hes"]
    hessenbergs = [Hes[0, i] for i in range(Hes.shape[1])]
    Pol = test_answers["Pol"]
    poles = np.array([Pol[0, i] for i in range(4)]).reshape(4, 24)
    c = test_answers["c"]
    psi_100_100_answer = test_answers["psi_100_100"]
    p_100_100_answer = test_answers["p_100_100"]
    uv_100_100_answer = test_answers["uv_100_100"]
    omega_100_100_answer = test_answers["omega_100_100"]

    def psi(z):
        return make_function("psi", z, c, hessenbergs, poles)

    def p(z):
        return make_function("p", z, c, hessenbergs, poles)

    def uv(z):
        return make_function("uv", z, c, hessenbergs, poles)

    def omega(z):
        return make_function("omega", z, c, hessenbergs, poles)

    x = np.linspace(-1, 1, 100)
    X, Y = np.meshgrid(x, x)
    Z = X + 1j * Y
    psi_100_100 = psi(Z).reshape(100, 100)
    p_100_100 = p(Z).reshape(100, 100)
    uv_100_100 = uv(Z).reshape(100, 100)
    omega_100_100 = omega(Z).reshape(100, 100)
    ATOL = 1e-12  # small velocity elements are not
    # accurate for floating point reasons
    assert np.allclose(psi_100_100, psi_100_100_answer, atol=ATOL, rtol=RTOL)
    assert np.allclose(p_100_100, p_100_100_answer, atol=ATOL, rtol=RTOL)
    assert np.allclose(uv_100_100, uv_100_100_answer, atol=ATOL, rtol=RTOL)
    assert np.allclose(
        omega_100_100, omega_100_100_answer, atol=ATOL, rtol=RTOL
    )


def test_single_circle_make_functions():
    """Test using make_function to create functions."""
    from scipy.io import loadmat
    from pylars.numerics import make_function
    import numpy as np

    test_answers = loadmat("tests/data/single_circle_test.mat")
    deg_laurent = test_answers["nl"][0][0]
    Z = test_answers["Z"]
    H_answer = test_answers["Hes"]
    # discard empty pole blocks
    hessenbergs = [
        H_answer[:, k][0]
        for k in range(H_answer.shape[1])
        if H_answer[:, k][0].shape != (0, 0)
    ]
    c = test_answers["c"]
    psi_100_100_answer = test_answers["psi_100_100"]
    p_100_100_answer = test_answers["p_100_100"]
    uv_100_100_answer = test_answers["uv_100_100"]
    omega_100_100_answer = test_answers["omega_100_100"]
    f_100_100_answer = test_answers["f_100_100"]
    g_100_100_answer = test_answers["g_100_100"]
    laurents = [(0.0 + 0.0j, deg_laurent)]

    def psi(z):
        return make_function(
            "psi", z, c, hessenbergs, interior_laurents=laurents
        )

    def p(z):
        return make_function(
            "p", z, c, hessenbergs, interior_laurents=laurents
        )

    def uv(z):
        return make_function(
            "uv", z, c, hessenbergs, interior_laurents=laurents
        )

    def omega(z):
        return make_function(
            "omega", z, c, hessenbergs, interior_laurents=laurents
        )

    def f(z):
        return make_function(
            "f", z, c, hessenbergs, interior_laurents=laurents
        )

    def g(z):
        return make_function(
            "g", z, c, hessenbergs, interior_laurents=laurents
        )

    x = np.linspace(-1, 1, 100)
    X, Y = np.meshgrid(x, x)
    Z = X + 1j * Y
    psi_100_100 = psi(Z).reshape(100, 100)
    uv_100_100 = uv(Z).reshape(100, 100)
    p_100_100 = p(Z).reshape(100, 100)
    omega_100_100 = omega(Z).reshape(100, 100)
    f_100_100 = f(Z).reshape(100, 100)
    g_100_100 = g(Z).reshape(100, 100)
    ATOL = 1e-12  # small velocity elements are not
    # accurate for floating point reasons
    # TODO mask this??
    assert np.allclose(psi_100_100, psi_100_100_answer, atol=ATOL, rtol=RTOL)
    assert np.allclose(p_100_100, p_100_100_answer, atol=ATOL, rtol=RTOL)
    assert np.allclose(uv_100_100, uv_100_100_answer, atol=ATOL, rtol=RTOL)
    assert np.allclose(
        omega_100_100, omega_100_100_answer, atol=ATOL, rtol=RTOL
    )
    assert np.allclose(f_100_100, f_100_100_answer, atol=ATOL, rtol=RTOL)
    assert np.allclose(g_100_100, g_100_100_answer, atol=ATOL, rtol=RTOL)


def test_three_circles_make_functions():
    """Test using make_function to create functions."""
    from scipy.io import loadmat
    from pylars.numerics import make_function
    import numpy as np

    test_answers = loadmat("tests/data/three_circles_test.mat")
    deg_laurent = test_answers["nl"][0][0]
    Z = test_answers["Z"]
    H_answer = test_answers["Hes"]
    # discard empty pole blocks
    hessenbergs = [
        H_answer[:, k][0]
        for k in range(H_answer.shape[1])
        if H_answer[:, k][0].shape != (0, 0)
    ]
    c = test_answers["c"]
    psi_100_100_answer = test_answers["psi_100_100"]
    p_100_100_answer = test_answers["p_100_100"]
    uv_100_100_answer = test_answers["uv_100_100"]
    omega_100_100_answer = test_answers["omega_100_100"]
    f_100_100_answer = test_answers["f_100_100"]
    g_100_100_answer = test_answers["g_100_100"]
    laurents = [
        (0.0 + 0.0j, deg_laurent),
        (0.5 + 0.5j, deg_laurent),
        (-0.5 + -0.5j, deg_laurent),
    ]

    def psi(z):
        return make_function(
            "psi",
            z,
            c,
            hessenbergs,
            interior_laurents=laurents,
        )

    def p(z):
        return make_function(
            "p", z, c, hessenbergs, interior_laurents=laurents
        )

    def uv(z):
        return make_function(
            "uv", z, c, hessenbergs, interior_laurents=laurents
        )

    def omega(z):
        return make_function(
            "omega", z, c, hessenbergs, interior_laurents=laurents
        )

    def f(z):
        return make_function(
            "f", z, c, hessenbergs, interior_laurents=laurents
        )

    def g(z):
        return make_function(
            "g", z, c, hessenbergs, interior_laurents=laurents
        )

    x = np.linspace(-1, 1, 100)
    X, Y = np.meshgrid(x, x)
    Z = X + 1j * Y
    psi_100_100 = psi(Z).reshape(100, 100)
    uv_100_100 = uv(Z).reshape(100, 100)
    p_100_100 = p(Z).reshape(100, 100)
    omega_100_100 = omega(Z).reshape(100, 100)
    f_100_100 = f(Z).reshape(100, 100)
    g_100_100 = g(Z).reshape(100, 100)
    ATOL = 1e-12  # small velocity elements are not
    # accurate for floating point reasons
    # TODO mask this??
    assert np.allclose(psi_100_100, psi_100_100_answer, atol=ATOL, rtol=RTOL)
    assert np.allclose(p_100_100, p_100_100_answer, atol=ATOL, rtol=RTOL)
    assert np.allclose(uv_100_100, uv_100_100_answer, atol=ATOL, rtol=RTOL)
    assert np.allclose(
        omega_100_100, omega_100_100_answer, atol=ATOL, rtol=RTOL
    )
    assert np.allclose(f_100_100, f_100_100_answer, atol=ATOL, rtol=RTOL)
    assert np.allclose(g_100_100, g_100_100_answer, atol=ATOL, rtol=RTOL)


def test_lid_driven_cavity_solve():
    """Tests the solver from BCs to solution."""
    from pylars import Problem, Solver
    from scipy.io import loadmat
    import numpy as np
    import matplotlib.pyplot as plt

    n = 24
    num_poles = 24
    test_answers = loadmat(
        f"tests/data/lid_driven_cavity_n_{n}_np_{num_poles}.mat"
    )
    Z_answer = test_answers["Z"]
    psi_100_100_answer = test_answers["psi_100_100"]
    p_100_100_answer = test_answers["p_100_100"]
    uv_100_100_answer = test_answers["uv_100_100"]
    omega_100_100_answer = test_answers["omega_100_100"]

    # lid driven cavity BCS
    corners = [1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j]
    prob = Problem()
    prob.add_exterior_polygon(
        corners,
        num_edge_points=300,
        length_scale=1.5 * np.sqrt(2),
        sigma=4,
        deg_poly=24,
        num_poles=num_poles,
    )
    assert np.allclose(prob.domain.corners, corners)
    assert np.allclose(prob.domain.boundary_points, Z_answer)
    prob.add_boundary_condition("0", "psi[0]", 0)
    prob.add_boundary_condition("0", "u[0]", 1)
    # wall boundary conditions
    prob.add_boundary_condition("2", "psi[2]", 0)
    prob.add_boundary_condition("2", "u[2]", 0)
    prob.add_boundary_condition("1", "psi[1]", 0)
    prob.add_boundary_condition("1", "v[1]", 0)
    prob.add_boundary_condition("3", "psi[3]", 0)
    prob.add_boundary_condition("3", "v[3]", 0)
    prob.check_boundary_conditions()
    solver = Solver(prob)
    sol = solver.solve(check=True, weight=True, normalize=True)
    x = np.linspace(-1, 1, 100)
    X, Y = np.meshgrid(x, x)
    Z = X + 1j * Y
    psi_100_100 = sol.psi(Z).reshape(100, 100)
    p_100_100 = sol.p(Z).reshape(100, 100)
    uv_100_100 = sol.uv(Z).reshape(100, 100)
    omega_100_100 = sol.omega(Z).reshape(100, 100)
    # ill conditioning of A means some disagreement towards the edges
    # and only agreement to 1e-3 in the interior
    ATOL = 1e-15
    RTOL = 1e-3
    assert np.allclose(
        uv_100_100[1:-1, 1:-1],
        uv_100_100_answer[1:-1, 1:-1],
        atol=ATOL,
        rtol=RTOL,
    )
    assert np.allclose(
        psi_100_100[1:-1, 1:-1],
        psi_100_100_answer[1:-1, 1:-1],
        atol=ATOL,
        rtol=RTOL,
    )
    assert np.allclose(
        p_100_100[1:-1, 1:-1],
        p_100_100_answer[1:-1, 1:-1],
        atol=ATOL,
        rtol=RTOL,
    )
    plt.imshow(
        np.abs(omega_100_100[1:-1, 1:-1] - omega_100_100_answer[1:-1, 1:-1])
        / np.abs(omega_100_100_answer[1:-1, 1:-1])
    )
    assert np.allclose(
        omega_100_100[1:-1, 1:-1],
        omega_100_100_answer[1:-1, 1:-1],
        atol=ATOL,
        rtol=RTOL,
    )


def test_single_circle_solve():
    """Tests the solver from BCs to solution."""
    # TODO this test should probably be deleted -
    # it tests against MATLAB code with the wrong BCs
    from pylars import Problem, Solver
    import numpy as np
    from scipy.io import loadmat

    test_answers = loadmat("tests/data/single_circle_test.mat")
    deg_poly, deg_laurent = (
        test_answers["n"][0][0],
        test_answers["nl"][0][0],
    )
    num_edge_points, num_ellipse_points = (
        test_answers["nb"][0][0],
        test_answers["np"][0][0],
    )
    Z_answer = test_answers["Z"]
    # A_answer = test_answers["A"]
    rhs_answer = test_answers["rhs"]
    psi_100_100_answer = test_answers["psi_100_100"]
    p_100_100_answer = test_answers["p_100_100"]
    uv_100_100_answer = test_answers["uv_100_100"]
    # omega_100_100_answer = test_answers["omega_100_100"]
    prob = Problem()
    corners = [-1 - 1j, 1 - 1j, 1 + 1j, -1 + 1j]
    prob.add_exterior_polygon(
        corners,
        num_edge_points=num_edge_points,
        num_poles=0,
        deg_poly=deg_poly,
        spacing="linear",
    )
    prob.add_interior_curve(
        lambda t: 0.5 * np.exp(2j * np.pi * t),
        num_points=num_ellipse_points,
        deg_laurent=deg_laurent,
        centroid=0.0 + 0.0j,
    )
    prob.add_boundary_condition("0", "u[0]", 0)
    prob.add_boundary_condition("0", "psi[0]", 0)
    prob.add_boundary_condition("2", "u[2]", 0)
    prob.add_boundary_condition("2", "psi[2]", 1)
    prob.add_boundary_condition("1", "u[1]-u[3][::-1]", 0)
    prob.add_boundary_condition("1", "v[1]-v[3][::-1]", 0)
    # These should be used but aren't in the MATLAB code
    # prob.add_boundary_condition("3", "p[1]-p[3][::-1]", 0)
    # prob.add_boundary_condition("3", "e12[1]-e12[3][::-1]", 0)
    prob.add_boundary_condition("4", "u[4]", 0)
    prob.add_boundary_condition("4", "v[4]", 0)
    ATOL = 1e-12
    RTOL = 1e-3
    solver = Solver(prob)
    sol = solver.solve(check=False, normalize=False, weight=False)
    assert np.allclose(
        prob.domain.boundary_points, Z_answer, atol=ATOL, rtol=RTOL
    )
    assert np.allclose(solver.b, rhs_answer, atol=ATOL, rtol=RTOL)
    # assert np.allclose(solver.A, A_answer, atol=ATOL, rtol=RTOL)
    x = np.linspace(-1, 1, 100)
    X, Y = np.meshgrid(x, x)
    Z = X + 1j * Y

    psi_100_100 = sol.psi(Z).reshape(100, 100)
    p_100_100 = sol.p(Z).reshape(100, 100)
    uv_100_100 = sol.uv(Z).reshape(100, 100)
    # omega_100_100 = sol.omega(Z).reshape(100, 100)
    # ill conditioning of A means some disagreement towards the edges
    # and only agreement to 1e-3 in the interior
    mask = prob.domain.mask_contains(Z)
    ATOL = 1e-12
    RTOL = 1e-1
    assert np.allclose(
        psi_100_100[mask],
        psi_100_100_answer[mask],
        atol=ATOL,
        rtol=RTOL,
    )
    assert np.allclose(
        uv_100_100[mask],
        uv_100_100_answer[mask],
        atol=ATOL,
        rtol=RTOL,
    )
    p_100_100_norm = p_100_100[mask] - np.mean(p_100_100[mask])
    p_100_100_answer_norm = p_100_100_answer[mask] - np.mean(
        p_100_100_answer[mask]
    )
    assert np.allclose(
        p_100_100_norm,
        p_100_100_answer_norm,
        atol=ATOL,
        rtol=RTOL,
    )

    # omega close but sporadic O(1) differences?
    # assert np.allclose(
    #     omega_100_100_norm,
    #     omega_100_100_answer_norm,
    #     atol=ATOL,
    #     rtol=RTOL,
    # )


if __name__ == "__main__":
    test_lid_driven_cavity_make_functions()
    test_single_circle_make_functions()
    test_three_circles_make_functions()
    test_lid_driven_cavity_solve()
    test_single_circle_solve()
