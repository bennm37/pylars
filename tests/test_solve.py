"""Test solver functions."""
ATOL, RTOL = 0, 1e-5


def test_create_functions():
    from scipy.io import loadmat
    from pyls.numerics import make_function
    import numpy as np

    n = 24
    num_poles = 24
    test_answers = loadmat(
        f"tests/data/lid_driven_cavity_n_{n}_np_{num_poles}.mat"
    )
    Z = test_answers["Z"]
    Hes = test_answers["Hes"]
    Pol = test_answers["Pol"]
    c = test_answers["c"]
    psi_100_100_answer = test_answers["psi_100_100"]
    p_100_100_answer = test_answers["p_100_100"]
    uv_100_100_answer = test_answers["uv_100_100"]
    omega_100_100_answer = test_answers["omega_100_100"]

    def psi(Z):
        return make_function("psi", Z, c, Hes, Pol)

    def p(Z):
        return make_function("p", Z, c, Hes, Pol)

    def uv(Z):
        return make_function("uv", Z, c, Hes, Pol)

    def omega(Z):
        return make_function("omega", Z, c, Hes, Pol)

    x = np.linspace(-1, 1, 100)
    X, Y = np.meshgrid(x, x)
    Z = X + 1j * Y
    psi_100_100 = psi(Z)
    p_100_100 = p(Z)
    uv_100_100 = uv(Z)
    omega_100_100 = omega(Z)
    assert np.allclose(
        psi_100_100.real, psi_100_100_answer.real, atol=ATOL, rtol=RTOL
    )
    assert np.allclose(
        psi_100_100.imag, psi_100_100_answer.imag, atol=ATOL, rtol=RTOL
    )
    assert np.allclose(
        p_100_100.real, p_100_100_answer.real, atol=ATOL, rtol=RTOL
    )
    assert np.allclose(
        p_100_100.imag, p_100_100_answer.imag, atol=ATOL, rtol=RTOL
    )
    assert np.allclose(
        uv_100_100.real, uv_100_100_answer.real, atol=ATOL, rtol=RTOL
    )
    assert np.allclose(
        uv_100_100.imag, uv_100_100_answer.imag, atol=ATOL, rtol=RTOL
    )
    assert np.allclose(
        omega_100_100.real, omega_100_100_answer.real, atol=ATOL, rtol=RTOL
    )
    assert np.allclose(
        omega_100_100.imag, omega_100_100_answer.imag, atol=ATOL, rtol=RTOL
    )


def test_solve():
    from pyls import Domain, Solver

    # typical BCs
    corners = [1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j]
    dom = Domain(corners, num_boundary_points=100)
    sol = Solver(dom, 10)
    # parabolic inlet flow
    sol.add_boundary_condition("0", "u(0)-y*(1-y)", 0)
    sol.add_boundary_condition("0", "v(0)", 0)
    # 0 pressure and normal velocity
    sol.add_boundary_condition("2", "p(2)", 0)
    sol.add_boundary_condition("2", "v(2)", 0)
    # no slip no penetration on the walls
    sol.add_boundary_condition("1", "u(1)", 0)
    sol.add_boundary_condition("1", "v(1)", 0)
    sol.add_boundary_condition("3", "u(3)", 0)
    sol.add_boundary_condition("3", "v(3)", 0)
    sol.check_boundary_conditions()
    functions = sol.solve()


if __name__ == "__main__":
    test_create_functions()
    test_solve()
