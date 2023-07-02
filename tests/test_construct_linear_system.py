"""Test contructing the linear system."""
from test_settings import ATOL, RTOL


def test_lid_driven_cavity_get_dependents():
    """Test constructing dependents from the correct basis."""
    from scipy.io import loadmat
    from pyls import Domain, Solver
    import numpy as np

    n = 24
    num_poles = 24
    test_answers = loadmat(
        f"tests/data/lid_driven_cavity_n_{n}_np_{num_poles}.mat"
    )
    basis_answer = test_answers["R0"]
    basis_deriv_answer = test_answers["R1"]
    PSI_answer = test_answers["PSI"]
    U_answer = test_answers["U"]
    V_answer = test_answers["V"]

    # set up lid driven cavity problem
    corners = [1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j]
    dom = Domain(corners, num_boundary_points=300, L=np.sqrt(2) * 1.5)
    # dom.show()
    sol = Solver(dom, degree=24)
    sol.basis = basis_answer
    sol.basis_derivatives = basis_deriv_answer
    sol.get_dependents()
    U = sol.U
    V = sol.V
    PSI = sol.stream_function
    assert np.allclose(U, U_answer, atol=ATOL, rtol=RTOL)
    assert np.allclose(V, V_answer, atol=ATOL, rtol=RTOL)
    assert np.allclose(PSI, PSI_answer, atol=ATOL, rtol=RTOL)


def test_lid_driven_cavity_construct_linear_system_1():
    """Test constructing the linear system from the MATLAB basis."""
    from scipy.io import loadmat
    from pyls import Domain, Solver
    import numpy as np
    from test_settings import ATOL, RTOL

    n = 24
    num_poles = 24
    test_answers = loadmat(
        f"tests/data/lid_driven_cavity_n_{n}_np_{num_poles}.mat"
    )
    A_answer = test_answers["A_standard"]
    b_answer = test_answers["rhs_standard"]
    U_answer = test_answers["U"]
    V_answer = test_answers["V"]
    PSI_answer = test_answers["PSI"]
    basis_answer = test_answers["R0"]
    basis_deriv_answer = test_answers["R1"]

    # set up lid driven cavity problem
    corners = [1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j]
    dom = Domain(corners, num_boundary_points=300, L=np.sqrt(2) * 1.5)
    # dom.show()
    sol = Solver(dom, degree=24)
    # moving lid
    sol.add_boundary_condition("0", "psi(0)", 0)
    sol.add_boundary_condition("0", "u(0)", 1)
    # wall boundary conditions
    sol.add_boundary_condition("2", "psi(2)", 0)
    sol.add_boundary_condition("2", "u(2)", 0)
    sol.add_boundary_condition("1", "psi(1)", 0)
    sol.add_boundary_condition("1", "v(1)", 0)
    sol.add_boundary_condition("3", "psi(3)", 0)
    sol.add_boundary_condition("3", "v(3)", 0)

    sol.U = U_answer
    sol.V = V_answer
    sol.stream_function = PSI_answer
    sol.basis = basis_answer
    sol.basis_derivatives = basis_deriv_answer
    sol.construct_linear_system()
    A = sol.A
    b = sol.b
    assert np.allclose(b, b_answer, atol=ATOL, rtol=RTOL)
    assert np.allclose(A, A_answer, atol=ATOL, rtol=RTOL)


def test_lid_driven_cavity_construct_linear_system_2():
    """Test constructing the linear system from scratch."""
    from scipy.io import loadmat, savemat
    from pyls import Domain, Solver
    from pyls.numerics import va_orthogonalise, va_evaluate
    import numpy as np
    from test_settings import ATOL, RTOL

    n = 24
    num_poles = 24
    test_answers = loadmat(
        f"tests/data/lid_driven_cavity_n_{n}_np_{num_poles}.mat"
    )
    A_answer = test_answers["A_standard"]
    b_answer = test_answers["rhs_standard"]
    A_weighted_answer = test_answers["A_weighted"]
    b_weighted_answer = test_answers["rhs_weighted"]
    A_normalized_answer = test_answers["A_normalized"]
    b_normalized_answer = test_answers["rhs_normalized"]
    U_answer = test_answers["U"]
    V_answer = test_answers["V"]
    PSI_answer = test_answers["PSI"]
    basis_answer = test_answers["R0"]
    basis_deriv_answer = test_answers["R1"]

    # set up lid driven cavity problem
    corners = [1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j]
    dom = Domain(corners, num_boundary_points=300, L=np.sqrt(2) * 1.5)
    # dom.show()
    sol = Solver(dom, degree=24)
    # moving lid
    sol.add_boundary_condition("0", "psi(0)", 0)
    sol.add_boundary_condition("0", "u(0)", 1)
    # wall boundary conditions
    sol.add_boundary_condition("2", "psi(2)", 0)
    sol.add_boundary_condition("2", "u(2)", 0)
    sol.add_boundary_condition("1", "psi(1)", 0)
    sol.add_boundary_condition("1", "v(1)", 0)
    sol.add_boundary_condition("3", "psi(3)", 0)
    sol.add_boundary_condition("3", "v(3)", 0)
    sol.hessenbergs, sol.Q = va_orthogonalise(
        sol.boundary_points, sol.degree, sol.domain.poles
    )
    sol.basis, sol.basis_derivatives = va_evaluate(
        sol.boundary_points, sol.hessenbergs, sol.domain.poles
    )
    assert np.allclose(sol.basis, basis_answer, atol=ATOL, rtol=RTOL)
    assert np.allclose(
        sol.basis_derivatives, basis_deriv_answer, atol=ATOL, rtol=RTOL
    )
    sol.get_dependents()
    RTOL = 1e-6
    assert np.allclose(sol.U, U_answer, atol=ATOL, rtol=RTOL)
    assert np.allclose(sol.V, V_answer, atol=ATOL, rtol=RTOL)
    assert np.allclose(sol.stream_function, PSI_answer, atol=ATOL, rtol=RTOL)
    sol.construct_linear_system()
    A = sol.A
    b = sol.b
    assert np.allclose(b, b_answer, atol=ATOL, rtol=RTOL)
    assert np.allclose(A, A_answer, atol=ATOL, rtol=RTOL)
    sol.weight_rows()
    assert np.allclose(sol.A, A_weighted_answer, atol=ATOL, rtol=RTOL)
    assert np.allclose(sol.b, b_weighted_answer, atol=ATOL, rtol=RTOL)
    sol.normalize()
    assert np.allclose(sol.A, A_normalized_answer, atol=ATOL, rtol=RTOL)
    assert np.allclose(sol.b, b_normalized_answer, atol=ATOL, rtol=RTOL)
    savemat(
        "tests/data/lid_driven_cavity_matrix_python.mat",
        {"A": sol.A, "b": sol.b},
    )


def test_row_weighting():
    """Test weighting rows of the linear system."""
    from pyls import Domain, Solver
    import numpy as np
    from scipy.io import loadmat

    n = 24
    num_poles = 24
    test_answers = loadmat(
        f"tests/data/lid_driven_cavity_n_{n}_np_{num_poles}.mat"
    )
    A_standard_answer = test_answers["A_standard"]
    rhs_standard_answer = test_answers["rhs_standard"]
    A_weighted_answer = test_answers["A_weighted"]
    rhs_weighted_answer = test_answers["rhs_weighted"]
    row_weights_answer = test_answers["row_weights"]
    Z = test_answers["Z"]
    corners = test_answers["w"]
    corners = [1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j]
    dom = Domain(corners, num_boundary_points=300, L=np.sqrt(2) * 1.5)
    assert np.allclose(dom.corners, corners, atol=ATOL, rtol=RTOL)
    assert np.allclose(dom.boundary_points, Z, atol=ATOL, rtol=RTOL)
    sol = Solver(dom, degree=24)
    sol.A = A_standard_answer
    sol.b = rhs_standard_answer
    row_weights = sol.weight_rows()
    assert np.allclose(row_weights, row_weights_answer, atol=ATOL, rtol=RTOL)
    assert np.allclose(sol.b, rhs_weighted_answer, atol=ATOL, rtol=RTOL)
    assert np.allclose(sol.A, A_weighted_answer, atol=ATOL, rtol=RTOL)


def test_normalize():
    """Test normalize method of Solver."""
    pass


if __name__ == "__main__":
    test_lid_driven_cavity_get_dependents()
    test_lid_driven_cavity_construct_linear_system_1()
    test_lid_driven_cavity_construct_linear_system_2()
    test_row_weighting()
    test_normalize()
