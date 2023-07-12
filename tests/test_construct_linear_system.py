"""Test contructing the linear system."""
from test_settings import ATOL, RTOL


def test_lid_driven_cavity_get_dependents():
    """Test constructing dependents from the correct basis."""
    from scipy.io import loadmat
    from pylars import Problem, Solver
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
    prob = Problem()
    prob.add_exterior_polygon(
        corners,
        num_edge_points=300,
        length_scale=1.5 * np.sqrt(2),
        deg_poly=24,
        num_poles=num_poles,
    )
    # prob.show()
    sol = Solver(prob)
    sol.basis = basis_answer
    sol.basis_derivatives = basis_deriv_answer
    sol.get_dependents()
    U = sol.U
    V = sol.V
    PSI = sol.PSI
    assert np.allclose(U, U_answer, atol=ATOL, rtol=RTOL)
    assert np.allclose(V, V_answer, atol=ATOL, rtol=RTOL)
    assert np.allclose(PSI, PSI_answer, atol=ATOL, rtol=RTOL)


def test_lid_driven_cavity_construct_linear_system_1():
    """Test constructing the linear system from the MATLAB basis."""
    from scipy.io import loadmat
    from pylars import Problem, Solver
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
    prob = Problem()
    prob.add_exterior_polygon(
        corners,
        num_edge_points=300,
        length_scale=1.5 * np.sqrt(2),
        deg_poly=24,
        num_poles=num_poles,
    )
    # moving lid
    prob.add_boundary_condition("0", "psi(0)", 0)
    prob.add_boundary_condition("0", "u(0)", 1)
    # wall boundary conditions
    prob.add_boundary_condition("2", "psi(2)", 0)
    prob.add_boundary_condition("2", "u(2)", 0)
    prob.add_boundary_condition("1", "psi(1)", 0)
    prob.add_boundary_condition("1", "v(1)", 0)
    prob.add_boundary_condition("3", "psi(3)", 0)
    prob.add_boundary_condition("3", "v(3)", 0)

    solver = Solver(prob)

    solver.U = U_answer
    solver.V = V_answer
    solver.PSI = PSI_answer
    solver.basis = basis_answer
    solver.basis_derivatives = basis_deriv_answer
    solver.construct_linear_system()
    A = solver.A
    b = solver.b
    assert np.allclose(b, b_answer, atol=ATOL, rtol=RTOL)
    assert np.allclose(A, A_answer, atol=ATOL, rtol=RTOL)


def test_lid_driven_cavity_construct_linear_system_2():
    """Test constructing the linear system from scratch."""
    from scipy.io import loadmat, savemat
    from pylars import Problem, Solver
    from pylars.numerics import va_orthogonalise, va_evaluate
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
    prob = Problem()
    prob.add_exterior_polygon(
        corners,
        num_edge_points=300,
        length_scale=1.5 * np.sqrt(2),
        deg_poly=24,
        num_poles=num_poles,
    )
    prob.add_boundary_condition("0", "psi(0)", 0)
    prob.add_boundary_condition("0", "u(0)", 1)
    prob.add_boundary_condition("2", "psi(2)", 0)
    prob.add_boundary_condition("2", "u(2)", 0)
    prob.add_boundary_condition("1", "psi(1)", 0)
    prob.add_boundary_condition("1", "v(1)", 0)
    prob.add_boundary_condition("3", "psi(3)", 0)
    prob.add_boundary_condition("3", "v(3)", 0)
    solver = Solver(prob)
    solver.hessenbergs, solver.Q = va_orthogonalise(
        solver.boundary_points, solver.degree, solver.probain.poles
    )
    solver.basis, solver.basis_derivatives = va_evaluate(
        solver.boundary_points, solver.hessenbergs, solver.probain.poles
    )
    assert np.allclose(solver.basis, basis_answer, atol=ATOL, rtol=RTOL)
    assert np.allclose(
        solver.basis_derivatives, basis_deriv_answer, atol=ATOL, rtol=RTOL
    )
    solver.get_dependents()
    RTOL = 1e-6
    assert np.allclose(solver.U, U_answer, atol=ATOL, rtol=RTOL)
    assert np.allclose(solver.V, V_answer, atol=ATOL, rtol=RTOL)
    assert np.allclose(solver.PSI, PSI_answer, atol=ATOL, rtol=RTOL)
    solver.construct_linear_system()
    A = solver.A
    b = solver.b
    assert np.allclose(b, b_answer, atol=ATOL, rtol=RTOL)
    assert np.allclose(A, A_answer, atol=ATOL, rtol=RTOL)
    solver.weight_rows()
    assert np.allclose(solver.A, A_weighted_answer, atol=ATOL, rtol=RTOL)
    assert np.allclose(solver.b, b_weighted_answer, atol=ATOL, rtol=RTOL)
    solver.normalize()
    assert np.allclose(solver.A, A_normalized_answer, atol=ATOL, rtol=RTOL)
    assert np.allclose(solver.b, b_normalized_answer, atol=ATOL, rtol=RTOL)
    savemat(
        "tests/data/lid_driven_cavity_matrix_python.mat",
        {"A": solver.A, "b": solver.b},
    )


def test_row_weighting():
    """Test weighting rows of the linear system."""
    from pylars import Problem, Solver
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
    prob = Problem()
    prob.add_exterior_polygon(
        corners,
        num_edge_points=300,
        length_scale=1.5 * np.sqrt(2),
        deg_poly=24,
        num_poles=num_poles,
    )
    assert np.allclose(prob.corners, corners, atol=ATOL, rtol=RTOL)
    assert np.allclose(prob.boundary_points, Z, atol=ATOL, rtol=RTOL)
    sol = Solver(prob, degree=24)
    sol.A = A_standard_answer
    sol.b = rhs_standard_answer
    row_weights = sol.weight_rows()
    assert np.allclose(row_weights, row_weights_answer, atol=ATOL, rtol=RTOL)
    assert np.allclose(sol.b, rhs_weighted_answer, atol=ATOL, rtol=RTOL)
    assert np.allclose(sol.A, A_weighted_answer, atol=ATOL, rtol=RTOL)


def test_normalize():
    pass


if __name__ == "__main__":
    test_lid_driven_cavity_get_dependents()
    test_lid_driven_cavity_construct_linear_system_1()
    test_lid_driven_cavity_construct_linear_system_2()
    test_row_weighting()
