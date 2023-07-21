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
        sigma=4,
        deg_poly=24,
        num_poles=num_poles,
    )
    # prob.show()
    solver = Solver(prob)
    solver.basis = basis_answer
    solver.basis_derivatives = basis_deriv_answer
    solver.basis_derivatives_2 = np.zeros_like(basis_deriv_answer)
    solver.get_dependents()
    U = solver.U
    V = solver.V
    PSI = solver.PSI
    assert np.allclose(U, U_answer, atol=ATOL, rtol=RTOL)
    assert np.allclose(V, V_answer, atol=ATOL, rtol=RTOL)
    assert np.allclose(PSI, PSI_answer, atol=ATOL, rtol=RTOL)


def test_single_circle_get_dependents():
    """Test constructing dependents from the correct basis."""
    from scipy.io import loadmat
    from pylars import Problem, Solver
    import numpy as np

    test_answers = loadmat("tests/data/single_circle_test.mat")
    deg_poly, num_poles, deg_laurent = (
        test_answers["n"][0][0],
        test_answers["np"][0][0],
        test_answers["nl"][0][0],
    )
    num_edge_points, num_ellipse_points = (
        test_answers["nb"][0][0],
        test_answers["np"][0][0],
    )
    basis_answer = test_answers["R0"]
    basis_deriv_answer = test_answers["R1"]
    U_answer = test_answers["U"]
    V_answer = test_answers["V"]
    PSI_answer = test_answers["PSI"]
    P_answer = test_answers["P"]
    prob = Problem()
    corners = [-1 - 1j, 1 - 1j, 1 + 1j, -1 + 1j]
    prob.add_exterior_polygon(
        corners,
        num_edge_points=num_edge_points,
        num_poles=num_poles,
        deg_poly=deg_poly,
        spacing="linear",
    )
    prob.add_interior_curve(
        lambda t: 0.5 * np.exp(2j * np.pi * t),
        num_points=num_ellipse_points,
        deg_laurent=deg_laurent,
        centroid=0.0 + 0.0j,
    )
    solver = Solver(prob)
    solver.basis = basis_answer
    solver.basis_derivatives = basis_deriv_answer
    solver.basis_derivatives_2 = np.zeros_like(basis_deriv_answer)
    solver.get_dependents()
    U = solver.U
    V = solver.V
    PSI = solver.PSI
    P = solver.P
    assert np.allclose(U, U_answer, atol=ATOL, rtol=RTOL)
    assert np.allclose(V, V_answer, atol=ATOL, rtol=RTOL)
    assert np.allclose(PSI, PSI_answer, atol=ATOL, rtol=RTOL)
    assert np.allclose(P, P_answer, atol=ATOL, rtol=RTOL)


def test_three_circles_get_dependents():
    """Test constructing dependents from the correct basis."""
    from scipy.io import loadmat
    from pylars import Problem, Solver
    import numpy as np

    test_answers = loadmat("tests/data/three_circles_test.mat")
    deg_poly, num_poles, deg_laurent = (
        test_answers["n"][0][0],
        test_answers["np"][0][0],
        test_answers["nl"][0][0],
    )
    num_edge_points, num_ellipse_points = (
        test_answers["nb"][0][0],
        test_answers["np"][0][0],
    )
    Z_answer = test_answers["Z"]
    basis_answer = test_answers["R0"]
    basis_deriv_answer = test_answers["R1"]
    U_answer = test_answers["U"]
    V_answer = test_answers["V"]
    PSI_answer = test_answers["PSI"]
    P_answer = test_answers["P"]
    prob = Problem()
    corners = [-1 - 1j, 1 - 1j, 1 + 1j, -1 + 1j]
    prob = Problem()
    prob.add_exterior_polygon(
        corners=corners,
        num_edge_points=num_edge_points,
        num_poles=num_poles,
        deg_poly=deg_poly,
        spacing="linear",
    )
    rs = [0.15, 0.15, 0.15]
    cs = [0.0 + 0.0j, 0.5 + 0.5j, -0.5 - 0.5j]
    for r, c in zip(rs, cs):
        prob.add_interior_curve(
            lambda t: c + r * np.exp(2j * np.pi * t),
            num_points=num_ellipse_points,
            deg_laurent=deg_laurent,
            centroid=c,
        )
    assert np.allclose(
        prob.domain.boundary_points, Z_answer, atol=ATOL, rtol=RTOL
    )
    solver = Solver(prob)
    solver.basis = basis_answer
    solver.basis_derivatives = basis_deriv_answer
    solver.basis_derivatives_2 = np.zeros_like(basis_deriv_answer)
    solver.get_dependents()
    U = solver.U
    V = solver.V
    PSI = solver.PSI
    P = solver.P
    # log terms are wrong for three circles but correct for 1
    assert np.allclose(U, U_answer, atol=ATOL, rtol=RTOL)
    assert np.allclose(V, V_answer, atol=ATOL, rtol=RTOL)
    assert np.allclose(PSI, PSI_answer, atol=ATOL, rtol=RTOL)
    assert np.allclose(P, P_answer, atol=ATOL, rtol=RTOL)


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
        sigma=4,
        deg_poly=24,
        num_poles=num_poles,
    )
    # moving lid
    prob.add_boundary_condition("0", "psi[0]", 0)
    prob.add_boundary_condition("0", "u[0]", 1)
    # wall boundary conditions
    prob.add_boundary_condition("2", "psi[2]", 0)
    prob.add_boundary_condition("2", "u[2]", 0)
    prob.add_boundary_condition("1", "psi[1]", 0)
    prob.add_boundary_condition("1", "v[1]", 0)
    prob.add_boundary_condition("3", "psi[3]", 0)
    prob.add_boundary_condition("3", "v[3]", 0)

    solver = Solver(prob)

    solver.U = U_answer
    solver.V = V_answer
    solver.PSI = PSI_answer
    solver.basis = basis_answer
    solver.basis_derivatives = basis_deriv_answer
    solver.basis_derivatives_2 = np.zeros_like(basis_deriv_answer)
    solver.construct_linear_system()
    A = solver.A
    b = solver.b
    assert np.allclose(b, b_answer, atol=ATOL, rtol=RTOL)
    assert np.allclose(A, A_answer, atol=ATOL, rtol=RTOL)


def test_lid_driven_cavity_construct_linear_system_2():
    """Test constructing the linear system from scratch."""
    from scipy.io import loadmat
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
        sigma=4,
        deg_poly=n,
        num_poles=num_poles,
    )
    prob.add_boundary_condition("0", "psi[0]", 0)
    prob.add_boundary_condition("0", "u[0]", 1)
    prob.add_boundary_condition("2", "psi[2]", 0)
    prob.add_boundary_condition("2", "u[2]", 0)
    prob.add_boundary_condition("1", "psi[1]", 0)
    prob.add_boundary_condition("1", "v[1]", 0)
    prob.add_boundary_condition("3", "psi[3]", 0)
    prob.add_boundary_condition("3", "v[3]", 0)
    solver = Solver(prob)
    solver.hessenbergs, solver.Q = va_orthogonalise(
        solver.boundary_points, solver.degree, solver.poles
    )
    (
        solver.basis,
        solver.basis_derivatives,
        solver.basis_derivatives_2,
    ) = va_evaluate(
        solver.boundary_points,
        solver.hessenbergs,
        solver.poles,
        second_deriv=True,
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
        sigma=4,
        deg_poly=24,
        num_poles=num_poles,
    )
    assert np.allclose(prob.domain.corners, corners, atol=ATOL, rtol=RTOL)
    assert np.allclose(prob.domain.boundary_points, Z, atol=ATOL, rtol=RTOL)
    solver = Solver(prob)
    solver.A = A_standard_answer
    solver.b = rhs_standard_answer
    row_weights = solver.weight_rows()
    assert np.allclose(row_weights, row_weights_answer, atol=ATOL, rtol=RTOL)
    assert np.allclose(solver.b, rhs_weighted_answer, atol=ATOL, rtol=RTOL)
    assert np.allclose(solver.A, A_weighted_answer, atol=ATOL, rtol=RTOL)


if __name__ == "__main__":
    test_lid_driven_cavity_get_dependents()
    test_single_circle_get_dependents()
    test_three_circles_get_dependents()
    test_lid_driven_cavity_construct_linear_system_1()
    test_lid_driven_cavity_construct_linear_system_2()
    test_row_weighting()
