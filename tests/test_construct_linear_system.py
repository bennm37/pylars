def test_lid_driven_cavity_dependents():
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
    sol.setup()
    # check basis functions
    assert np.allclose(sol.basis.real, basis_answer.real)
    assert np.allclose(sol.basis.imag, basis_answer.imag)
    assert np.allclose(sol.basis_derivatives, basis_deriv_answer)
    assert np.allclose(sol.basis_derivatives.real, basis_deriv_answer.real)
    assert np.allclose(sol.basis_derivatives.imag, basis_deriv_answer.imag)
    U = sol.U
    V = sol.V
    PSI = sol.stream_fuction
    # test a block of MATLAB against Python basis
    assert np.allclose(U[:, 121:242], sol.basis_derivatives.real)
    assert np.allclose(U_answer[:, 121:242], basis_deriv_answer.real)
    assert np.allclose(U[:, 121:242], basis_deriv_answer.real)
    assert np.allclose(U, U_answer)
    assert np.allclose(V, V_answer)
    assert np.allclose(PSI, PSI_answer)


def test_lid_driven_cavity_linear_system():
    from scipy.io import loadmat
    from pyls import Domain, Solver
    import numpy as np

    A_answer = loadmat("tests/data/lid_driven_cavity_A.mat")["A"]
    b_answer = loadmat("tests/data/lid_driven_cavity_rhs.mat")["rhs"]

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

    sol.setup()
    sol.construct_linear_system()
    A = sol.A
    b = sol.b
    assert np.allclose(A, A_answer)
    assert np.allclose(b, b_answer)

def test_row_weighting():
    from pyls import Domain, Solver
    import numpy as np
    from scipy.io import loadmat
    
    n = 24
    num_poles = 24
    test_answers = loadmat(
        f"tests/data/lid_driven_cavity_n_{n}_np_{num_poles}.mat"
    )
    A_standard = test_answers["A_standard"]
    rhs_standard = test_answers["rhs_standard"]
    A_weighted = test_answers["A_weighted"]
    rhs_weighted = test_answers["rhs_weighted"]
    Z = test_answers["Z"]
    corners = test_answers["w"]
    corners = [1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j]
    dom = Domain(corners, num_boundary_points=300, L=np.sqrt(2) * 1.5)
    assert np.allclose(dom.corners, corners, atol=1e-15)
    assert np.allclose(dom.boundary_points, Z, atol=1e-15)
    sol = Solver(dom, degree=24)
    sol.A = A_standard
    sol.b = rhs_standard
    sol.weight_rows()
    assert np.allclose(sol.A, A_weighted)
    assert np.allclose(sol.b, rhs_weighted)


def test_row_weighting():
    from pyls import Domain, Solver
    import numpy as np
    from scipy.io import loadmat

    n = 24
    num_poles = 24
    test_answers = loadmat(
        f"tests/data/lid_driven_cavity_n_{n}_np_{num_poles}.mat"
    )
    A_standard = test_answers["A_standard"]
    rhs_standard = test_answers["rhs_standard"]
    A_weighted = test_answers["A_weighted"]
    rhs_weighted = test_answers["rhs_weighted"]
    Z = test_answers["Z"]
    corners = test_answers["w"]
    corners = [1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j]
    dom = Domain(corners, num_boundary_points=300, L=np.sqrt(2) * 1.5)
    assert np.allclose(dom.corners, corners, atol=1e-15)
    assert np.allclose(dom.boundary_points, Z, atol=1e-15)
    sol = Solver(dom, degree=24)
    sol.A = A_standard
    sol.b = rhs_standard
    sol.weight_rows()
    assert np.allclose(sol.A, A_weighted)
    assert np.allclose(sol.b, rhs_weighted)


if __name__ == "__main__":
    # test_lid_driven_cavity_dependents()
    test_row_weighting()
    # test_lid_driven_cavity_linear_system()
