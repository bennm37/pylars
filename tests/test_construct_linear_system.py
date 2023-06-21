def test_lid_driven_cavity_dependents():
    from scipy.io import loadmat
    from pyls import Domain, Solver
    import numpy as np

    basis_answer = loadmat("tests/data/lid_driven_cavity_R0.mat")["R0"]
    basis_deriv_answer = loadmat("tests/data/lid_driven_cavity_R1.mat")["R1"]
    PSI_answer = loadmat("tests/data/lid_driven_cavity_PSI.mat")["PSI"]
    U_answer = loadmat("tests/data/lid_driven_cavity_U.mat")["U"]
    V_answer = loadmat("tests/data/lid_driven_cavity_V.mat")["V"]

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


if __name__ == "__main__":
    test_lid_driven_cavity_dependents()
    test_lid_driven_cavity_linear_system()
