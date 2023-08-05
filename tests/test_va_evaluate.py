"""Test the va_evaluate function.

This is used to generate the basis functions and their derivatives
given the boundary points, the hessenbergs and the poles.
"""
from test_settings import ATOL, RTOL


def test_import_va_evaluate():
    """Test that the va_evaluate function can be imported."""
    from pylars.numerics import va_evaluate

    assert va_evaluate is not None


def test_small_va_evaluate():
    """Test the va_evaluate function for a small example."""
    from pylars.numerics import va_orthogonalise, va_evaluate
    import numpy as np

    Z = np.array([0, 2, 1 + 1j]).reshape(3, 1)
    n = 1
    hessenbergs, Q = va_orthogonalise(Z, n)
    H = hessenbergs[0]
    basis, basis_deriv = va_evaluate(Z, hessenbergs)
    basis_deriv_answer = np.array(
        [[0, 0, 0], [1 / H[1, 0], 1 / H[1, 0], 1 / H[1, 0]]]
    ).T
    assert np.allclose(basis, Q, atol=ATOL, rtol=RTOL)
    assert np.allclose(basis_deriv, basis_deriv_answer, atol=ATOL, rtol=RTOL)
    assert np.allclose(basis_deriv, basis_deriv_answer, atol=ATOL, rtol=RTOL)


def test_small_second_va_evaluate():
    """Investigating va_evaluate."""
    from pylars.numerics import va_evaluate
    from tests.test_settings import ATOL, RTOL
    import numpy as np

    X = np.linspace(0, 1, 100)
    Z = X + 1j * X[::-1]
    H = np.array([[1, 1, 1], [1, 1, 1], [0, 1, 1], [0, 0, 1]])
    hessenbergs = [H]
    basis, basis_deriv, basis_second_deriv = va_evaluate(
        Z, hessenbergs, second_deriv=True
    )
    # allowing lambdas for conciseness
    poly0 = lambda z: np.ones_like(z)
    poly_deriv0 = lambda z: np.zeros_like(z)
    poly_second_deriv0 = lambda z: np.zeros_like(z)
    poly1 = lambda z: z - 1
    poly_deriv1 = lambda z: np.ones_like(z)
    poly_second_deriv1 = lambda z: np.zeros_like(z)
    poly2 = lambda z: z**2 - 2 * z
    poly_deriv2 = lambda z: 2 * z - 2
    poly_second_deriv2 = lambda z: 2 * np.ones_like(z)
    poly3 = lambda z: z**3 - 3 * z**2 + z
    poly_deriv3 = lambda z: 3 * z**2 - 6 * z + 1
    poly_second_deriv3 = lambda z: 6 * z - 6 * np.ones_like(z)
    basis_answer = np.array([poly0(Z), poly1(Z), poly2(Z), poly3(Z)]).T
    basis_deriv_answer = np.array(
        [poly_deriv0(Z), poly_deriv1(Z), poly_deriv2(Z), poly_deriv3(Z)]
    ).T
    basis_second_deriv_answer = np.array(
        [
            poly_second_deriv0(Z),
            poly_second_deriv1(Z),
            poly_second_deriv2(Z),
            poly_second_deriv3(Z),
        ]
    ).T
    assert np.allclose(basis, basis_answer, atol=ATOL, rtol=RTOL)
    assert np.allclose(basis_deriv, basis_deriv_answer, atol=ATOL, rtol=RTOL)
    assert np.allclose(
        basis_second_deriv, basis_second_deriv_answer, atol=ATOL, rtol=RTOL
    )


def test_small_poles_second_va_evaluate():
    """Investigating va_evaluate."""
    from pylars.numerics import va_evaluate
    from tests.test_settings import ATOL, RTOL
    import numpy as np

    X = np.linspace(0, 1, 100)
    Z = X + 1j * X[::-1]
    poles = [np.array([2, 3])]
    H = np.array([[1, 1], [1, 1], [0, 1]])
    hessenbergs = [H, H]
    basis, basis_deriv, basis_second_deriv = va_evaluate(
        Z, hessenbergs, poles=poles, second_deriv=True
    )
    # polynomial part
    poly0 = lambda z: np.ones_like(z)
    poly_deriv0 = lambda z: np.zeros_like(z)
    poly_second_deriv0 = lambda z: np.zeros_like(z)
    poly1 = lambda z: z - 1
    poly_deriv1 = lambda z: np.ones_like(z)
    poly_second_deriv1 = lambda z: np.zeros_like(z)
    poly2 = lambda z: z**2 - 2 * z
    poly_deriv2 = lambda z: 2 * z - 2
    poly_second_deriv2 = lambda z: 2 * np.ones_like(z)
    # pole part
    pole1 = lambda z: 1 / (z - 2) - 1
    pole_deriv1 = lambda z: -1 / (z - 2) ** 2
    pole_second_deriv1 = lambda z: 2 / (z - 2) ** 3
    pole2 = lambda z: 1 / ((z - 2) * (z - 3)) - 1 / (z - 2) - 1 / (z - 3)
    pole_deriv2 = (
        lambda z: 1 / (z - 2) ** 2
        + 1 / (z - 3) ** 2
        - (2 * z - 5) / ((z - 2) * (z - 3)) ** 2
    )
    pole_second_deriv2 = lambda z: -4 / (z - 2) ** 3
    basis_answer = np.array(
        [poly0(Z), poly1(Z), poly2(Z), pole1(Z), pole2(Z)]
    ).T
    basis_deriv_answer = np.array(
        [
            poly_deriv0(Z),
            poly_deriv1(Z),
            poly_deriv2(Z),
            pole_deriv1(Z),
            pole_deriv2(Z),
        ]
    ).T
    basis_second_deriv_answer = np.array(
        [
            poly_second_deriv0(Z),
            poly_second_deriv1(Z),
            poly_second_deriv2(Z),
            pole_second_deriv1(Z),
            pole_second_deriv2(Z),
        ]
    ).T
    assert np.allclose(basis, basis_answer, atol=ATOL, rtol=RTOL)
    assert np.allclose(basis_deriv, basis_deriv_answer, atol=ATOL, rtol=RTOL)
    assert np.allclose(
        basis_second_deriv, basis_second_deriv_answer, atol=ATOL, rtol=RTOL
    )


def test_small_laurent_second_va_evaluate():
    """Investigating va_evaluate."""
    from pylars.numerics import va_evaluate
    from tests.test_settings import ATOL, RTOL
    import numpy as np

    X = np.linspace(0, 1, 100)
    Z = X + 1j * X[::-1]
    laurents = [(2.0 + 0.0j, 2), (3.0 + 0.0j, 1)]
    H = np.array([[1, 1], [1, 1], [0, 1]])
    H1 = np.array([[1], [1]])
    hessenbergs = [H, H, H1]
    basis, basis_deriv, basis_second_deriv = va_evaluate(
        Z, hessenbergs, interior_laurents=laurents, second_deriv=True
    )
    # polynomial part
    poly0 = lambda z: np.ones_like(z)
    poly_deriv0 = lambda z: np.zeros_like(z)
    poly_second_deriv0 = lambda z: np.zeros_like(z)
    poly1 = lambda z: z - 1
    poly_deriv1 = lambda z: np.ones_like(z)
    poly_second_deriv1 = lambda z: np.zeros_like(z)
    poly2 = lambda z: z**2 - 2 * z
    poly_deriv2 = lambda z: 2 * z - 2
    poly_second_deriv2 = lambda z: 2 * np.ones_like(z)
    # laurent series part for 2
    laurent_21 = lambda z: 1 / (z - 2) - 1
    laurent_2_deriv1 = lambda z: -1 / (z - 2) ** 2
    laurent_2_second_deriv1 = lambda z: 2 / (z - 2) ** 3
    laurent_22 = lambda z: 1 / (z - 2) ** 2 - 2 / (z - 2)
    laurent_2_deriv2 = lambda z: -2 / (z - 2) ** 3 + 2 / (z - 2) ** 2
    laurent_2_second_deriv2 = lambda z: 6 / (z - 2) ** 4 - 4 / (z - 2) ** 3
    # laurent series part for 3
    laurent_31 = lambda z: 1 / (z - 3) - 1
    laurent_3_deriv1 = lambda z: -1 / (z - 3) ** 2
    laurent_3_second_deriv1 = lambda z: 2 / (z - 3) ** 3
    basis_answer = np.array(
        [
            poly0(Z),
            poly1(Z),
            poly2(Z),
            laurent_21(Z),
            laurent_22(Z),
            laurent_31(Z),
        ]
    ).T
    basis_deriv_answer = np.array(
        [
            poly_deriv0(Z),
            poly_deriv1(Z),
            poly_deriv2(Z),
            laurent_2_deriv1(Z),
            laurent_2_deriv2(Z),
            laurent_3_deriv1(Z),
        ]
    ).T
    basis_second_deriv_answer = np.array(
        [
            poly_second_deriv0(Z),
            poly_second_deriv1(Z),
            poly_second_deriv2(Z),
            laurent_2_second_deriv1(Z),
            laurent_2_second_deriv2(Z),
            laurent_3_second_deriv1(Z),
        ]
    ).T
    assert np.allclose(basis, basis_answer, atol=ATOL, rtol=RTOL)
    assert np.allclose(basis_deriv, basis_deriv_answer, atol=ATOL, rtol=RTOL)
    assert np.allclose(
        basis_second_deriv, basis_second_deriv_answer, atol=ATOL, rtol=RTOL
    )


def test_large_va_evaluate():
    """Test va_evaluate for a large example generated by the MATLAB code."""
    from pylars.numerics import va_orthogonalise, va_evaluate
    import numpy as np
    from scipy.io import loadmat

    Z = np.exp(1j * np.linspace(0, 2 * np.pi, 100)).reshape(100, 1)
    hessenbergs, Q = va_orthogonalise(Z, 10)
    basis, basis_deriv = va_evaluate(Z, hessenbergs)
    basis_answer = loadmat("tests/data/VAorthog_circle_R0.mat")["R0"]
    basis_deriv_answer = loadmat("tests/data/VAorthog_circle_R1.mat")["R1"]
    assert np.allclose(basis, basis_answer, atol=ATOL, rtol=RTOL)
    assert np.allclose(basis_deriv, basis_deriv_answer, atol=ATOL, rtol=RTOL)


def test_small_poles_va_evaluate_1():
    """Test the va_orthogonalise with poles for a small example."""
    from pylars.numerics import va_orthogonalise, va_evaluate
    import numpy as np

    Z = np.array([-4, 4]).reshape(2, 1)
    poles = [np.array([0])]
    hessenbergs, Q = va_orthogonalise(Z, 1, poles)
    basis, basis_deriv = va_evaluate(Z, hessenbergs, poles)
    basis_answer = np.array([[1, -1, -1], [1, 1, 1]])
    basis_deriv_answer = np.array([[0, 1 / 4, -1 / 4], [0, 1 / 4, -1 / 4]])
    assert np.allclose(basis, basis_answer, atol=ATOL, rtol=RTOL)
    assert np.allclose(basis_deriv, basis_deriv_answer, atol=ATOL, rtol=RTOL)


def test_small_poles_va_evaluate_2():
    """Test the va_orthogonalise with poles against the MATLAB code."""
    from pylars.numerics import va_orthogonalise, va_evaluate
    import numpy as np
    from scipy.io import loadmat

    Z = loadmat("tests/data/small_poles.mat")["Z"].reshape(3, 1)
    poles = loadmat("tests/data/small_poles.mat")["Pol"]
    poles = [
        poles[i, 0].reshape(
            -1,
        )
        for i in range(poles.shape[0])
    ]
    basis_answer = loadmat("tests/data/small_poles.mat")["R0"]
    basis_deriv_answer = loadmat("tests/data/small_poles.mat")["R1"]
    hessenbergs, Q = va_orthogonalise(Z, 0, poles)
    basis, basis_deriv = va_evaluate(Z, hessenbergs, poles)
    assert np.allclose(basis, basis_answer, atol=ATOL, rtol=RTOL)
    assert np.allclose(basis_deriv, basis_deriv_answer, atol=ATOL, rtol=RTOL)


def test_large_poles_va_evaluate():
    """Test the va_orthogonalise with poles against the MATLAB code."""
    from pylars.numerics import va_orthogonalise, va_evaluate
    from pylars import Problem
    import numpy as np
    from scipy.io import loadmat

    n = 24
    num_poles = 24
    test_answers = loadmat(
        f"tests/data/lid_driven_cavity_n_{n}_np_{num_poles}.mat"
    )
    Z_answer = test_answers["Z"]
    poles_answer = test_answers["Pol"]
    basis_answer = test_answers["R0"]
    basis_deriv_answer = test_answers["R1"]
    poles_answer = np.array([poles_answer[0, i] for i in range(4)]).reshape(
        4, num_poles
    )
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
    # check the MATALB prob.domainain points and poles are the same
    # check the polynomial coefficients are the same
    assert np.allclose(
        prob.domain.boundary_points, Z_answer, atol=ATOL, rtol=RTOL
    )
    assert np.allclose(prob.domain.poles, poles_answer, atol=ATOL, rtol=RTOL)

    hessenbergs, Q = va_orthogonalise(
        prob.domain.boundary_points.reshape(1200, 1),
        n,
        poles=prob.domain.poles,
    )
    basis, basis_deriv = va_evaluate(
        prob.domain.boundary_points, hessenbergs, poles=prob.domain.poles
    )
    # check the polynomial basis is the same
    assert np.allclose(
        basis[:, :25], basis_answer[:, :25], atol=ATOL, rtol=RTOL
    )
    assert np.allclose(
        basis_deriv[:, :25],
        basis_deriv_answer[:, :25],
        atol=ATOL,
        rtol=RTOL,
    )
    # check all the basis functions are the same
    assert np.allclose(basis, basis_answer, atol=ATOL, rtol=RTOL)
    assert np.allclose(basis_deriv, basis_deriv_answer, atol=ATOL, rtol=RTOL)


def test_large_poles_va_evaluate_hypothesis():
    """Test the va_evaluate with poles against the MATLAB code."""
    from pylars.numerics import va_orthogonalise, va_evaluate
    from pylars import Problem
    import numpy as np
    from scipy.io import loadmat

    n = 0
    num_poles = 10
    test_answers = loadmat(
        f"tests/data/lid_driven_cavity_n_{n}_np_{num_poles}.mat"
    )
    Z_answer = test_answers["Z"]
    poles_answer = test_answers["Pol"]
    basis_answer = test_answers["R0"]
    basis_deriv_answer = test_answers["R1"]
    poles_answer = np.array([poles_answer[0, i] for i in range(4)]).reshape(
        4, num_poles
    )
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
    # check the MATALB domain points and poles are the same
    # check the polynomial coefficients are the same
    assert np.allclose(
        prob.domain.boundary_points, Z_answer, atol=ATOL, rtol=RTOL
    )
    assert np.allclose(prob.domain.poles, poles_answer, atol=ATOL, rtol=RTOL)

    hessenbergs, Q = va_orthogonalise(
        prob.domain.boundary_points.reshape(1200, 1),
        0,
        poles=prob.domain.poles,
    )
    basis, basis_deriv = va_evaluate(
        prob.domain.boundary_points, hessenbergs, poles=prob.domain.poles
    )
    # check the polynomial basis is the same
    assert np.allclose(
        basis[:, :25], basis_answer[:, :25], atol=ATOL, rtol=RTOL
    )
    assert np.allclose(
        basis_deriv[:, :25], basis_deriv_answer[:, :25], atol=ATOL, rtol=RTOL
    )
    # check all the basis functions are the same
    assert np.allclose(basis, basis_answer, atol=ATOL, rtol=RTOL)
    assert np.allclose(basis_deriv, basis_deriv_answer, atol=ATOL, rtol=RTOL)


def test_laurent_va_evaluate():
    """Test the va_evaluate with laurent against the MATLAB code."""
    from pylars.numerics import va_orthogonalise, va_evaluate
    from pylars import Problem
    import numpy as np
    from scipy.io import loadmat

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
    H_answer = test_answers["Hes"]
    # discard empty pole blocks
    hessenbergs_answer = [
        H_answer[:, k][0]
        for k in range(H_answer.shape[1])
        if H_answer[:, k][0].shape != (0, 0)
    ]
    Q_answer = test_answers["Q"]
    Z_answer = test_answers["Z"]
    basis_answer = test_answers["R0"]
    basis_deriv_answer = test_answers["R1"]
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
    assert np.allclose(
        prob.domain.boundary_points, Z_answer, atol=ATOL, rtol=RTOL
    )
    hessenbergs, Q = va_orthogonalise(
        prob.domain.boundary_points.reshape(-1, 1),
        deg_poly,
        interior_laurents=prob.domain.interior_laurents,
    )
    for hessenberg, hessenberg_answer in zip(hessenbergs, hessenbergs_answer):
        assert np.allclose(hessenberg, hessenberg_answer, atol=ATOL, rtol=RTOL)
    assert np.allclose(Q, Q_answer, atol=ATOL, rtol=RTOL)
    basis, basis_deriv = va_evaluate(
        prob.domain.boundary_points,
        hessenbergs,
        interior_laurents=prob.domain.interior_laurents,
    )
    # check all the basis functions are the same
    assert np.allclose(basis, basis_answer, atol=ATOL, rtol=RTOL)
    assert np.allclose(basis_deriv, basis_deriv_answer, atol=ATOL, rtol=RTOL)


if __name__ == "__main__":
    test_import_va_evaluate()
    test_small_va_evaluate()
    test_small_second_va_evaluate()
    test_small_poles_second_va_evaluate()
    test_small_laurent_second_va_evaluate()
    test_large_va_evaluate()
    test_small_poles_va_evaluate_1()
    test_small_poles_va_evaluate_2()
    test_large_poles_va_evaluate()
    test_laurent_va_evaluate()
