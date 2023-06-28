"""Test the va_orthogonalise function.

The va_orthogonalise function is used to orthogonalise a series of
functions using the Vandermonde with Arnoldi method. This test suite
checks that the function can be imported and that it works for a
small simple analytic example and a large example generated by the
MATLAB code. The MATLAB results are included in the tests/data
directory.
"""
from test_settings import ATOL, RTOL


def test_import_va_orthogonalise():
    """Test that the VA_orthogonalise function can be imported."""
    from pyls.numerics import va_orthogonalise

    assert va_orthogonalise is not None


def test_simple_va_orthogonalise():
    """Test against a simple VA_orthogonalise example."""
    from pyls.numerics import va_orthogonalise
    import numpy as np
    from numpy import sqrt

    Z = np.array([0, 2, 1 + 1j]).reshape(3, 1)
    n = 1
    H_answer = np.array([1 + 1j / 3, 2 * sqrt(2) / 3]).reshape(2, 1)
    Q_answer = np.array(
        [
            [1, 1, 1],
            [
                -3 / (2 * sqrt(2)) - 1j / (2 * sqrt(2)),
                3 / (2 * sqrt(2)) - 1j / (2 * sqrt(2)),
                1j / sqrt(2),
            ],
        ]
    ).T
    hessenbergs, Q = va_orthogonalise(Z, n)
    H = hessenbergs[0]
    assert np.allclose(H.real, H_answer.real, atol=ATOL, rtol=RTOL)
    assert np.allclose(H.imag, H_answer.imag, atol=ATOL, rtol=RTOL)
    assert np.allclose(Q.real, Q_answer.real, atol=ATOL, rtol=RTOL)
    assert np.allclose(Q.imag, Q_answer.imag, atol=ATOL, rtol=RTOL)
    assert np.allclose((Q @ H).real, Z.real, atol=ATOL, rtol=RTOL)
    assert np.allclose((Q @ H).imag, Z.imag, atol=ATOL, rtol=RTOL)


def test_large_va_orthongalise():
    """Test against a large polynomial example generated by the MATLAB code."""
    from pyls.numerics import va_orthogonalise
    import numpy as np
    from scipy.io import loadmat

    Z = np.exp(1j * np.linspace(0, 2 * np.pi, 100)).reshape(100, 1)
    H_answer = loadmat("tests/data/VAorthog_circle_H.mat")["H"]
    Q_answer = loadmat("tests/data/VAorthog_circle_Q.mat")["Q"]
    # check matlab input is correct using arnoldi recurrence relation
    for k in range(1, 10):
        left = np.diag(Z.flatten()) @ Q_answer[:, :k]
        right = Q_answer[:, : k + 1] @ H_answer[: k + 1, :k]
        assert np.allclose(left.real, right.real, atol=ATOL, rtol=RTOL)
        assert np.allclose(left.imag, right.imag, atol=ATOL, rtol=RTOL)
    Z = np.exp(1j * np.linspace(0, 2 * np.pi, 100)).reshape(100, 1)
    hessenbergs, Q = va_orthogonalise(Z, 10)
    H = hessenbergs[0]
    assert np.allclose(H.real, H_answer.real, atol=ATOL, rtol=RTOL)
    assert np.allclose(H.imag, H_answer.imag, atol=ATOL, rtol=RTOL)
    assert np.allclose(Q.real, Q_answer.real, atol=ATOL, rtol=RTOL)
    assert np.allclose(Q.imag, Q_answer.imag, atol=ATOL, rtol=RTOL)
    for k in range(1, 10):
        left = np.diag(Z.flatten()) @ Q[:, :k]
        right = Q[:, : k + 1] @ H[: k + 1, :k]
        assert np.allclose(left.real, right.real, atol=ATOL, rtol=RTOL)
        assert np.allclose(left.imag, right.imag, atol=ATOL, rtol=RTOL)
    # check Q has orthogonal columns with norm M
    # an exception for ATOL is made here as Q will only be
    # approximately orthogonal
    assert np.allclose(
        (Q.conj().T @ Q), len(Z) * np.eye(11), atol=1e-10, rtol=RTOL
    )


def test_poles_va_orthogonalise():
    """Test the va_orthogonalise with poles against the MATLAB code."""
    from pyls.numerics import va_orthogonalise
    from pyls import Domain
    import numpy as np
    from scipy.io import loadmat

    n, num_poles = 24, 24
    test_answers = loadmat(
        f"tests/data/lid_driven_cavity_n_{n}_np_{num_poles}.mat"
    )
    H_answer = test_answers["Hes"]
    hessenbergs_answer = [H_answer[:, k][0] for k in range(H_answer.shape[1])]
    Q_answer = test_answers["Q"]
    Z_answer = test_answers["Z"]
    poles_answer = test_answers["Pol"]
    poles_answer = np.array([poles_answer[0, i] for i in range(4)]).reshape(
        4, 24
    )
    corners = [1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j]
    dom = Domain(
        corners, num_poles=24, num_boundary_points=300, L=np.sqrt(2) * 1.5
    )
    # check the MATALB domain points and poles are the same
    assert np.allclose(
        dom.boundary_points.real, Z_answer.real, atol=ATOL, rtol=RTOL
    )
    assert np.allclose(
        dom.boundary_points.imag, Z_answer.imag, atol=ATOL, rtol=RTOL
    )
    assert np.allclose(dom.poles.real, poles_answer.real, atol=ATOL, rtol=RTOL)
    assert np.allclose(dom.poles.imag, poles_answer.imag, atol=ATOL, rtol=RTOL)
    hessenbergs, Q = va_orthogonalise(
        dom.boundary_points.reshape(1200, 1), 24, poles=dom.poles
    )
    for hessenberg, hessenberg_answer in zip(hessenbergs, hessenbergs_answer):
        assert np.allclose(
            hessenberg.real, hessenberg_answer.real, atol=ATOL, rtol=RTOL
        )
    assert np.allclose(Q.real, Q_answer.real, atol=ATOL, rtol=RTOL)
    assert np.allclose(Q.imag, Q_answer.imag, atol=ATOL, rtol=RTOL)


def test_va_orthogonalise_debug():
    """Test the va_orthogonalise with poles against the MATLAB code."""
    from pyls.numerics import va_orthogonalise
    from pyls import Domain
    import numpy as np
    from scipy.io import loadmat

    n, num_poles = 24, 24
    test_answers = loadmat(
        f"tests/data/lid_driven_cavity_n_{n}_np_{num_poles}.mat"
    )
    Z_answer = test_answers["Z"]
    poles_answer = test_answers["Pol"]
    poles_answer = np.array([poles_answer[0, i] for i in range(4)]).reshape(
        4, 24
    )
    corners = [1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j]
    dom = Domain(
        corners, num_poles=24, num_boundary_points=300, L=np.sqrt(2) * 1.5
    )
    assert np.allclose(
        dom.boundary_points.real, Z_answer.real, atol=ATOL, rtol=RTOL
    )
    assert np.allclose(
        dom.boundary_points.imag, Z_answer.imag, atol=ATOL, rtol=RTOL
    )
    assert np.allclose(dom.poles.real, poles_answer.real, atol=ATOL, rtol=RTOL)
    assert np.allclose(dom.poles.imag, poles_answer.imag, atol=ATOL, rtol=RTOL)
    hessenbergs, Q = va_orthogonalise_debug(
        dom.boundary_points.reshape(1200, 1), 24, poles=dom.poles
    )


def test_va_orthogonalise_jit():
    """Run the jit test works."""
    from pyls.numerics import va_orthogonalise, va_orthogonalise_jit
    import numpy as np

    Z = np.exp(1j * np.linspace(0, 2 * np.pi, 100)).reshape(100, 1)
    hessenbergs_jit, Q_jit = va_orthogonalise_jit(Z, 10)
    hessenbergs, Q = va_orthogonalise(Z, 10)
    for hessenberg_jit, hessenberg in zip(hessenbergs_jit, hessenbergs):
        assert np.allclose(
            hessenberg_jit.real, hessenberg.real, atol=ATOL, rtol=RTOL
        )
    assert np.allclose(Q_jit.real, Q.real, atol=ATOL, rtol=RTOL)
    assert np.allclose(Q_jit.imag, Q.imag, atol=ATOL, rtol=RTOL)


def va_orthogonalise_debug(Z, n, poles=None):
    """Orthogonalise the series using the Vandermonde with Arnoldi method.

    The matrix Q has orthogonal columns of norm sqrt(m) so that the elements
    are of order 1. The matrix H is upper Hessenberg and the columns of Q
    span the same space as the columns of the Vandermonde matrix.
    """
    import numpy as np
    from scipy.io import loadmat

    if Z.shape[1] != 1:
        raise ValueError("Z must be a column vector")
    m = len(Z)
    n, num_poles = 24, 24
    test_answers = loadmat("tests/data/lid_driven_cavity_n_24_np_24.mat")
    Hes_answer = test_answers["Hes"]
    H_answers_list = [Hes_answer[:, k][0] for k in range(Hes_answer.shape[1])]
    Q_answer = test_answers["Q"]
    Q_answers_list = [
        Q_answer[:, 24 * i + 1 : 24 * (i + 1) + 1] for i in range(5)
    ]
    Q_answers_list = [
        np.append(Q_answer[:, 0].reshape(-1, 1), Q, axis=1)
        for Q in Q_answers_list
    ]
    H = np.zeros((n + 1, n), dtype=np.complex128)
    Q = np.zeros((m, n + 1), dtype=np.complex128)
    q = np.ones(len(Z)).reshape(m, 1)
    Q[:, 0] = q.reshape(m)
    for k in range(n):
        q = Z * Q[:, k].reshape(m, 1)
        for j in range(k + 1):
            H[j, k] = np.dot(Q[:, j].conj(), q)[0] / m
            q = q - H[j, k] * Q[:, j].reshape(m, 1)
        H[k + 1, k] = np.linalg.norm(q) / np.sqrt(m)
        Q[:, k + 1] = (q / H[k + 1, k]).reshape(m)
    hessenbergs = [H]
    if poles is not None:
        for i, pole_group in enumerate(poles):
            num_poles = len(pole_group)
            Hp = np.zeros((num_poles + 1, num_poles), dtype=np.complex128)
            Qp = np.zeros((m, num_poles + 1), dtype=np.complex128)
            qp = np.ones(m).reshape(m, 1)
            Qp[:, 0] = qp.reshape(m)
            for k in range(num_poles):
                qp = Qp[:, k].reshape(m, 1) / (Z - pole_group[k])
                pole_answer = loadmat(
                    f"tests/data/VAorthog_debug/hes_{i+1}/pol_k_{k+1}.mat"
                )["pol_k"]
                assert np.isclose(
                    pole_group[k].real, pole_answer.real, atol=ATOL, rtol=RTOL
                )
                assert np.isclose(
                    pole_group[k].imag, pole_answer.imag, atol=ATOL, rtol=RTOL
                )
                for j in range(k + 1):
                    qp_answer = loadmat(
                        f"tests/data/VAorthog_debug/hes_{i+1}/q_k_{k+1}_j_{j+1}.mat"
                    )["q"]
                    assert np.allclose(
                        qp.real, qp_answer.real, atol=1e-8, rtol=1
                    )
                    assert np.allclose(
                        qp.imag, qp_answer.imag, atol=1e-8, rtol=1
                    )
                    Hp[j, k] = np.dot(Qp[:, j].conj(), qp)[0] / m
                    qp = qp - Hp[j, k] * Qp[:, j].reshape(m, 1)
                Hp[k + 1, k] = np.linalg.norm(qp) / np.sqrt(m)
                Qp[:, k + 1] = (qp / Hp[k + 1, k]).reshape(m)
            hessenbergs += [Hp]
            Q = np.concatenate((Q, Qp[:, 1:]), axis=1)
    return hessenbergs, Q


if __name__ == "__main__":
    test_import_va_orthogonalise()
    test_simple_va_orthogonalise()
    test_large_va_orthongalise()
    # test_poles_va_orthogonalise()
    # test_va_orthogonalise_jit()
    # test_va_orthogonalise_debug()
