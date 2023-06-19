"""Test the va_orthogonalise function.

The va_orthogonalise function is used to orthogonalise a series of
functions using the Vandermonde with Arnoldi method. This test suite
checks that the function can be imported and that it works for a
small simple analytic example and a large example generated by the
MATLAB code. The MATLAB results are included in the tests/data
directory.
"""


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
    assert np.allclose(H, H_answer)
    assert np.allclose(Q, Q_answer)
    assert np.allclose(Q @ H, Z, atol=1e-10)


def test_large_va_orthongalise():
    """Test against a large example generated by the MATLAB code."""
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
        assert np.allclose(left, right, atol=1e-10)
    Z = np.exp(1j * np.linspace(0, 2 * np.pi, 100)).reshape(100, 1)
    hessenbergs, Q = va_orthogonalise(Z, 10)
    H = hessenbergs[0]
    assert np.allclose(H, H_answer)
    assert np.allclose(Q, Q_answer)
    for k in range(1, 10):
        left = np.diag(Z.flatten()) @ Q[:, :k]
        right = Q[:, : k + 1] @ H[: k + 1, :k]
        assert np.allclose(left, right, atol=1e-10)
    # check Q has orthogonal columns with norm M
    assert np.allclose(Q.conj().T @ Q, len(Z) * np.eye(11), atol=1e-10)


def test_va_orthogonalise_jit():
    """Run the jit test works."""
    from pyls.numerics import va_orthogonalise, va_orthogonalise_jit
    import numpy as np

    Z = np.exp(1j * np.linspace(0, 2 * np.pi, 100)).reshape(100, 1)
    hessenbergs_jit, Q_jit = va_orthogonalise_jit(Z, 10)
    hessenbergs, Q = va_orthogonalise(Z, 10)
    assert np.allclose(hessenbergs_jit, hessenbergs)
    assert np.allclose(Q_jit, Q)


if __name__ == "__main__":
    test_import_va_orthogonalise()
    test_simple_va_orthogonalise()
    test_large_va_orthongalise()
    test_va_orthogonalise_jit()
