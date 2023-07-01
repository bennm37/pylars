"""Numerical functions to create and evaluate the rational basis.

cart:
    Convert from imaginary to cartesian coordinates.
cluster:
    Generate the lighting pole spacing.
split:
    Split the coefficients for f and g.
make_function:
    Construct the function from the basis and coefficients.
va_orthogonalise:
    Create an orthogonal basis using the vandermonde
    with arnoldi method.
va_orthogonalise_jit:
    Create an orthogonal basis using the vandermonde
    with arnoldi method using numba.
va_evaluate:
    Evaluate the orthogonal basis at a new set of points.
"""
import numpy as np
from numba import njit
from numbers import Number


def cart(z):
    """Convert from imaginary to cartesian coordinates."""
    return np.array([z.real, z.imag]).T


def cluster(num_points, L, sigma):  # noqa: N803
    """Generate exponentially clustered pole spacing.

    Notes
    -----
    Uses the formula:
        L * exp(sigma * (sqrt(n) - sqrt(N)))
    from Brubeck and Trefethen (2023).
    """
    pole_spacing = L * np.exp(
        sigma * (np.sqrt(range(num_points, 0, -1)) - np.sqrt(num_points))
    )
    return pole_spacing


def split(coefficients):
    """Split the coefficients for f and g."""
    num_coeff = len(coefficients) // 4
    cf = (
        coefficients[:num_coeff]
        + 1j * coefficients[2 * num_coeff : 3 * num_coeff]
    )
    cg = (
        coefficients[num_coeff : 2 * num_coeff]
        + 1j * coefficients[3 * num_coeff :]
    )
    return cf, cg


def make_function(name, z, coefficients, hessenbergs, poles):
    """Construct the function from the hessenbergs and coefficients.

    Parameters
    ----------
    z : (M, 1) array_like
        The points at which to evaluate the function.
    coefficients : (4 * N, 1) array_like
        The coefficients of the goursat functions.
    hessenbergs : list of (N, N) array_like
        The upper Hessenberg matrices that define the rational basis.
    poles : list of lists of complex numbers
        The poles of the rational basis.

    Returns
    -------
    function : function
        A function that evaluates a quantity at a new set of points.

    Notes
    -----
    Changing coefficients will change the function.
    """
    z = np.array([z]).reshape(-1, 1)
    basis, basis_deriv = va_evaluate(z, hessenbergs, poles)
    cf, cg = split(coefficients)
    match name:
        case "f":
            return basis @ cf
        case "g":
            return basis @ cg
        case "psi":
            return np.imag(np.conj(z) * (basis @ cf) + basis @ cg)
        case "uv":
            return (
                z * np.conj(basis_deriv @ cf)
                - basis @ cf
                + np.conj(basis_deriv @ cg)
            )
        case "p":
            return np.real(4 * basis_deriv @ cf)
        case "omega":
            return np.imag(-4 * basis_deriv @ cf)


def va_orthogonalise(z, n, poles=None):
    """Orthogonalise the series using the Vandermonde with Arnoldi method.

    Parameters
    ----------
    z : (M, 1) array_like
        The points to evaluate the basis at.
    n : int
        The degree of the polynomial.
    poles : list of lists of complex numbers
        The poles of the rational basis.

    Returns
    -------
    Q: (M, ...) array_like
        The orthogonal basis.
    hessenbergs: list of (n+1, n) array_like
        The upper Hessenberg matrices. Note n is the degree of the polynomial
        or the number of poles in a pole group.

    Notes
    -----
    The matrix Q is made up of orthogonal matrices but
    is not itself orthogonal. Q spans the rational basis.
    """
    if z.shape[1] != 1:
        raise ValueError("z must be a column vector")
    m = len(z)
    H = np.zeros((n + 1, n), dtype=np.complex128)
    Q = np.zeros((m, n + 1), dtype=np.complex128)
    q = np.ones(len(z)).reshape(m, 1)
    Q[:, 0] = q.reshape(m)
    for k in range(n):
        q = z * Q[:, k].reshape(m, 1)
        for j in range(k + 1):
            H[j, k] = np.dot(Q[:, j].conj(), q)[0] / m
            q = q - H[j, k] * Q[:, j].reshape(m, 1)
        H[k + 1, k] = np.linalg.norm(q) / np.sqrt(m)
        Q[:, k + 1] = (q / H[k + 1, k]).reshape(m)
    hessenbergs = [H]
    if poles is not None:
        for pole_group in poles:
            num_poles = len(pole_group)
            Hp = np.zeros((num_poles + 1, num_poles), dtype=np.complex128)
            Qp = np.zeros((m, num_poles + 1), dtype=np.complex128)
            qp = np.ones(m).reshape(m, 1)
            Qp[:, 0] = qp.reshape(m)
            for k in range(num_poles):
                qp = Qp[:, k].reshape(m, 1) / (z - pole_group[k])
                for j in range(k + 1):
                    Hp[j, k] = np.dot(Qp[:, j].conj(), qp)[0] / m
                    qp = qp - Hp[j, k] * Qp[:, j].reshape(m, 1)
                Hp[k + 1, k] = np.linalg.norm(qp) / np.sqrt(m)
                Qp[:, k + 1] = (qp / Hp[k + 1, k]).reshape(m)
            hessenbergs += [Hp]
            Q = np.concatenate((Q, Qp[:, 1:]), axis=1)
    return hessenbergs, Q


@njit
def va_orthogonalise_jit(z, n):
    """Orthogonalise the series using the Vandermonde with Arnoldi method.

    See va_orthogonalise for more details. This function is JIT compiled.
    """
    if z.shape[1] != 1:
        raise ValueError("z must be a column vector")
    m = len(z)
    H = np.zeros((n + 1, n), dtype=np.complex128)
    Q = np.zeros((m, n + 1), dtype=np.complex128)
    q = np.ones(len(z)).reshape(m, 1)
    Q[:, 0] = q.reshape(m)
    for k in range(n):
        q = z * Q[:, k].copy().reshape(m, 1)
        for j in range(k + 1):
            H[j, k] = np.dot(Q[:, j].conj(), q)[0] / m
            q = q - H[j, k] * Q[:, j].copy().reshape(m, 1)
        H[k + 1, k] = np.linalg.norm(q) / np.sqrt(m)
        Q[:, k + 1] = (q / H[k + 1, k]).reshape(m)
    hessenbergs = [H]
    return hessenbergs, Q


def va_evaluate(z, hessenbergs, poles=None):
    """Construct the basis from the Hessenberg matrices.

    Parameters
    ----------
    z : (M, 1) array_like
        The points to evaluate the basis at.
    hessenbergs : list of (n+1, n) array_like
        The upper Hessenberg matrices. Note n is the degree of the polynomial
        or the number of poles in a pole group.
    poles : list of lists of complex numbers
        The poles of the rational basis.

    Returns
    -------
    R0 : (M, ...) array_like
        The rational basis.
    R1 : (M, ...) array_like
        The derivative of the rational basis.
    """
    H = hessenbergs[0]
    n = H.shape[1]
    if isinstance(z, np.ndarray):
        m = len(z)
        if m != z.shape[0]:
            raise ValueError("z must be a column vector")
        z = z.reshape(-1, 1)
    elif isinstance(z, Number):
        m = 1
        z = np.array([z]).reshape(1, 1)
    else:
        raise TypeError("z must be a numpy array or a number")
    Q = np.zeros((m, n + 1), dtype=np.complex128)
    D = np.zeros((m, n + 1), dtype=np.complex128)
    Q[:, 0] = np.ones(m)
    for k in range(n):
        hkk = H[k + 1, k]
        Q[:, k + 1] = (
            (
                z * Q[:, k].reshape(m, 1)
                - Q[:, : k + 1].reshape(m, k + 1)
                @ H[: k + 1, k].reshape(k + 1, 1)
            )
            / hkk
        ).reshape(m)
        D[:, k + 1] = (
            (
                z * D[:, k].reshape(m, 1)
                - D[:, : k + 1].reshape(m, k + 1)
                @ H[: k + 1, k].reshape(k + 1, 1)
                + Q[:, k].reshape(m, 1)
            )
            / hkk
        ).reshape(m)
    if poles is not None:
        for i, pole_group in enumerate(poles):
            num_poles = len(pole_group)
            Hp = hessenbergs[i + 1]
            Qp = np.zeros((m, num_poles + 1), dtype=np.complex128)
            Dp = np.zeros((m, num_poles + 1), dtype=np.complex128)
            Qp[:, 0] = np.ones(m)
            for k in range(num_poles):
                hkk = Hp[k + 1, k]
                z_pole = 1 / (z - pole_group[k])
                Qp[:, k + 1] = (
                    (
                        z_pole * Qp[:, k].reshape(m, 1)
                        - Qp[:, : k + 1].reshape(m, k + 1)
                        @ Hp[: k + 1, k].reshape(k + 1, 1)
                    )
                    / hkk
                ).reshape(m)
                Dp[:, k + 1] = (
                    (
                        z_pole * Dp[:, k].reshape(m, 1)
                        - Dp[:, : k + 1].reshape(m, k + 1)
                        @ Hp[: k + 1, k].reshape(k + 1, 1)
                        - z_pole**2 * Qp[:, k].reshape(m, 1)
                    )
                    / hkk
                ).reshape(m)
            Q = np.concatenate((Q, Qp[:, 1:]), axis=1)
            D = np.concatenate((D, Dp[:, 1:]), axis=1)
    return Q, D
