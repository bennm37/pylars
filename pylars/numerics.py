"""Numerical methods for solving the linear system."""
import numpy as np
from numbers import Number


def cart(z):
    """Convert from imaginary to cartesian coordinates."""
    return np.array([z.real, z.imag]).T


def cluster(num_points, length_scale, sigma):
    """Generate exponentially clustered pole spacing."""
    pole_spacing = length_scale * np.exp(
        sigma * (np.sqrt(range(num_points, 0, -1)) - np.sqrt(num_points))
    )
    return pole_spacing


def split(coefficients):
    """Split the coefficients for f and g."""
    # TODO check this is consitent with the dependent matricies.
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
    """Make a function with the given name."""
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


def va_orthogonalise(z, n, poles=None, laurents=None):
    """Orthogonalise the series using the Vandermonde with Arnoldi method.

    The matrix Q has orthogonal columns of norm sqrt(m) so that the elements
    are of order 1. The matrix H is upper Hessenberg and the columns of Q
    span the same space as the columns of the Vandermonde matrix.
    """
    if z.shape[1] != 1:
        raise ValueError("Z must be a column vector")
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
    if laurents is not None:
        for laurent_series in laurents:
            deg_laurent = laurent_series[1]
            Hp = np.zeros((deg_laurent + 1, deg_laurent), dtype=np.complex128)
            Qp = np.zeros((m, deg_laurent + 1), dtype=np.complex128)
            qp = np.ones(m).reshape(m, 1)
            Qp[:, 0] = qp.reshape(m)
            for k in range(deg_laurent):
                qp = Qp[:, k].reshape(m, 1) / (z - laurent_series[0])
                for j in range(k + 1):
                    Hp[j, k] = np.dot(Qp[:, j].conj(), qp)[0] / m
                    qp = qp - Hp[j, k] * Qp[:, j].reshape(m, 1)
                Hp[k + 1, k] = np.linalg.norm(qp) / np.sqrt(m)
                Qp[:, k + 1] = (qp / Hp[k + 1, k]).reshape(m)
            hessenbergs += [Hp]
            Q = np.concatenate((Q, Qp[:, 1:]), axis=1)
    return hessenbergs, Q


def va_evaluate(z, hessenbergs, poles=None):
    """Construct the basis and its derivatives."""
    H = hessenbergs[0]
    n = H.shape[1]
    if isinstance(z, np.ndarray):
        m = len(z)
        if m != z.shape[0]:
            raise ValueError("Z must be a column vector")
        z = z.reshape(-1, 1)
    elif isinstance(z, Number):
        m = 1
        z = np.array([z]).reshape(1, 1)
    else:
        raise TypeError("Z must be a numpy array or a number")
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
                Z_pole = 1 / (z - pole_group[k])
                Qp[:, k + 1] = (
                    (
                        Z_pole * Qp[:, k].reshape(m, 1)
                        - Qp[:, : k + 1].reshape(m, k + 1)
                        @ Hp[: k + 1, k].reshape(k + 1, 1)
                    )
                    / hkk
                ).reshape(m)
                Dp[:, k + 1] = (
                    (
                        Z_pole * Dp[:, k].reshape(m, 1)
                        - Dp[:, : k + 1].reshape(m, k + 1)
                        @ Hp[: k + 1, k].reshape(k + 1, 1)
                        - Z_pole**2 * Qp[:, k].reshape(m, 1)
                    )
                    / hkk
                ).reshape(m)
            Q = np.concatenate((Q, Qp[:, 1:]), axis=1)
            D = np.concatenate((D, Dp[:, 1:]), axis=1)
    return Q, D
