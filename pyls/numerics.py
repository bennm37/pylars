"""Numerical methods for solving the linear system."""
import numpy as np


def cart(z):
    """Convert from imaginary to cartesian coordinates."""
    return np.array([z.real, z.imag]).T


def cluster(num_points, L, sigma):
    """Generate exponentially clustered pole spacing."""
    pole_spacing = L * np.exp(
        sigma * (np.sqrt(range(num_points, 0, -1)) - np.sqrt(num_points))
    )
    return pole_spacing


def va_orthogonalise(Z, n):
    """Orthogonalise the series using the Vandermonde with Arnoldi method.

    The matrix Q has orthogonal columns of norm sqrt(m) so that the elements
    are of order 1. The matrix H is upper Hessenberg and the columns of Q
    span the same space as the columns of the Vandermonde matrix.
    """
    if Z.shape[1] != 1:
        raise ValueError("Z must be a column vector")
    m = len(Z)
    H = np.zeros((n + 1, n), dtype=complex)
    Q = np.zeros((m, n + 1), dtype=complex)
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
    return hessenbergs, Q
