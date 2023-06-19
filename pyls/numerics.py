"""Numerical methods for solving the linear system."""
import numpy as np


def va_orthogonalise(Z, n):
    """Orthogonalise the series using the Vandermonde with Arnoldi method."""
    H = np.zeros((n + 1, n), dtype=complex)
    Q = np.zeros((len(Z), n + 1), dtype=complex)
    hessenbergs = [H]
    return hessenbergs, Q


def cart(z):
    """Convert from imaginary to cartesian coordinates."""
    return np.array([z.real, z.imag]).T
