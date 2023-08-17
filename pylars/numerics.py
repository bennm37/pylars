"""Numerical methods for solving the linear system."""
import numpy as np
from numbers import Number
from scipy.sparse import diags
from scipy.linalg import svd, eig


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
    num_coeff = len(coefficients) // 4
    complex_coeff = (
        coefficients[: 2 * num_coeff] + 1j * coefficients[2 * num_coeff :]
    )
    cf = complex_coeff[:num_coeff]
    cg = complex_coeff[num_coeff : 2 * num_coeff]
    return cf, cg


def split_laurent(coefficients, interior_laurents):
    """Split the coefficients for f and g for laurent series."""
    coefficients = np.array(coefficients)
    num_logs = len(interior_laurents)
    num_coeff = (len(coefficients) - 4 * num_logs) // 4
    complex_coeff = (
        coefficients[: len(coefficients) // 2]
        + 1j * coefficients[len(coefficients) // 2 :]
    )
    cf = complex_coeff[:num_coeff]
    cg = complex_coeff[num_coeff : 2 * num_coeff]
    clf = complex_coeff[2 * num_coeff : 2 * num_coeff + num_logs]
    clg = complex_coeff[2 * num_coeff + num_logs :]
    return cf, cg, clf, clg


def make_function(
    name,
    z,
    coefficients,
    hessenbergs,
    poles=None,
    interior_laurents=None,
    exterior_laurents=None,
):
    """Make a function with the given name."""
    z = np.array([z]).reshape(-1, 1)
    if name == "eij":
        basis, basis_deriv, basis_deriv_2 = va_evaluate(
            z,
            hessenbergs,
            poles,
            interior_laurents,
            exterior_laurents,
            second_deriv=True,
        )
    else:
        basis, basis_deriv = va_evaluate(
            z, hessenbergs, poles, interior_laurents, exterior_laurents
        )
    if not interior_laurents:
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
            case "eij":
                result = z * np.conj(basis_deriv_2 @ cf) + np.conj(
                    basis_deriv_2 @ cg
                )
                eij = np.array(
                    [[result.real, result.imag], [result.imag, -result.real]]
                )
                return np.moveaxis(eij, 2, 0)
    else:
        cf, cg, clf, clg = split_laurent(coefficients, interior_laurents)
        centers = np.array(
            [laurent_series[0] for laurent_series in interior_laurents]
        ).reshape(1, -1)
        match name:
            case "f":
                return basis @ cf + np.log(z - centers) @ clf
            case "g":
                return (
                    basis @ cg
                    + np.log(z - centers) @ clg
                    - ((z - centers) * np.log(z - centers) - z) @ np.conj(clf)
                )
            case "psi":
                result = np.imag(
                    np.conj(z) * (basis @ cf) + basis @ cg
                ) + np.imag(
                    np.conj(z) * (np.log(z - centers) @ clf)
                    + np.log(z - centers) @ clg
                    - ((z - centers) * np.log(z - centers) - z) @ np.conj(clf)
                )
                return result
            case "uv":
                return (
                    z * np.conj(basis_deriv @ cf)
                    - basis @ cf
                    + np.conj(basis_deriv @ cg)
                    + z * np.conj((1 / (z - centers)) @ clf)
                    - np.log(z - centers) @ clf
                    + np.conj(
                        (1 / (z - centers)) @ clg
                        - np.log(z - centers) @ np.conj(clf)
                    )
                )
            case "p":
                return np.real(4 * basis_deriv @ cf) + np.real(
                    (4 / (z - centers)) @ clf
                )
            case "omega":
                return np.imag(-4 * basis_deriv @ cf) + np.imag(
                    (-4 / (z - centers)) @ clf
                )
            case "eij":
                result = (
                    z * np.conj(basis_deriv_2 @ cf)
                    + z * np.conj((-1 / (z - centers) ** 2) @ clf)
                    + np.conj(basis_deriv_2 @ cg)
                    + np.conj(
                        (-1 / (z - centers) ** 2) @ clg
                        + (-1 / (z - centers)) @ np.conj(clf)
                    )
                )
                eij = np.array(
                    [[result.real, result.imag], [result.imag, -result.real]]
                )
                return np.moveaxis(eij, 2, 0)
                # result = (
                #     z * np.conj(basis_deriv_2 @ cf)
                #     + np.conj(basis_deriv_2 @ cg)
                #     + z * np.conj((-1 / (z - centers) ** 2) @ clf)
                #     + np.conj((-1 / (z - centers) ** 2) @ clg)
                # )
                # eij = np.array(
                #     [[result.real, result.imag], [result.imag, -result.real]]
                # )
                # return np.moveaxis(eij, 2, 0)


def va_orthogonalise(
    z, n, poles=None, interior_laurents=None, exterior_laurents=None
):
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
    laurents = []
    if interior_laurents is not None:
        laurents += interior_laurents
    if exterior_laurents is not None:
        laurents += exterior_laurents
    if len(laurents) > 0:
        for laurent_series in laurents:
            deg_laurent = laurent_series[1]
            Hl = np.zeros((deg_laurent + 1, deg_laurent), dtype=np.complex128)
            Qp = np.zeros((m, deg_laurent + 1), dtype=np.complex128)
            qp = np.ones(m).reshape(m, 1)
            Qp[:, 0] = qp.reshape(m)
            for k in range(deg_laurent):
                qp = Qp[:, k].reshape(m, 1) / (z - laurent_series[0])
                for j in range(k + 1):
                    Hl[j, k] = np.dot(Qp[:, j].conj(), qp)[0] / m
                    qp = qp - Hl[j, k] * Qp[:, j].reshape(m, 1)
                Hl[k + 1, k] = np.linalg.norm(qp) / np.sqrt(m)
                Qp[:, k + 1] = (qp / Hl[k + 1, k]).reshape(m)
            hessenbergs += [Hl]
            Q = np.concatenate((Q, Qp[:, 1:]), axis=1)
    return hessenbergs, Q


def va_evaluate(
    z,
    hessenbergs,
    poles=None,
    interior_laurents=None,
    exterior_laurents=None,
    second_deriv=False,
):
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
    if second_deriv:
        D2 = np.zeros((m, n + 1), dtype=np.complex128)
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
        if second_deriv:
            D2[:, k + 1] = (
                (
                    z * D2[:, k].reshape(m, 1)
                    - D2[:, : k + 1].reshape(m, k + 1)
                    @ H[: k + 1, k].reshape(k + 1, 1)
                    + 2 * D[:, k].reshape(m, 1)
                )
                / hkk
            ).reshape(m)
    if poles is not None:
        for i, pole_group in enumerate(poles):
            num_poles = len(pole_group)
            Hp = hessenbergs[i + 1]
            Qp = np.zeros((m, num_poles + 1), dtype=np.complex128)
            Dp = np.zeros((m, num_poles + 1), dtype=np.complex128)
            if second_deriv:
                Dp2 = np.zeros((m, num_poles + 1), dtype=np.complex128)
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
                        - z_pole**2 * Qp[:, k].reshape(m, 1)
                        - Dp[:, : k + 1].reshape(m, k + 1)
                        @ Hp[: k + 1, k].reshape(k + 1, 1)
                    )
                    / hkk
                ).reshape(m)
                if second_deriv:
                    Dp2[:, k + 1] = (
                        (
                            z_pole * Dp2[:, k].reshape(m, 1)
                            - 2 * z_pole**2 * Dp[:, k].reshape(m, 1)
                            + 2 * z_pole**3 * Qp[:, k].reshape(m, 1)
                            - Dp2[:, : k + 1].reshape(m, k + 1)
                            @ Hp[: k + 1, k].reshape(k + 1, 1)
                        )
                        / hkk
                    ).reshape(m)
            Q = np.concatenate((Q, Qp[:, 1:]), axis=1)
            D = np.concatenate((D, Dp[:, 1:]), axis=1)
            if second_deriv:
                D2 = np.concatenate((D2, Dp2[:, 1:]), axis=1)
    laurents = []
    if interior_laurents is not None:
        laurents += interior_laurents
    if exterior_laurents is not None:
        laurents += exterior_laurents
    if len(laurents) > 0:
        for i, laurent_series in enumerate(laurents):
            deg_laurent = laurent_series[1]
            if poles is not None:
                Hl = hessenbergs[len(poles) + i + 1]
            else:
                Hl = hessenbergs[i + 1]
            Ql = np.zeros((m, deg_laurent + 1), dtype=np.complex128)
            Dl = np.zeros((m, deg_laurent + 1), dtype=np.complex128)
            if second_deriv:
                Dl2 = np.zeros((m, deg_laurent + 1), dtype=np.complex128)
            Ql[:, 0] = np.ones(m)
            one_over_z = 1 / (z - laurent_series[0])
            one_over_z2 = one_over_z**2
            one_over_z3 = one_over_z**3
            for k in range(deg_laurent):
                hkk = Hl[k + 1, k]
                Ql[:, k + 1] = (
                    (
                        one_over_z * Ql[:, k].reshape(m, 1)
                        - Ql[:, : k + 1].reshape(m, k + 1)
                        @ Hl[: k + 1, k].reshape(k + 1, 1)
                    )
                    / hkk
                ).reshape(m)
                Dl[:, k + 1] = (
                    (
                        one_over_z * Dl[:, k].reshape(m, 1)
                        - one_over_z2 * Ql[:, k].reshape(m, 1)
                        - Dl[:, : k + 1].reshape(m, k + 1)
                        @ Hl[: k + 1, k].reshape(k + 1, 1)
                    )
                    / hkk
                ).reshape(m)
                if second_deriv:
                    Dl2[:, k + 1] = (
                        (
                            one_over_z * Dl2[:, k].reshape(m, 1)
                            - 2 * one_over_z2 * Dl[:, k].reshape(m, 1)
                            + 2 * one_over_z3 * Ql[:, k].reshape(m, 1)
                            - Dl2[:, : k + 1].reshape(m, k + 1)
                            @ Hl[: k + 1, k].reshape(k + 1, 1)
                        )
                        / hkk
                    ).reshape(m)
            Q = np.concatenate((Q, Ql[:, 1:]), axis=1)
            D = np.concatenate((D, Dl[:, 1:]), axis=1)
            if second_deriv:
                D2 = np.concatenate((D2, Dl2[:, 1:]), axis=1)
    if second_deriv:
        return Q, D, D2
    return Q, D


def aaa(F, Z, tol=1e-13, mmax=100):
    """Use the AAA algorithm to compute a rational approximation of f(z)."""
    M = len(Z)
    if hasattr(F, "__call__"):
        F = F(Z)
    Z = np.array(Z).reshape(-1, 1)
    M = len(Z)
    F = np.array(F).reshape(-1, 1)
    SF = diags(F.flatten())
    J = np.array(range(M))
    z = np.empty((0, 1))
    f = np.empty((0, 1))
    C = np.empty((M, 0))
    errvec = np.empty((0, 1))
    R = np.mean(F)
    for M in range(mmax - 1):
        j = np.argmax(np.abs(F - R))  # select next support point
        z = np.vstack([z, Z[j]])  # update support points, data values'
        f = np.vstack([f, F[j]])  # update support points, data values
        J = np.delete(J, np.where(J == j))  # update index vector
        C = np.hstack([C, 1 / (Z - Z[j])])  # next column of Cauchy matrix
        Sf = diags(f.flatten())  # right scaling matrix
        A = SF @ C - C @ Sf  # Loewner matrix
        _, _, V = svd(A[J, :])  # SVD
        V = np.conj(V.T)
        w = V[:, M].reshape(-1, 1)  # weight vector = min sing vector
        N = C @ (w * f)
        D = C @ w  # numerator and denominator
        R = F.copy()
        R[J] = N[J] / D[J]  # rational approximation
        err = np.max(np.abs(F - R))
        errvec = np.vstack([errvec, err])  # max error at sample points
        if err <= np.abs(tol * np.max(F)):
            break  # stop if converged

    def r(zz):
        return rhandle(zz, z, f, w)

    poles, residues, zeros = prz(r, z, f, w)
    return r, poles, residues, zeros


def prz(r, z, f, w):
    """Compute poles, residues, zeros for aaa."""
    M = len(w)
    B = np.eye(M + 1)
    B[0, 0] = 0
    E = np.hstack([np.ones((M, 1)), np.diag(z.flatten())])
    E = np.vstack([np.append(0, w.T), E])
    pol = eig(E, B)[0]
    pol = pol[~np.isinf(pol)]  # poles
    dz = 1e-5 * np.exp(2j * np.pi * np.arange(1, 5) / 4)
    res = r(pol[:, None] + dz) @ dz.T / 4  # residues
    E = np.hstack([np.ones((M, 1)), np.diag(z.flatten())])
    E = np.vstack([np.append(0, (w * f).T), E])
    zer = eig(E, B)[0]
    zer = zer[~np.isinf(zer)]  # zeros
    return pol, res, zer


def rhandle(zz, z, f, w):
    """Create the function for r."""
    zv = zz.reshape(-1, 1)
    CC = 1 / (zv - z.T)
    r = (CC @ (w * f)) / (CC @ w)
    ii = np.where(np.isnan(r))
    for i in ii:
        r[i] = f[np.where(z[i] == z)]
    return r.reshape(zz.shape)
