"""Solution class for pylars."""
from numbers import Number
import numpy as np
from scipy.integrate import quad

# import pickle as pkl


class Solution:
    """Solution class to store the solution to a problem."""

    def __init__(
        self,
        problem,
        psi,
        uv,
        p,
        omega,
        eij,
        L=1,  # noqa N803
        U=1,
        mu=1,
        status="nd",
        max_residual=None,
    ):
        self.problem = problem
        self.psi = psi
        self.uv = uv
        self.p = p
        self.omega = omega
        self.eij = eij
        self.functions = [psi, uv, p, omega, eij]
        self.max_residual = max_residual
        self.L = L
        self.U = U
        self.mu = mu
        self.status = status

    def stress_discrete(self, z, dx=1e-6):
        """Calculate the total stress using finite differences."""
        z = np.array(z)
        zshape = z.shape
        z = z.reshape(-1)
        pressures = self.p(z)
        isotropic = np.array([-np.eye(2) * p for p in pressures]).reshape(
            zshape + (2, 2)
        )
        # caluclate velocity gradients using central difference for interior
        # points
        # TODO handle boundary points
        U_x = (self.uv(z + dx) - self.uv(z - dx)) / (2 * dx)
        u_x, v_x = U_x.real, U_x.imag
        U_y = (self.uv(z + dx * 1j) - self.uv(z - dx * 1j)) / (2 * dx)
        u_y, v_y = U_y.real, U_y.imag
        deviatoric = np.array([[2 * u_x, u_y + v_x], [u_y + v_x, 2 * v_y]])
        deviatoric = np.moveaxis(deviatoric, (2), (0)).reshape(zshape + (2, 2))
        deviatoric *= self.mu
        stress = isotropic + deviatoric
        return stress

    def stress_goursat(self, z):
        """Calculate the total stress using finite differences."""
        z = np.array(z)
        zshape = z.shape
        z = z.reshape(-1)
        pressures = self.p(z)
        isotropic = np.array([-np.eye(2) * p for p in pressures]).reshape(
            zshape + (2, 2)
        )
        # TODO this is a bit of a mess.
        # eij and p are dimensionalized by dimensionalize, but stress isn't.
        deviatoric = 2 * self.mu * self.eij(z).reshape(zshape + (2, 2))
        stress = isotropic + deviatoric
        return stress

    def force(self, curve, deriv):
        """Calculate the force on exerted by the fluid on a curve.

        The curve should be positively oriented.
        """

        def integrand(s):
            stress = self.stress_goursat(curve(s))
            normal = -1j * deriv(s)  # outward facing normal
            norm = np.array([normal.real, normal.imag])
            result = norm @ stress
            return result[0] + 1j * result[1]

        return quad(integrand, 0, 1, complex_func=True)[0]

    def torque(self, curve, deriv, centroid):
        """Calculate the torque exerted by the fluid on a curve.

        The curve should be positively oriented.
        """

        def integrand(s):
            z = curve(s)
            stress = self.stress_goursat(z)
            normal = -1j * deriv(s)  # outward facing normal
            norm = np.array([normal.real, normal.imag])
            result = norm @ stress
            stress_vec = result[0] + 1j * result[1]
            torque = np.conj(z - centroid) * stress_vec
            return torque.imag

        return quad(integrand, 0, 1)[0]

    def dimensionalize(self, L, U, mu):  # noqa N803
        """Dimensionalize all physical quantities.

        Units are assumed to be [L] = m, [U] = m/s, [mu] = Pa s.
        """
        # TODO should this return a new solution object?
        if self.status == "d":
            raise ValueError("Solution is already dimensionalized.")
        self.L = L
        self.U = U
        self.mu = mu

        def dim_psi(z):
            return self.psi(z / L) * U * L

        def dim_uv(z):
            return self.uv(z / L) * U

        def dim_p(z):
            return self.p(z / L) * mu * U / L

        def dim_omega(z):
            return self.omega(z / L) * U / L

        def dim_eij(z):
            return self.eij(z / L) * U / L

        self.problem.domain.dimensionalize(L)

        return Solution(
            self.problem,
            dim_psi,
            dim_uv,
            dim_p,
            dim_omega,
            dim_eij,
            L=L,
            U=U,
            mu=mu,
            status="d",
        )

    # def pickle(self, filename):
    #     """Pickle the solution."""
    #     pkl.dump(self, open(filename, "wb"))

    def __add__(self, other):
        """Add two solutions together."""
        if isinstance(other, Solution):
            psi_1, uv_1, p_1, omega_1, eij_1 = self.functions
            psi_2, uv_2, p_2, omega_2, eij_2 = other.functions

            def psi_combined(z):
                return psi_1(z) + psi_2(z)

            def uv_combined(z):
                return uv_1(z) + uv_2(z)

            def p_combined(z):
                return p_1(z) + p_2(z)

            def omega_combined(z):
                return omega_1(z) + omega_2(z)

            def eij_combined(z):
                return eij_1(z) + eij_2(z)

            sol_combined = Solution(
                self.problem,
                psi_combined,
                uv_combined,
                p_combined,
                omega_combined,
                eij_combined,
            )
            return sol_combined
        else:
            return NotImplemented

    def __sub__(self, other):
        """Subtract one solutions from another."""
        if isinstance(other, Solution):
            psi_1, uv_1, p_1, omega_1, eij_1 = self.functions
            psi_2, uv_2, p_2, omega_2, eij_2 = other.functions

            def psi_combined(z):
                return psi_1(z) - psi_2(z)

            def uv_combined(z):
                return uv_1(z) - uv_2(z)

            def p_combined(z):
                return p_1(z) - p_2(z)

            def omega_combined(z):
                return omega_1(z) - omega_2(z)

            def eij_combined(z):
                return eij_1(z) - eij_2(z)

            sol_combined = Solution(
                self.problem,
                psi_combined,
                uv_combined,
                p_combined,
                omega_combined,
                eij_combined,
            )
            return sol_combined
        else:
            return NotImplemented

    def __mul__(self, other):
        """Multiply solution by a scalar."""
        if isinstance(other, Number):
            psi_1, uv_1, p_1, omega_1, eij_1 = self.functions

            def psi_combined(z):
                return psi_1(z) * other

            def uv_combined(z):
                return uv_1(z) * other

            def p_combined(z):
                return p_1(z) * other

            def omega_combined(z):
                return omega_1(z) * other

            def eij_combined(z):
                return eij_1(z) * other

            sol_combined = Solution(
                self.problem,
                psi_combined,
                uv_combined,
                p_combined,
                omega_combined,
                eij_combined,
            )
            return sol_combined

    def __rmul__(self, other):
        """Reverse multiply solution by a scalar."""
        return self.__mul__(other)

    def __div__(self, other):
        """Divide solution by a scalar."""
        if isinstance(other, Number):
            psi_1, uv_1, p_1, omega_1, eij_1 = self.functions

            def psi_combined(z):
                return psi_1(z) / other

            def uv_combined(z):
                return uv_1(z) / other

            def p_combined(z):
                return p_1(z) / other

            def omega_combined(z):
                return omega_1(z) / other

            def eij_combined(z):
                return eij_1(z) / other

            sol_combined = Solution(
                self.problem,
                psi_combined,
                uv_combined,
                p_combined,
                omega_combined,
                eij_combined,
            )
            return sol_combined
        else:
            return NotImplemented

    def __neg__(self):
        """Negate the solution."""
        psi_1, uv_1, p_1, omega_1, eij_1 = self.functions

        def psi_combined(z):
            return -psi_1(z)

        def uv_combined(z):
            return -uv_1(z)

        def p_combined(z):
            return -p_1(z)

        def omega_combined(z):
            return -omega_1(z)

        def eij_combined(z):
            return -eij_1(z)

        sol_combined = Solution(
            self.problem,
            psi_combined,
            uv_combined,
            p_combined,
            omega_combined,
            eij_combined,
        )
        return sol_combined
