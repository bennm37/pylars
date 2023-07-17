"""Solution class for pylars."""
from numbers import Number
import numpy as np


class Solution:
    """Solution class to store the solution to a problem."""

    def __init__(self, psi, uv, p, omega, eij):
        self.psi = psi
        self.uv = uv
        self.p = p
        self.omega = omega
        self.eij = eij
        self.functions = [psi, uv, p, omega, eij]

    def stress_discrete(self, z, dx=1e-6):
        """Calculate the total stress using finite differences."""
        zshape = z.shape
        z = z.reshape(-1)
        pressures = self.p(z)
        isotropic = np.array([np.eye(2) * p for p in pressures]).reshape(
            zshape + (2, 2)
        )
        # caluclate velocity gradients using forward difference for interior
        # points
        # TODO handle boundary points
        uv = self.uv(z)
        U_x = (self.uv(z + dx) - uv) / dx
        u_x, v_x = U_x.real, U_x.imag
        U_y = (self.uv(z + dx * 1j) - uv) / dx
        u_y, v_y = U_y.real, U_y.imag
        deviatoric = np.array([[2 * u_x, u_y + v_x], [u_y + v_x, 2 * v_y]])
        deviatoric = np.moveaxis(deviatoric, (2), (0)).reshape(zshape + (2, 2))
        stress = isotropic + deviatoric
        return stress

    def stress_goursat(self, z):
        """Calculate the total stress using finite differences."""
        zshape = z.shape
        z = z.reshape(-1)
        pressures = self.p(z)
        isotropic = np.array([np.eye(2) * p for p in pressures]).reshape(
            zshape + (2, 2)
        )
        deviatoric = 2 * self.eij(z).reshape(zshape + (2, 2))
        stress = isotropic + deviatoric
        return stress

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
            psi_combined,
            uv_combined,
            p_combined,
            omega_combined,
            eij_combined,
        )
        return sol_combined
