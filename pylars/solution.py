"""Solution class for pylars."""
from numbers import Number

class Solution:
    
    def __init__(self,psi,uv,p,omega):
        self.psi = psi
        self.uv = uv
        self.p = p
        self.omega = omega
    
    def __add__(self, other):
        """Add two solutions together."""
        if isinstance(other, Solution):
            sol_combined = Solution(self.domain, self.degree)
            psi_1, uv_1, p_1, omega_1 = self.functions
            psi_2, uv_2, p_2, omega_2 = other.functions

            def psi_combined(z):
                return psi_1(z) + psi_2(z)

            def uv_combined(z):
                return uv_1(z) + uv_2(z)

            def p_combined(z):
                return p_1(z) + p_2(z)

            def omega_combined(z):
                return omega_1(z) + omega_2(z)

            sol_combined.functions = (
                psi_combined,
                uv_combined,
                p_combined,
                omega_combined,
            )
            return sol_combined
        else:
            return NotImplemented

    def __sub__(self, other):
        """Subtract one solutions from another."""
        if isinstance(other, Solution):
            sol_combined = Solution(self.domain, self.degree)
            psi_1, uv_1, p_1, omega_1 = self.functions
            psi_2, uv_2, p_2, omega_2 = other.functions

            def psi_combined(z):
                return psi_1(z) - psi_2(z)

            def uv_combined(z):
                return uv_1(z) - uv_2(z)

            def p_combined(z):
                return p_1(z) - p_2(z)

            def omega_combined(z):
                return omega_1(z) - omega_2(z)

            sol_combined.functions = (
                psi_combined,
                uv_combined,
                p_combined,
                omega_combined,
            )
            return sol_combined
        else:
            return NotImplemented

    def __mul__(self, other):
        """Multiply solution by a scalar."""
        if isinstance(other, Number):
            sol_combined = Solution(self.domain, self.degree)
            psi_1, uv_1, p_1, omega_1 = self.functions

            def psi_combined(z):
                return psi_1(z) * other

            def uv_combined(z):
                return uv_1(z) * other

            def p_combined(z):
                return p_1(z) * other

            def omega_combined(z):
                return omega_1(z) * other

            sol_combined.functions = (
                psi_combined,
                uv_combined,
                p_combined,
                omega_combined,
            )
            return sol_combined

    def __rmul__(self, other):
        """Reverse multiply solution by a scalar."""
        return self.__mul__(other)

    def __div__(self, other):
        """Divide solution by a scalar."""
        if isinstance(other, Number):
            sol_combined = Solution(self.domain, self.degree)
            psi_1, uv_1, p_1, omega_1 = self.functions

            def psi_combined(z):
                return psi_1(z) / other

            def uv_combined(z):
                return uv_1(z) / other

            def p_combined(z):
                return p_1(z) / other

            def omega_combined(z):
                return omega_1(z) / other

            sol_combined.functions = (
                psi_combined,
                uv_combined,
                p_combined,
                omega_combined,
            )
            return sol_combined
        else:
            return NotImplemented

    def __neg__(self):
        """Negate the solution."""
        sol_combined = Solution(self.domain, self.degree)
        psi_1, uv_1, p_1, omega_1 = self.functions

        def psi_combined(z):
            return -psi_1(z)

        def uv_combined(z):
            return -uv_1(z)

        def p_combined(z):
            return -p_1(z)

        def omega_combined(z):
            return -omega_1(z)

        sol_combined.functions = (
            psi_combined,
            uv_combined,
            p_combined,
            omega_combined,
        )
        return sol_combined
