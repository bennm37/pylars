"""Test dimensionalizing solutions."""


def test_dimensionalize():
    """Test dimensionalizing solutions."""
    from pylars import Problem, Solution
    import numpy as np

    prob = Problem()
    prob.add_exterior_polygon([0, 1, 1j])

    def psi(z):
        return z

    def uv(z):
        return z**2

    def p(z):
        return z**3

    def omega(z):
        return z**4

    def eij(z):
        return z**5

    sol = Solution(prob, psi, uv, p, omega, eij)
    assert sol.status == "nd"
    sol_dim = sol.dimensionalize(L=2, U=5, mu=0.1)
    assert sol_dim.status == "d"
    assert sol_dim.L == 2
    assert sol_dim.U == 5
    assert sol_dim.mu == 0.1
    x = np.linspace(-1, 1, 100)
    y = np.linspace(-1, 2, 100)
    X, Y = np.meshgrid(x, y, indexing="ij")
    Z = X + 1j * Y
    assert np.allclose(sol_dim.psi(Z * 2), Z * 2 * 5)
    assert np.allclose(sol_dim.uv(Z * 2), Z**2 * 5)
    assert np.allclose(sol_dim.p(Z * 2), Z**3 * 0.1 * 5 / 2)
    assert np.allclose(sol_dim.omega(Z * 2), Z**4 * 5 / 2)
    assert np.allclose(sol_dim.eij(Z * 2), Z**5 * 5 / 2)


if __name__ == "__main__":
    test_dimensionalize()
