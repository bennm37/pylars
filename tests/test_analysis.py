"""Tests basic properties of the Analysis class."""


def test_import_analysis():
    """Test importing analysis."""
    from pylars import Analysis

    assert Analysis is not None


def test_save_pgf():
    """Test saving figure as a pgf."""
    from pylars import Problem, Solver, Analysis

    corners = [1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j]
    prob = Problem()
    prob.add_exterior_polygon(
        corners,
        num_edge_points=300,
        num_poles=0,
        deg_poly=24,
        spacing="linear",
    )
    prob.add_boundary_condition("0", "u[0]", 0)
    prob.add_boundary_condition("0", "v[0]", 0)
    prob.add_boundary_condition("2", "u[2]", 0)
    prob.add_boundary_condition("2", "v[2]", 0)
    prob.add_boundary_condition("1", "p[1]", 2)
    prob.add_boundary_condition("1", "v[1]", 0)
    prob.add_boundary_condition("3", "p[3]", -2)
    prob.add_boundary_condition("3", "v[3]", 0)
    solver = Solver(prob)
    sol = solver.solve(weight=False, normalize=False)
    analysis = Analysis(sol)
    fig, ax = analysis.plot(imshow=True, resolution=200)
    ax.set(title="PGF test")
    analysis.save_pgf("media/pgf_test", fig)


def test_curve_length():
    from pylars import Problem, Solution, Analysis
    import numpy as np
    from test_linear_combination import tensorify

    prob = Problem()
    sol = Solution(
        prob,
        psi=lambda z: z - z.imag**3 / 3,
        uv=lambda z: 1 - z.imag**2,
        p=lambda z: z,
        omega=lambda z: z,
        eij=lambda z: tensorify(z),
    )
    curve = lambda t: 1 - 1j + 2j * t
    curve_deriv = lambda t: 2j
    an = Analysis(sol)
    length = an.get_length(curve_deriv)
    assert np.isclose(length, 2)
    curve = lambda t: np.exp(2j * np.pi * t)
    curve_deriv = lambda t: 2j * np.pi * np.exp(2j * np.pi * t)
    length = an.get_length(curve_deriv)
    assert np.isclose(length, 2 * np.pi)


def test_volume_flux():
    from pylars import Problem, Solution, Analysis
    import numpy as np
    from test_linear_combination import tensorify

    prob = Problem()
    sol = Solution(
        prob,
        psi=lambda z: z - z.imag**3 / 3,
        uv=lambda z: 1 - z.imag**2,
        p=lambda z: z,
        omega=lambda z: z,
        eij=lambda z: tensorify(z),
    )
    curve = lambda t: 1 - 1j + 2j * t
    curve_deriv = lambda t: 2j
    an = Analysis(sol)
    vf = an.get_volume_flux(curve, curve_deriv)
    assert np.isclose(vf, 4 / 3)
    curve = lambda t: np.exp(2j * np.pi * t)
    curve_deriv = lambda t: 2j * np.pi * np.exp(2j * np.pi * t)
    vf = an.get_volume_flux(curve, curve_deriv)
    assert np.isclose(vf, 0)


if __name__ == "__main__":
    test_import_analysis()
    test_save_pgf()
    test_curve_length()
    test_volume_flux()
