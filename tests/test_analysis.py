"""Tests basic properties of the Analysis class."""


def test_import_analysis():
    """Test importing analysis."""
    from pylars import Analysis

    assert Analysis is not None


def test_save_pgf():
    """Test saving figure as a pgf."""
    from pylars import Problem, Solver, Analysis
    import matplotlib.pyplot as plt

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


if __name__ == "__main__":
    test_import_analysis()
    test_save_pgf()
