def test_poiseuille_permeability():
    from pylars import Problem, Solver, Analysis
    import numpy as np

    corners = [1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j]
    prob = Problem()
    prob.add_exterior_polygon(
        corners,
        num_edge_points=300,
        num_poles=0,
        deg_poly=24,
        spacing="linear",
    )
    p_drop = 2
    prob.add_boundary_condition("0", "u[0]", 0)
    prob.add_boundary_condition("0", "v[0]", 0)
    prob.add_boundary_condition("2", "u[2]", 0)
    prob.add_boundary_condition("2", "v[2]", 0)
    prob.add_boundary_condition("1", "p[1]", p_drop)
    prob.add_boundary_condition("1", "v[1]", 0)
    prob.add_boundary_condition("3", "p[3]", 0)
    prob.add_boundary_condition("3", "v[3]", 0)
    solver = Solver(prob)
    sol = solver.solve(weight=False, normalize=False)
    print(f"Error: {solver.max_error}")
    an = Analysis(sol)
    curve = lambda t: 1 + 2j * t - 1j
    points = curve(np.linspace(0, 1, 100))
    curve_deriv = lambda t: 2j * np.ones_like(t)
    permeability = an.get_permeability(curve, curve_deriv, delta_x=2, delta_p=p_drop)
    y = curve(np.linspace(0, 1, 100)).imag
    assert np.allclose(sol.uv(1 + y * 1j).reshape(-1), (1 - y**2) / 2)
    assert np.isclose(permeability, 1 / 3)
    L, U, mu = 1e-5, 1e-5, 1e-3
    dim_sol = sol.dimensionalize(L=L, U=U, mu=mu)
    dim_an = Analysis(dim_sol)
    dim_curve = lambda t: L * (1 + 2j * t - 1j)
    dim_curve_deriv = lambda t: 2j * np.ones_like(t) * L
    dim_delta_x = 2 * L
    dim_delta_p = p_drop * mu * U / L
    dim_permeability = dim_an.get_permeability(
        dim_curve, dim_curve_deriv, delta_x=dim_delta_x, delta_p=dim_delta_p
    )
    assert np.allclose(dim_sol.uv(L * (1 + y * 1j)).reshape(-1), U * (1 - y**2) / 2)
    assert np.isclose(dim_permeability, L**2 / 3)


if __name__ == "__main__":
    test_poiseuille_permeability()
