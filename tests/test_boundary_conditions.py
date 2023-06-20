def test_validate_expression():
    from pyls import Domain, Solver

    corners = [1 + 1j, -1 + 1j, -1 -1j, 1 - 1j]
    dom = Domain(corners)
    sol = Solver(dom, 10)
    sol.validate("v(0)")
    sol.validate("p(0)+psi(3)")
    sol.validate("psi(0)+psi(1)")
    sol.validate("u(0)-y*(1-y)^2.0")
    # check validate throws a ValueError if the expression is invalid
    try:
        sol.validate("v(0")
        assert False
    except ValueError:
        pass
    # check validate throws a ValueError if the expression contains invalid characters
    try:
        sol.validate("%")
        assert False
    except ValueError:
        pass
    # check validate throws a ValueError for mismatched parentheses
    try:
        sol.validate("v(0))")
        assert False
    except ValueError:
        pass

def test_add_boundary_conditions():
    from pyls import Domain, Solver
    import numpy as np
    # typical BCs
    corners = [1 + 1j, -1 + 1j, -1 -1j, 1 - 1j]
    dom = Domain(corners, num_boundary_points=100)
    dom.show()
    sol = Solver(dom, 10)
    # parabolic inlet flow 
    sol.add_boundary_condition("0", "u(0)-y*(1-y)", 0)
    sol.add_boundary_condition("0", "v(0)", 0)
    # 0 pressure and normal velocity
    sol.add_boundary_condition("2", "p(2)", 0)
    sol.add_boundary_condition("2", "v(2)", 0)
    # no slip no penetration on the walls
    sol.add_boundary_condition("1", "u(1)", 0)
    sol.add_boundary_condition("1", "v(1)", 0)
    sol.add_boundary_condition("3", "u(3)", 0)
    sol.add_boundary_condition("3", "v(3)", 0)
    print(sol.boundary_conditions)

def test_evaluate_expression():
    pass

if __name__ == "__main__":
    test_validate_expression()
    test_add_boundary_conditions()
