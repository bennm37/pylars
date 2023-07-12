"""Test boundary conditions."""
from test_settings import ATOL, RTOL


def test_validate_expression():
    """Test the syntax validator of the Problem class."""
    from pylars import Problem
    import numpy as np
    corners = [1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j]
    prob = Problem()
    prob.add_exterior_polygon(
        corners
    )
    prob.validate("v[0]")
    prob.validate("p[0]+psi[3]")
    prob.validate("psi[0]+psi[1]")
    prob.validate("u[0]-y*(1-y)**2.0")
    # check validate throws a ValueError if the expression is invalid
    try:
        prob.validate("v(0")
        assert False
    except ValueError:
        pass
    # check validate throws a ValueError if the expression contains invalid
    # characters
    try:
        prob.validate("%")
        assert False
    except ValueError:
        pass
    # check validate throws a ValueError for mismatched parentheses
    try:
        prob.validate("v[0]]")
    except ValueError:
        pass


def test_add_boundary_conditions():
    """Test adding boundary conditions."""
    from pylars import Problem

    # typical BCs
    corners = [1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j]
    prob = Problem()
    prob.add_exterior_polygon(corners, num_edge_points=100)
    # parabolic inlet flow
    prob.add_boundary_condition("0", "u[0]-y*(1-y)", 0)
    prob.add_boundary_condition("0", "v[0]", 0)
    # 0 pressure and normal velocity
    prob.add_boundary_condition("2", "p[2]", 0)
    prob.add_boundary_condition("2", "v[2]", 0)
    # no slip no penetration on the walls
    prob.add_boundary_condition("1", "u[1]", 0)
    prob.add_boundary_condition("1", "v[1]", 0)
    prob.add_boundary_condition("3", "u[3]", 0)
    prob.add_boundary_condition("3", "v[3]", 0)
    prob.check_boundary_conditions()


def test_evaluate_expression():
    """Test evaluating an expression."""
    from pylars import Problem, Solver
    import numpy as np

    corners = [1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j]
    prob = Problem()
    prob.add_exterior_polygon(corners)
    solver = Solver(prob)
    solver.setup()
    assert np.all(solver.evaluate("u[0]-u[0]", 0) == 0)
    assert np.all(solver.evaluate("psi[0]-psi[0]", 1) == 0)
    expression = "u[0]-y*(1-y)"
    solver.problem.validate(expression)
    y = np.linspace(0, 99, 100).reshape(100, 1)
    result = solver.evaluate(expression, 1j * y)
    expected = solver.U[prob.domain.indices["0"]] - (y * (1 - y))
    assert np.allclose(result, expected, atol=ATOL, rtol=RTOL)


def test_evaluate_expression_names():
    """Test evaluating an expression."""
    from pylars import Problem, Solver
    import numpy as np

    corners = [1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j]
    prob = Problem()
    prob.add_exterior_polygon(corners)
    prob.domain.name_side("0", "inlet")
    prob.domain.name_side("2", "outlet")
    prob.domain.group_sides(["1", "3"], "walls")
    solver = Solver(prob)
    solver.setup()
    periodic = solver.evaluate("u[inlet]-u[outlet][::-1]", 0)
    stream = solver.evaluate("psi[walls]", 0)
    periodic_answer = (
        solver.U[prob.domain.indices["inlet"]]
        - solver.U[prob.domain.indices["outlet"]][::-1]
    )
    stream_answer = solver.PSI[prob.domain.indices["walls"]]
    assert np.allclose(periodic, periodic_answer, atol=ATOL, rtol=RTOL)
    assert np.allclose(stream, stream_answer, atol=ATOL, rtol=RTOL)


if __name__ == "__main__":
    test_validate_expression()
    test_add_boundary_conditions()
    test_evaluate_expression()
    test_evaluate_expression_names()
