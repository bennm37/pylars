"""Test Linear Add and Mul methods of Solver class."""


def test_add():
    """Test adding two Solver objects."""
    from pyls import Domain, Solver
    import numpy as np

    corners = [1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j]
    dom = Domain(
        corners, num_boundary_points=300, num_poles=0, spacing="linear"
    )
    sol1 = Solver(dom, 24)
    sol2 = Solver(dom, 24)
    sol1.functions = [lambda x: x, lambda x: x, lambda x: x, lambda x: x]
    sol2.functions = [
        lambda x: x**2 - 1,
        lambda x: x**2 - 2,
        lambda x: x**2 - 3,
        lambda x: x**2 - 4,
    ]
    combination_functions = [
        lambda x: x + x**2 - 1,
        lambda x: x + x**2 - 2,
        lambda x: x + x**2 - 3,
        lambda x: x + x**2 - 4,
    ]
    sol3 = sol1 + sol2
    points = np.linspace(0, 1, 100)
    for func3, func in zip(sol3.functions, combination_functions):
        assert np.allclose(func(points), func3(points))


def test_mul():
    """Test multiplying a Solver object by a scalar."""
    from pyls import Domain, Solver
    import numpy as np

    corners = [1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j]
    dom = Domain(
        corners, num_boundary_points=300, num_poles=0, spacing="linear"
    )
    sol1 = Solver(dom, 24)
    sol1.functions = [
        lambda x: x**2 - 1,
        lambda x: x**2 - 2,
        lambda x: x**2 - 3,
        lambda x: x**2 - 4,
    ]
    combination_functions = [
        lambda x: 5 * (x**2 - 1),
        lambda x: 5 * (x**2 - 2),
        lambda x: 5 * (x**2 - 3),
        lambda x: 5 * (x**2 - 4),
    ]
    sol2 = sol1 * 5
    points = np.linspace(0, 1, 100)
    for func2, func in zip(sol2.functions, combination_functions):
        assert np.allclose(func(points), func2(points))


def test_linear_combination():
    """Test linear combination of solver objects."""
    from pyls import Domain, Solver
    import numpy as np

    corners = [1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j]
    dom = Domain(
        corners, num_boundary_points=300, num_poles=0, spacing="linear"
    )
    sol1 = Solver(dom, 24)
    sol1.functions = [lambda x: x, lambda x: x, lambda x: x, lambda x: x]
    sol2 = Solver(dom, 24)
    sol2.functions = [
        lambda x: x**2,
        lambda x: x**2,
        lambda x: x**2,
        lambda x: x**2,
    ]
    combination_functions = [
        lambda x: 3 * x**2 - 2 * x,
        lambda x: 3 * x**2 - 2 * x,
        lambda x: 3 * x**2 - 2 * x,
        lambda x: 3 * x**2 - 2 * x,
    ]
    sol3 = 3 * sol2 - sol1 * 2
    points = np.linspace(0, 1, 100)
    for func3, func in zip(sol3.functions, combination_functions):
        assert np.allclose(func(points), func3(points))


def test_negate():
    """Test negation of solver objects."""
    from pyls import Domain, Solver
    import numpy as np

    corners = [1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j]
    dom = Domain(
        corners, num_boundary_points=300, num_poles=0, spacing="linear"
    )
    sol1 = Solver(dom, 24)
    sol1.functions = [lambda x: x, lambda x: -x, lambda x: x, lambda x: -x]
    sol2 = -sol1
    negation = [
        lambda x: -x,
        lambda x: x,
        lambda x: -x,
        lambda x: x,
    ]
    points = np.linspace(0, 1, 100)
    for func2, func in zip(sol2.functions, negation):
        assert np.allclose(func(points), func2(points))


if __name__ == "__main__":
    test_add()
    test_mul()
    test_linear_combination()
    test_negate()
