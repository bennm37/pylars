"""Test Linear Add and Mul methods of Solver class."""
import numpy as np


def tensorify(expr):
    """Restructure an expression into a 2x2 tensor."""
    return np.moveaxis([[expr, expr], [expr, expr]], 2, 0)


def test_add():
    """Test adding two Solver objects."""
    from pylars import Problem, Solution
    import numpy as np

    prob = Problem()
    sol1 = Solution(
        prob,
        lambda x: x,
        lambda x: x,
        lambda x: x,
        lambda x: x,
        lambda x: tensorify(x),
    )
    sol2 = Solution(
        prob,
        lambda x: x**2 - 1,
        lambda x: x**2 - 2,
        lambda x: x**2 - 3,
        lambda x: x**2 - 4,
        lambda x: tensorify(x**2 - 5),
    )
    combination_functions = [
        lambda x: x + x**2 - 1,
        lambda x: x + x**2 - 2,
        lambda x: x + x**2 - 3,
        lambda x: x + x**2 - 4,
        lambda x: tensorify(x + x**2 - 5),
    ]
    sol3 = sol1 + sol2
    points = np.linspace(0, 1, 100)
    for func3, func in zip(sol3.functions, combination_functions):
        assert np.allclose(func(points), func3(points))


def test_mul():
    """Test multiplying a Solver object by a scalar."""
    from pylars import Problem, Solution
    import numpy as np

    prob = Problem()
    sol1 = Solution(
        prob,
        lambda x: x**2 - 1,
        lambda x: x**2 - 2,
        lambda x: x**2 - 3,
        lambda x: x**2 - 4,
        lambda x: tensorify(x**2 - 5),
    )
    combination_functions = [
        lambda x: 5 * (x**2 - 1),
        lambda x: 5 * (x**2 - 2),
        lambda x: 5 * (x**2 - 3),
        lambda x: 5 * (x**2 - 4),
        lambda x: tensorify(5 * (x**2 - 5)),
    ]
    sol2 = sol1 * 5
    points = np.linspace(0, 1, 100)
    for func2, func in zip(sol2.functions, combination_functions):
        assert np.allclose(func(points), func2(points))


def test_linear_combination():
    """Test linear combination of solver objects."""
    from pylars import Problem, Solution
    import numpy as np

    prob = Problem()
    sol1 = Solution(
        prob,
        lambda x: x,
        lambda x: x,
        lambda x: x,
        lambda x: x,
        lambda x: tensorify(x),
    )
    sol2 = Solution(
        prob,
        lambda x: x**2,
        lambda x: x**2,
        lambda x: x**2,
        lambda x: x**2,
        lambda x: tensorify(x**2),
    )
    combination_functions = [
        lambda x: 3 * x**2 - 2 * x,
        lambda x: 3 * x**2 - 2 * x,
        lambda x: 3 * x**2 - 2 * x,
        lambda x: 3 * x**2 - 2 * x,
        lambda x: tensorify(3 * x**2 - 2 * x),
    ]
    sol3 = 3 * sol2 - sol1 * 2
    points = np.linspace(0, 1, 100)
    for func3, func in zip(sol3.functions, combination_functions):
        assert np.allclose(func(points), func3(points))


def test_negate():
    """Test negation of solver objects."""
    from pylars import Problem, Solution
    import numpy as np

    prob = Problem()
    sol1 = Solution(
        prob,
        lambda x: x,
        lambda x: -x,
        lambda x: x,
        lambda x: -x,
        lambda x: tensorify(x),
    )
    sol2 = -sol1
    negation = [
        lambda x: -x,
        lambda x: x,
        lambda x: -x,
        lambda x: x,
        lambda x: tensorify(-x),
    ]
    points = np.linspace(0, 1, 100)
    for func2, func in zip(sol2.functions, negation):
        assert np.allclose(func(points), func2(points))


if __name__ == "__main__":
    test_add()
    test_mul()
    test_linear_combination()
    test_negate()
