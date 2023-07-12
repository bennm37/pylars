"""Test Linear Add and Mul methods of Solver class."""


def test_add():
    """Test adding two Solver objects."""
    from pylars import Solution
    import numpy as np

    sol1 = Solution(lambda x: x, lambda x: x, lambda x: x, lambda x: x)
    sol2 = Solution(
        lambda x: x**2 - 1,
        lambda x: x**2 - 2,
        lambda x: x**2 - 3,
        lambda x: x**2 - 4,
    )
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
    from pylars import Solution
    import numpy as np

    sol1 = Solution(
        lambda x: x**2 - 1,
        lambda x: x**2 - 2,
        lambda x: x**2 - 3,
        lambda x: x**2 - 4,
    )
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
    from pylars import Solution
    import numpy as np

    sol1 = Solution(lambda x: x, lambda x: x, lambda x: x, lambda x: x)
    sol2 = Solution(
        lambda x: x**2,
        lambda x: x**2,
        lambda x: x**2,
        lambda x: x**2,
    )
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
    from pylars import Solution
    import numpy as np

    sol1 = Solution(lambda x: x, lambda x: -x, lambda x: x, lambda x: -x)
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