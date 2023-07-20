"""Test translation and rotation of moving objects."""


def test_move_square():
    """Test translation and rotation of a square."""
    from pylars import Problem
    import numpy as np

    def square(t):
        if t <= 0.25:
            return 1 - t * 8 + 1j
        elif t <= 0.5:
            return -1 + 1j - (t - 0.25) * 8j
        elif t <= 0.75:
            return -1 + (t - 0.5) * 8 - 1j
        else:
            return 1 - 1j + (t - 0.75) * 8j

    square = np.vectorize(square)
    centroid = 0.5 + 0.1j
    small_square = lambda t: centroid + square(t) / 5
    corners = [-2 - 1j, 2 - 1j, 2 + 1j, -2 + 1j]

    prob1 = Problem()
    prob1.add_exterior_polygon(
        corners,
        num_edge_points=100,
        num_poles=0,
        deg_poly=20,
        spacing="linear",
    )
    prob1.add_interior_curve(
        small_square, num_points=100, deg_laurent=10, centroid=centroid
    )
    disp = 0.1 + 0.2j
    translated_square_func = lambda t: small_square(t) + disp
    translated_square_answer = translated_square_func(np.linspace(0, 1, 100))
    prob1.domain.translate("4", disp)
    assert np.allclose(
        prob1.domain.boundary_points[prob1.domain.indices["4"]],
        translated_square_answer.reshape(-1, 1),
    )

    prob2 = Problem()
    prob2.add_exterior_polygon(
        corners,
        num_edge_points=100,
        num_poles=0,
        deg_poly=20,
        spacing="linear",
    )
    prob2.add_interior_curve(
        small_square, num_points=100, deg_laurent=10, centroid=centroid
    )
    angle = np.pi / 4
    prob2.domain.rotate("4", angle)
    rotated_square_func = lambda t: centroid + square(t) / 5 * np.exp(
        1j * angle
    )
    rotated_square_answer = rotated_square_func(np.linspace(0, 1, 100))
    assert np.allclose(
        prob2.domain.boundary_points[prob2.domain.indices["4"]],
        rotated_square_answer.reshape(-1, 1),
    )


if __name__ == "__main__":
    test_move_square()
