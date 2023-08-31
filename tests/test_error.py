"""Test evaluating the error."""


def test_domain_error_points():
    """Test generating the error point mesh."""
    import numpy as np
    from pylars import Problem

    corners = np.array([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j])
    prob = Problem()
    num_edge_points = 600
    prob.add_exterior_polygon(
        corners,
        num_edge_points=num_edge_points,
        deg_poly=120,
        num_poles=0,
        spacing="linear",
    )
    centroid = -0.4j
    R = 5e-1j
    circle = lambda t: centroid + R * np.exp(2j * np.pi * t)
    num_circle_points = 400
    prob.add_interior_curve(
        circle,
        num_points=num_circle_points,
        deg_laurent=80,
        centroid=centroid,
    )
    error_points_rect = [prob.domain.error_points[str(i)] for i in range(4)]
    error_points_circle = prob.domain.error_points["4"]
    for i in range(4):
        assert np.allclose(
            error_points_rect[i],
            np.linspace(corners[i], corners[(i + 1) % 4], num_edge_points * 2),
        )
    assert np.allclose(
        error_points_circle,
        circle(np.linspace(0, 1, num_circle_points * 2)),
    )

    # import matplotlib.pyplot as plt

    # fig, ax = prob.domain.plot()
    # for side in prob.domain.sides:
    #     points = prob.domain.error_points[side]
    #     ax.scatter(points.real, points.imag, s=1, color="k")
    # plt.show()


def test_get_error():
    import numpy as np
    from pylars import Problem, Solver

    corners = np.array([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j])
    prob = Problem()
    num_edge_points = 600
    prob.add_exterior_polygon(
        corners,
        num_edge_points=num_edge_points,
        deg_poly=120,
        num_poles=0,
        spacing="linear",
    )
    centroid = -0.4j
    R = 5e-1j
    circle = lambda t: centroid + R * np.exp(2j * np.pi * t)
    prob.add_interior_curve(
        circle,
        num_points=400,
        deg_laurent=80,
        centroid=centroid,
    )
    prob.add_boundary_condition("0", "u[0]", 0)
    prob.add_boundary_condition("0", "v[0]", 0)
    prob.add_boundary_condition("2", "u[2]", 0)
    prob.add_boundary_condition("2", "v[2]", 0)
    prob.add_boundary_condition("1", "p[1]", 2)
    prob.add_boundary_condition("1", "v[1]", 0)
    prob.add_boundary_condition("3", "p[3]", -2)
    prob.add_boundary_condition("3", "v[3]", 0)
    prob.add_boundary_condition("4", "u[4]", 0)
    prob.add_boundary_condition("4", "v[4]", 0)
    solver = Solver(prob)
    sol = solver.solve(weight=False, normalize=False)
    max_error, error = solver.get_error()
    print(f"Max Error: {max_error}")


if __name__ == "__main__":
    test_domain_error_points()
    test_get_error()
