"""Test the Problem class."""
from test_settings import ATOL, RTOL


def test_import_problem():
    """Test that the Problem class can be imported."""
    from pylars import Problem

    assert Problem is not None


def test_domain_spacing_rectangle():
    """Test that the spacing is correct for a rectangle."""
    from pylars import Problem
    from pylars.numerics import cart
    import numpy as np

    corners = [1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j]
    prob = Problem()
    num_poles = 24
    prob.add_exterior_polygon(
        corners,
        num_edge_points=300,
        length_scale=1.5 * np.sqrt(2),
        sigma=4,
        deg_poly=24,
        num_poles=num_poles,
    )
    # check the points move anticlockwise around the domain
    # using the cross prodcut.
    for i in range(1, len(prob.domain.boundary_points) - 1):
        assert (
            np.cross(
                cart(
                    prob.domain.boundary_points[i + 1] - prob.domain.boundary_points[i]
                ),
                cart(
                    prob.domain.boundary_points[i] - prob.domain.boundary_points[i - 1]
                ),
            )
            >= -1e-10
        )


def test_domain_spacing_circle():
    """Test domain spacing on a circle against MATLAB code."""
    from scipy.io import loadmat
    import numpy as np
    from pylars import Problem

    test_answers = loadmat("tests/data/single_circle_test.mat")
    deg_poly, num_poles, deg_laurent = (
        test_answers["n"][0][0],
        test_answers["np"][0][0],
        test_answers["nl"][0][0],
    )
    num_edge_points, num_ellipse_points = (
        test_answers["nb"][0][0],
        test_answers["np"][0][0],
    )
    Z_answer = test_answers["Z"]
    prob = Problem()
    corners = [-1 - 1j, 1 - 1j, 1 + 1j, -1 + 1j]
    prob.add_exterior_polygon(
        corners,
        num_edge_points=num_edge_points,
        num_poles=num_poles,
        deg_poly=deg_poly,
        spacing="linear",
    )
    prob.add_interior_curve(
        lambda t: 0.5 * np.exp(2j * np.pi * t),
        num_points=num_ellipse_points,
        deg_laurent=deg_laurent,
        centroid=0.0 + 0.0j,
    )
    assert np.allclose(prob.domain.boundary_points, Z_answer, atol=ATOL, rtol=RTOL)


def test_poles_square():
    """Test that the poles against MATLAB code for a square domain."""
    from pylars import Problem
    from scipy.io import loadmat
    import numpy as np

    corners = [1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j]
    prob = Problem()
    num_poles = 24
    prob.add_exterior_polygon(
        corners,
        num_edge_points=300,
        length_scale=1.5 * np.sqrt(2),
        sigma=4,
        deg_poly=24,
        num_poles=num_poles,
    )
    poles = prob.domain.poles

    poles_answer = loadmat("tests/data/square_domain_poles.mat")["Pol"][0]
    poles_answer = np.array(
        [poles_answer[i] for i in range(len(poles_answer))]
    ).reshape(len(poles_answer), poles_answer[0].shape[1])
    assert np.allclose(poles, poles_answer, atol=ATOL, rtol=RTOL)


def test_contains():
    """Test the contains special method of the Domain class."""
    import numpy as np
    from pylars import Problem

    corners = [1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j]
    prob = Problem()
    num_poles = 24
    prob.add_exterior_polygon(
        corners,
        num_edge_points=300,
        length_scale=1.5 * np.sqrt(2),
        sigma=4,
        deg_poly=24,
        num_poles=num_poles,
    )
    assert 0j in prob.domain
    assert 2 + 2j not in prob.domain
    assert 0.99 + 0j in prob.domain
    assert -1.01 + 0j not in prob.domain
    assert np.array([0.99 + 0j, -1.01 + 0j]) not in prob.domain
    assert np.array([0.99 + 0j, -0.99 + 0j]) in prob.domain
    assert (
        np.array([[0.5 + 0.5j, -0.5 + 0.5j], [-0.5 + 0.5j, -0.5 - 0.5j]]) in prob.domain
    )


def test_mask_contains():
    """Test the mask_contains method of the Domain class."""
    import numpy as np
    from pylars import Problem

    corners = [1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j]
    Z = np.array([[1.5 + 0.5j, -0.5 + 0.5j], [-0.5 + 0.5j, -0.5 - 0.5j]])
    inside_answer = np.array([[False, True], [True, True]])
    prob = Problem()
    prob.add_exterior_polygon(corners, num_edge_points=100, deg_poly=24)
    inside = prob.domain.mask_contains(Z)
    assert np.allclose(inside, inside_answer)


def test_simple_poles_in_polygon():
    """Test that simple poles are inside the polygon."""
    import numpy as np
    from pylars import Problem

    corners = [1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j]
    prob = Problem()
    num_poles = 24
    prob.add_exterior_polygon(
        corners,
        num_edge_points=300,
        length_scale=1.5 * np.sqrt(2),
        sigma=4,
        deg_poly=24,
        num_poles=num_poles,
    )
    prob.add_interior_curve(
        lambda t: 0.5 * np.exp(2j * np.pi * t),
        num_points=100,
        deg_laurent=10,
        centroid=0.0 + 0.0j,
    )
    prob.add_interior_curve(
        lambda t: 0.5 + 0.5j + 0.1 * np.exp(2j * np.pi * t),
        num_points=100,
        deg_laurent=10,
        centroid=0.5 + 0.5j,
    )
    poly1 = [0.4 + 0.4j, -0.5 + 0.5j, -0.1 - 0.5j]
    indices_answer1 = [120]
    assert prob.domain.simple_poles_in_polygon(poly1) == indices_answer1
    poly2 = corners = [1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j]
    indices_answer2 = [120, 130]
    assert prob.domain.simple_poles_in_polygon(poly2) == indices_answer2


if __name__ == "__main__":
    test_import_problem()
    test_domain_spacing_rectangle()
    test_domain_spacing_circle()
    test_poles_square()
    test_contains()
    test_simple_poles_in_polygon()
    test_mask_contains()
