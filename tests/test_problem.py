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
        deg_poly=24,
        num_poles=num_poles,
    )
    # check the points move anticlockwise around the domain
    # using the cross prodcut.
    for i in range(1, len(prob.boundary_points) - 1):
        assert (
            np.cross(
                cart(prob.boundary_points[i + 1] - prob.boundary_points[i]),
                cart(prob.boundary_points[i] - prob.boundary_points[i - 1]),
            )
            >= -1e-10
        )


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
        deg_poly=24,
        num_poles=num_poles,
    )
    poles = prob.poles

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
        np.array([[0.5 + 0.5j, -0.5 + 0.5j], [-0.5 + 0.5j, -0.5 - 0.5j]])
        in prob.domain
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


if __name__ == "__main__":
    test_import_problem()
    test_domain_spacing_rectangle()
    test_poles_square()
    test_contains()
    test_mask_contains()
