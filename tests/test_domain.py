"""Test the domain module."""
from test_settings import ATOL, RTOL


def test_import_domain():
    """Test that the domain module can be imported."""
    from pyls.domain import Domain

    assert Domain is not None


def test_domain_spacing_rectangle():
    """Test that the spacing is correct for a rectangle."""
    from pyls.domain import Domain
    from pyls.numerics import cart
    import numpy as np

    corners = [1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j]
    dom = Domain(corners, num_boundary_points=100)
    # check the points move anticlockwise around the domain
    # using the cross prodcut.
    for i in range(1, len(dom.boundary_points) - 1):
        assert (
            np.cross(
                cart(dom.boundary_points[i + 1] - dom.boundary_points[i]),
                cart(dom.boundary_points[i] - dom.boundary_points[i - 1]),
            )
            >= -1e-10
        )


def test_poles_square():
    """Test that the poles against MATLAB code for a square domain."""
    from pyls.domain import Domain
    from scipy.io import loadmat
    import numpy as np

    corners = [1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j]
    dom = Domain(corners, num_boundary_points=100, sigma=4, L=1.5 * np.sqrt(2))
    poles = dom.poles

    poles_answer = loadmat("tests/data/square_domain_poles.mat")["Pol"][0]
    poles_answer = np.array(
        [poles_answer[i] for i in range(len(poles_answer))]
    ).reshape(len(poles_answer), poles_answer[0].shape[1])
    assert np.allclose(poles, poles_answer, atol=ATOL, rtol=RTOL)


def test_contains():
    import numpy as np
    from pyls.domain import Domain

    dom = Domain([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j], num_boundary_points=100)
    assert 0j in dom
    assert 2 + 2j not in dom
    assert 0.99 + 0j in dom
    assert -1.01 + 0j not in dom
    assert np.array([0.99 + 0j, -1.01 + 0j]) not in dom
    assert np.array([0.99 + 0j, -0.99 + 0j]) in dom
    assert (
        np.array([[0.5 + 0.5j, -0.5 + 0.5j], [-0.5 + 0.5j, -0.5 - 0.5j]])
        in dom
    )

def test_mask_contains():
    import numpy as np
    from pyls.domain import Domain
    corners = [1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j]
    Z = np.array([[1.5 + 0.5j, -0.5 + 0.5j], [-0.5 + 0.5j, -0.5 - 0.5j]])
    mask_answer = np.array([[False, True], [True, True]])
    dom = Domain(corners, num_boundary_points=100)
    mask = dom.mask_contains(Z)
    assert np.allclose(mask, mask_answer)

if __name__ == "__main__":
    test_import_domain()
    test_domain_spacing_rectangle()
    test_poles_square()
    test_contains()
    test_mask_contains()
