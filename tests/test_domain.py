"""Test the domain module."""


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
    assert np.allclose(dom.boundary_points[0], 1 - 1j)
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

def test_poles_rectangle():
    from pyls.domain import Domain
    from pyls.numerics import cart
    import numpy as np
    corners = [1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j]
    dom = Domain(corners, num_boundary_points=100)
    dom.generate_boundary_points()