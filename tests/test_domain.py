"""Test the domain module."""


def test_import_domain():
    """Test that the domain module can be imported."""
    from pyls.domain import Domain

    assert Domain is not None


def test_domain_spacing_rectangle():
    """Test that the spacing is correct for a rectangle."""
    from pyls.domain import Domain

    corners = [0, 1, 1 + 1j, 1j]
    dom = Domain(corners)
    dom.get_boundary_points()
    assert dom.boundary_points[0][0] == 0
