"""Test creating multiply connected domains."""


def test_create_circles():
    """Test creating circles in the domain."""
    from pylars import Problem
    import numpy as np

    corners = [1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j]
    prob = Problem()
    prob.add_exterior_polygon(
        corners=corners,
        num_edge_points=300,
        num_poles=0,
        deg_poly=24,
        spacing="linear",
    )
    rs = [0.2, 0.2, 0.2]
    cs = [0.0 + 0.0j, -0.5 - 0.5j, +0.5 + 0.5j]
    for r, c in zip(rs, cs):
        prob.add_interior_curve(
            lambda t: c + r * np.exp(2j * np.pi * t), num_points=100,
            centroid=c,
        )
    assert len(prob.domain.interior_curves) == 3
    for i in range(3):
        assert len(prob.domain.interior_curves[i]) == 100
    assert 0.0 + 0.0j not in prob.domain
    assert -0.5 - 0.5j not in prob.domain
    assert 0.5 - 0.5j in prob.domain


if __name__ == "__main__":
    test_create_circles()
