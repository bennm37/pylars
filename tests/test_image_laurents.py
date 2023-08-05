"""Test adding image laurents."""


def test_mirror_laurents():
    """Test adding mirror laurents to Domain."""
    from pylars import Problem
    import numpy as np

    prob = Problem()
    corners = [1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j]
    prob.add_exterior_polygon(corners, spacing="linear", num_poles=0)
    R, centroid = 0.2, 0.75 + 0.75j
    circle = lambda t: R * np.exp(2j * np.pi * t) + centroid
    deg_laurent = 15
    prob.add_interior_curve(
        circle,
        deg_laurent=deg_laurent,
        centroid=centroid,
        mirror_laurents=True,
    )
    dom = prob.domain
    dom.plot(set_lims=False)
    assert np.allclose(
        np.array(dom.exterior_laurents)[dom.mirror_indices["4"]],
        np.array([[1.25 + 0.75j, deg_laurent], [0.75 + 1.25j, deg_laurent]]),
    )


def test_image_laurents():
    """Test adding image laurents to Domain."""
    from pylars import Problem
    import numpy as np

    prob = Problem()
    prob.add_periodic_domain(2, 2)
    R, centroid = 0.2, 0.75 + 0.75j
    circle = lambda t: R * np.exp(2j * np.pi * t) + centroid
    deg_laurent = 15
    prob.add_periodic_curve(
        circle,
        deg_laurent=deg_laurent,
        centroid=centroid,
        image_laurents=True,
    )
    dom = prob.domain
    dom.plot(set_lims=False)
    image_answer = np.array(
        [
            [-1.25 - 1.25j, deg_laurent],
            [0.75 - 1.25j, deg_laurent],
            [-1.25 + 0.75j, deg_laurent],
        ]
    )
    images = np.array(dom.exterior_laurents)[dom.image_indices["4"]]
    assert np.allclose(images, image_answer)


if __name__ == "__main__":
    test_mirror_laurents()
    test_image_laurents()
