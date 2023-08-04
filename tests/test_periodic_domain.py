"""Test adding curves over periodic domain boundaries."""


def test_remove_points():
    """Test the domain.remove method."""
    from pylars import Domain
    import matplotlib.pyplot as plt
    import numpy as np

    corners = [0, 1, 1 + 1j, 1j]
    dom = Domain(corners, num_edge_points=100, spacing="linear")
    dom.remove(np.array(range(75, 125)))
    assert np.all(dom.indices["0"] == np.array(np.array(range(75))))
    assert np.all(dom.indices["1"] == np.array(np.array(range(75, 150))))
    assert np.all(dom.indices["2"] == np.array(np.array(range(150, 250))))
    assert np.all(dom.indices["3"] == np.array(np.array(range(250, 350))))
    dom.plot()
    plt.show()


def test_remove_points_2():
    """Test the domain.remove method."""
    from pylars import Domain
    import numpy as np
    import matplotlib.pyplot as plt

    corners = [-1, 1, 1 + 1j, 1j]
    dom = Domain(corners, num_edge_points=10, spacing="linear")
    dom.remove(np.array(range(16, 28, 2)))
    assert len(dom.boundary_points) == 34
    assert np.all(dom.indices["0"] == np.array(np.array(range(10))))
    assert np.all(dom.indices["1"] == np.array(np.array(range(10, 18))))
    assert np.all(dom.indices["2"] == np.array(np.array(range(18, 24))))
    assert np.all(dom.indices["3"] == np.array(np.array(range(24, 34))))
    dom.plot()
    plt.show()


def test_get_nnic():
    """Test adding a circle over a corner."""
    from pylars import PeriodicDomain
    import matplotlib.pyplot as plt
    import numpy as np
    from collections import Counter

    dom = PeriodicDomain(2, 2)
    dom.plot()
    plt.show()
    R = 0.5
    centroid = 0.75 + 0.75j
    circle = lambda t: centroid + R * np.exp(2j * np.pi * t)
    dom.add_periodic_curve(circle)
    assert dom.periodic_curves[0] == "4"
    dom.get_nn_image_centroids(centroid)
    assert Counter(dom.nnic) == Counter(
        [
            centroid - 2,
            centroid - 2 - 2j,
            centroid - 2 + 2j,
            centroid,
            centroid - 2j,
            centroid + 2j,
            centroid + 2,
            centroid + 2 - 2j,
            centroid + 2 + 2j,
        ]
    )
    assert len(dom.indices["0"]) == len(dom.indices["2"])
    assert len(dom.indices["1"]) == len(dom.indices["3"])
    fig, ax = dom.plot()
    plt.show()


def test_generate_periodic_curve():
    """Test adding a circle over a corner."""
    from pylars import PeriodicDomain
    import matplotlib.pyplot as plt
    import numpy as np
    from collections import Counter

    dom = PeriodicDomain(2, 2)
    R = 0.5
    # centroid = 0.75 + 0.75j
    # circle = lambda t: centroid + R * np.exp(2j * np.pi * t)
    # dom.add_periodic_curve(circle, centroid)
    centroid = 0.0 - 0.9j
    circle = lambda t: centroid + 0.01 * np.exp(2j * np.pi * t)
    dom.add_periodic_curve(circle, centroid)
    fig, ax = dom.plot()
    plt.show()


if __name__ == "__main__":
    # test_remove_points()
    # test_remove_points_2()
    # test_get_nnic()
    test_generate_periodic_curve()
