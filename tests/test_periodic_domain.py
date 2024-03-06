"""Test adding curves over periodic domain boundaries."""


def test_remove_points():
    """Test the domain.remove method."""
    from pylars import Domain
    import numpy as np

    corners = [0, 1, 1 + 1j, 1j]
    dom = Domain()
    dom.add_exterior_polygon(corners, num_edge_points=100, spacing="linear")
    dom.remove(np.array(range(75, 125)))
    assert np.all(dom.indices["0"] == np.array(np.array(range(75))))
    assert np.all(dom.indices["1"] == np.array(np.array(range(75, 150))))
    assert np.all(dom.indices["2"] == np.array(np.array(range(150, 250))))
    assert np.all(dom.indices["3"] == np.array(np.array(range(250, 350))))
    dom.plot()
    # plt.show()


def test_remove_points_2():
    """Test the domain.remove method."""
    from pylars import Domain
    import numpy as np

    corners = [-1, 1, 1 + 1j, 1j]
    dom = Domain()
    dom.add_exterior_polygon(corners, num_edge_points=10, spacing="linear")
    dom.remove(np.array(range(16, 28, 2)))
    assert len(dom.boundary_points) == 34
    assert np.all(dom.indices["0"] == np.array(np.array(range(10))))
    assert np.all(dom.indices["1"] == np.array(np.array(range(10, 18))))
    assert np.all(dom.indices["2"] == np.array(np.array(range(18, 24))))
    assert np.all(dom.indices["3"] == np.array(np.array(range(24, 34))))
    dom.plot()


def test_get_nnic():
    """Test adding a circle over a corner."""
    from pylars import PeriodicDomain
    import numpy as np
    from collections import Counter
    import warnings

    warnings.filterwarnings("error", category=DeprecationWarning)
    with warnings.catch_warnings():
        dom = PeriodicDomain()
        dom.add_periodic_domain(2, 2)
        R = 0.5
        centroid = 0.75 + 0.75j
        circle = lambda t: centroid + R * np.exp(2j * np.pi * t)
        dom.add_periodic_curve(circle, centroid)
        assert dom.periodic_curves[0] == "4"
        nnic = dom.get_nn_image_centroids(centroid)
        assert Counter(nnic) == Counter(
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


def test_generate_periodic_curve():
    """Test adding a circle over a corner."""
    from pylars import PeriodicDomain
    import matplotlib.pyplot as plt
    import numpy as np

    dom = PeriodicDomain()
    dom.add_periodic_domain(2, 2, num_edge_points=300)
    R = 0.5
    centroid = 0.75 + 0.75j
    circle = lambda t: centroid + R * np.exp(2j * np.pi * t)
    dom.add_periodic_curve(circle, centroid, num_points=200)
    centroid = 0.0 - 0.9j
    circle = lambda t: centroid + 0.2 * np.exp(2j * np.pi * t)
    dom.add_periodic_curve(circle, centroid)
    plt.close("all")
    fig, ax = dom.plot(set_lims=False)
    # plt.show()
    assert np.allclose(
        dom.boundary_points[dom.indices["0"]].real,
        dom.boundary_points[dom.indices["2"]][::-1].real,
    )
    assert np.allclose(
        dom.boundary_points[dom.indices["1"]].imag,
        dom.boundary_points[dom.indices["3"]][::-1].imag,
    )


def test_flow_periodic_curve():
    """Test solving flow past a circle over a corner."""
    from pylars import Problem, Solver, Analysis
    import matplotlib.pyplot as plt
    import numpy as np

    prob = Problem()
    prob.add_periodic_domain(2, 2, num_edge_points=300)
    R = 0.5
    centroid = 0.75 + 0.75j
    circle = lambda t: centroid + R * np.exp(2j * np.pi * t)
    prob.add_periodic_curve(circle, centroid, num_points=200)
    R = 0.2
    # centroid = -0.2 - 0.9j
    # circle = lambda t: centroid + R * np.exp(2j * np.pi * t)
    # prob.add_periodic_curve(circle, centroid, num_points=200)
    prob.domain.plot(set_lims=False)
    prob.add_boundary_condition("0", "u[0]-u[2][::-1]", 0)
    prob.add_boundary_condition("0", "v[0]-v[2][::-1]", 0)
    prob.add_boundary_condition("2", "p[0]-p[2][::-1]", 0)
    prob.add_boundary_condition("2", "e12[0]-e12[2][::-1]", 0)
    prob.add_boundary_condition("1", "u[1]-u[3][::-1]", 0)
    prob.add_boundary_condition("1", "v[1]-v[3][::-1]", 0)
    prob.add_boundary_condition("3", "p[1]-p[3][::-1]", 2)
    prob.add_boundary_condition("3", "e12[1]-e12[3][::-1]", 0)
    prob.add_boundary_condition("4", "u[4]", 0)
    prob.add_boundary_condition("4", "v[4]", 0)
    # prob.add_boundary_condition("5", "u[5]", 0)
    # prob.add_boundary_condition("5", "v[5]", 0)
    solver = Solver(prob, verbose=False)
    sol = solver.solve(normalize=False, weight=False)

    print(f"Error: {solver.max_error}")
    an = Analysis(sol)
    # fig, ax = an.plot(resolution=301, streamline_type="linear")
    # plt.show()
    # fig, ax = an.plot_periodic(resolution=501)
    # plt.show()


if __name__ == "__main__":
    test_remove_points()
    test_remove_points_2()
    test_get_nnic()
    test_generate_periodic_curve()
    test_flow_periodic_curve()
