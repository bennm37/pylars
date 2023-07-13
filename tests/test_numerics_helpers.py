"""Test the helper numerical functions in pylars.numerics."""
from test_settings import ATOL, RTOL


def test_cluster():
    """Test the lightning poles clustering function."""
    import numpy as np
    from pylars.numerics import cluster
    from scipy.io import loadmat

    cluster_10_answer = loadmat("tests/data/cluster_10.mat")["cluster_10"]
    cluster_10 = cluster(10, 1, 4)
    assert np.allclose(cluster_10, cluster_10_answer, atol=ATOL, rtol=RTOL)


def test_split():
    """Test the splitting of coefficients."""
    import numpy as np
    from pylars.numerics import split

    coefficients = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    cf_answer = np.array([0 + 4j, 1 + 5j])
    cg_answer = np.array([2 + 6j, 3 + 7j])
    cf, cg = split(coefficients)
    assert np.allclose(cf, cf_answer, atol=ATOL, rtol=RTOL)
    assert np.allclose(cg, cg_answer, atol=ATOL, rtol=RTOL)


def test_split_laurent():
    """Test the splitting of coefficients for laurent basis."""
    import numpy as np
    from pylars.numerics import split_laurent

    coeff = [0, 0, 0, 1, 1, 1, 2, 3, 4, 4, 4, 5, 5, 5, 6, 7]
    laurents = [(0 + 0.0j, 10)]
    cf, cg, clf, clg = split_laurent(coeff, laurents)
    assert np.allclose(cf, np.array([0 + 4j, 0 + 4j, 0 + 4j]))
    assert np.allclose(clf, np.array([2 + 6j]))
    assert np.allclose(cg, np.array([1 + 5j, 1 + 5j, 1 + 5j]))
    assert np.allclose(clg, np.array([3 + 7j]))


def test_lid_driven_cavity_split():
    import numpy as np
    from pylars.numerics import split
    from scipy.io import loadmat

    n = 24
    num_poles = 24
    test_answers = loadmat(
        f"tests/data/lid_driven_cavity_n_{n}_np_{num_poles}.mat"
    )
    c = test_answers["c"]
    cf_answer = test_answers["cf"]
    cg_answer = test_answers["cg"]
    cf, cg = split(c)
    assert np.allclose(cf, cf_answer, atol=ATOL, rtol=RTOL)
    assert np.allclose(cg, cg_answer, atol=ATOL, rtol=RTOL)


if __name__ == "__main__":
    test_cluster()
    test_split()
    test_split_laurent()
    test_lid_driven_cavity_split()
