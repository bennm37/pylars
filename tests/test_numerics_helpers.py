"""Test the helper numerical functions in pyls.numerics."""


def test_cluster():
    """Test the lighting poles clustering function."""
    import numpy as np
    from pyls.numerics import cluster
    from scipy.io import loadmat

    cluster_10_answer = loadmat("tests/data/cluster_10.mat")["cluster_10"]
    cluster_10 = cluster(10, 1, 4)
    assert np.allclose(cluster_10, cluster_10_answer)


def test_split():
    """Test the splitting of coefficients."""
    import numpy as np
    from pyls.numerics import split
    from scipy.io import loadmat

    coefficients = loadmat("tests/data/coefficients.mat")["coefficients"]
    cf_answer = loadmat("tests/data/cf.mat")["cf"]
    cg_answer = loadmat("tests/data/cg.mat")["cg"]
    cf, cg = split(coefficients)
    assert np.allclose(cf, cf_answer)
    assert np.allclose(cg, cg_answer)


if __name__ == "__main__":
    test_cluster()
    test_split()
