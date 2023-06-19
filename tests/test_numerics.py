def test_cluster():
    import numpy as np
    from pyls.numerics import cluster
    from scipy.io import loadmat

    cluster_10_answer = loadmat("tests/data/cluster_10.mat")["cluster_10"]
    cluster_10 = cluster(10, 1, 4)
    assert np.allclose(cluster_10, cluster_10_answer)
