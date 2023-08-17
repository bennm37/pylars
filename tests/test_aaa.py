"""Test the AAA rational approximation function from numerics.py."""
ATOL, RTOL = 1e-10, 1e-1


def test_gamma_aaa():
    """Test AAA against the gamma example from the paper."""
    from scipy.io import loadmat
    import numpy as np
    from pylars.numerics import aaa

    test_answers = loadmat("tests/data/gamma_aaa.mat")
    z = test_answers["z"]
    f = test_answers["f"]
    points = test_answers["points"].flatten()
    r_points_answer = test_answers["r_points"].flatten()
    poles_answers = test_answers["pol"].flatten()
    residues_answers = test_answers["res"].flatten()
    zeros_answers = test_answers["zer"].flatten()

    r, poles, residues, zeros = aaa(f, z)
    r_points = r(points)
    assert np.allclose(r_points_answer, r_points, atol=ATOL, rtol=RTOL)
    assert np.allclose(poles, poles_answers, atol=ATOL, rtol=RTOL)
    assert np.allclose(residues, residues_answers, atol=ATOL, rtol=RTOL)
    assert np.allclose(zeros, zeros_answers, atol=ATOL, rtol=RTOL)


def test_tan_aaa():
    """Test AAA against the gamma example from the paper."""
    from scipy.io import loadmat
    import numpy as np
    from pylars.numerics import aaa

    test_answers = loadmat("tests/data/tan_aaa.mat")
    z = test_answers["z"]
    f = test_answers["f"]
    points = test_answers["points"].flatten()
    r_points_answer = test_answers["r_points"].flatten()
    poles_answers = test_answers["pol"].flatten()
    residues_answers = test_answers["res"].flatten()
    zeros_answers = test_answers["zer"].flatten()
    r, poles, residues, zeros = aaa(f, z)
    r_points = r(points)
    assert np.allclose(r_points_answer, r_points, atol=ATOL, rtol=RTOL)
    assert np.allclose(poles, poles_answers, atol=ATOL, rtol=RTOL)
    assert np.allclose(residues, residues_answers, atol=ATOL, rtol=RTOL)
    assert np.allclose(zeros, zeros_answers, atol=ATOL, rtol=RTOL)


# def test_ellipse_aaa():
# """Test AAA for an ellipse."""
# # TODO is this numerical error or is there something going
# # wrong here? The first 2 choices of points are the same,
# # then off by 1, off by 2 etc.
# from scipy.io import loadmat
# import numpy as np
# from pylars.numerics import aaa

# test_answers = loadmat("tests/data/ellipse_aaa.mat")
# z = test_answers["z"]
# f = test_answers["f"]
# points = test_answers["points"]
# r_points_answer = test_answers["r_points"]
# poles_answers = test_answers["pol"].flatten()
# residues_answers = test_answers["res"].flatten()
# zeros_answers = test_answers["zer"].flatten()
# r, poles, residues, zeros = aaa(f, z, tol=1e-13)
# r_points = r(points)
# assert np.allclose(r_points_answer, r_points, atol=ATOL, rtol=RTOL)
# assert np.allclose(poles, poles_answers, atol=ATOL, rtol=RTOL)
# assert np.allclose(residues, residues_answers, atol=ATOL, rtol=RTOL)
# assert np.allclose(zeros, zeros_answers, atol=ATOL, rtol=RTOL)


if __name__ == "__main__":
    test_gamma_aaa()
    test_tan_aaa()
    # test_ellipse_aaa()
