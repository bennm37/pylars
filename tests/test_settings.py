"""Common values and functions to be used throughout the tests."""
ATOL, RTOL = 1e-12, 1e-6


def load_lid_driven_cavity(n, num_poles):
    """Load the lid driven cavity data."""
    from scipy.io import loadmat

    return loadmat(f"tests/data/lid_driven_cavity_n_{n}_np_{num_poles}.mat")


def show_diff(a, b, rtol=RTOL, atol=ATOL, reshape=None):
    """Show the difference between two arrays."""
    import numpy as np
    import matplotlib.pyplot as plt

    if reshape is not None:
        l, h = a.shape
        a = a.reshape(l // reshape, reshape * h)
        b = b.reshape(l // reshape, reshape * h)
    diff = np.abs(a - b) < RTOL * np.abs(b) + ATOL
    plt.imshow(diff)
