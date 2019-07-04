import numpy as np
from PIL import Image

EPSILON = 1e-7


def normalize_data(data):
    """ Normalize data according to the paper.
    """
    norm = np.mean(data[data > EPSILON], axis=-1)
    return data / norm


def binarize_data(data):
    """ Make the data binary.
    """
    return (data > EPSILON).astype(float)


def load_data(filename, norm=normalize_data):
    """ Loads the data from a TIFF file and returns a normalized numpy array.
    """
    im = Image.open(filename)
    return norm(np.array(im))


def generate_random_filtered_data(size, num_neurons=1000, clip_to=0.995):
    unfiltered_data = np.reshape(np.random.random(
        size * num_neurons), (num_neurons, size))
    return normalize_data(np.clip(unfiltered_data, clip_to, 1.0) - clip_to)


def compute_correlation_coefficients(data):
    """ Returns a matrix of correlation coefficients from the given data.
    """
    return np.corrcoef(data)


def p_zero(data):
    zero_elems = data <= EPSILON
    fraction = np.sum(zero_elems, axis=-1) / np.size(zero_elems, axis=-1)
    return np.mean(fraction)


def p_zero_err(data):
    zero_elems = data <= EPSILON
    fraction = np.sum(zero_elems, axis=-1) / np.size(zero_elems, axis=-1)
    return np.std(fraction)


def make_histogram(data):
    """ Make a histogram from the (possibly renormalized) data.

    Returns a list containing two arrays:
    - the first array contains the center x-values of the bins, i.e. the x-axis
      of the data
    - the second array contains the histogram height, i.e. the y-axis

    The returned distribution's integral is normalized to 1. This corresponds
    to the value Q_K(x) from the paper.
    """
    y, x = np.histogram(data, bins='auto', density='true',
                        range=(EPSILON, data.max()))
    # compute the mid points of the bin edges
    x = (x[:-1] + x[1:]) / 2
    return np.array([x, y])


def compute_p_trajectory(renormalized_data):
    """ Computes the values of `P_0` (notation from the paper) for the data
    at successive RG steps. The data must already be renormalized (i.e. use
    the output of `perform_renormalization`).

    This function returns the values used to create the plot on the left of
    fig. 3.

    Returns a tuple (values, errors, cluster_sizes)
    """
    vals = np.array([p_zero(rg_data) for rg_data in renormalized_data])
    errors = np.array([p_zero_err(rg_data) for rg_data in renormalized_data])
    cluster_sizes = 2 ** np.arange(len(vals))
    return (vals, errors, cluster_sizes)


def correlation_matrix_eigenvalues_sorted(data):
    """ Computes the matrix of correlation coefficients from data and returns
    its eigenvalues sorted in decreasing order.
    """
    corr_coef = compute_correlation_coefficients(data)
    eigen_vals, _ = np.linalg.eig(corr_coef)
    return -np.sort(-eigen_vals)
