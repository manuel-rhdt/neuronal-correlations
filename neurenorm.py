import math
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


def renormalization_step(data, correlation_matrix, norm=normalize_data):
    """ returns the renormalized data
    """

    assert len(data) == len(correlation_matrix)

    n = len(data)
    n_half = math.floor(n / 2)

    # subtract the identity matrix since the diagonal of the correlation matrix
    # contains only 1's.
    correlation_matrix = correlation_matrix - \
        np.identity(len(correlation_matrix))

    # we sort the negative correlation matrix to get the biggest element first.
    # This works because all entries in the correlation matrix range from 0 to 1.
    #
    # `indices` contains the sorted flattened indices of the most correlated pairs.
    indices = np.argsort(-correlation_matrix, axis=None)

    # a list of bool's to keep track of the visited neurons
    visited_neurons = np.zeros(n, dtype=bool)
    most_correlated_pairs = np.zeros((2, n_half), dtype=int)
    len_pairs = 0

    # only look at the upper triangular elements of the correlation matrix
    i, j = np.unravel_index(indices, correlation_matrix.shape)
    i_gt_j = i > j
    i = i[i_gt_j]
    j = j[i_gt_j]

    for i, j in zip(i, j):
        if not visited_neurons[i] and not visited_neurons[j]:
            visited_neurons[i] = True
            visited_neurons[j] = True
            most_correlated_pairs[:, len_pairs] = [i, j]
            len_pairs += 1
            if len_pairs >= n_half:
                break

    i = most_correlated_pairs[0]
    j = most_correlated_pairs[1]

    renormalized_data = norm(data[i] + data[j])

    return renormalized_data


def perform_renormalization(data, times=1):
    """ Succesively renormalizes the `data` according to the method described
    in the paper. The number of RG steps performed is `times`.

    Returns a list of data matrices where the first element is simply the
    original data and the following elements are
    - data after one RG step,
    - data after two RG steps,
    - ...

    The returned list therefore is of length `times + 1`.
    """
    newdata = [data]
    for _ in range(0, times):
        correlation_matrix = compute_correlation_coefficients(newdata[-1])
        newdata.append(renormalization_step(newdata[-1], correlation_matrix))
    return newdata


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


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Compute correlations and perform renormalization.')
    parser.add_argument('input')
    args = parser.parse_args()

    data = load_data(args.input)
    print(compute_p_trajectory(perform_renormalization(data, times=8)))


if __name__ == "__main__":
    main()
