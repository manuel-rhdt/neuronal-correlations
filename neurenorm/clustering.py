""" This file contains the different clustering routines.
"""

import numpy as np

from .helpers import compute_correlation_coefficients, normalize_data


def cluster_by_correlation(correlation_matrix):
    """ Performs a single renormalization step (clustering neurons into pairs).
    The clustering is performed based on correlation. Therefore this function
    correlation matrix as an input to compute the clustering.

    Returns `[i, j]` where `i` and `j` are both arrays of indices of length
    `n//2`. These indicate which neurons should be clustered together. For
    example the first cluster is formed by neuron `i[0]` and `j[0]` and the
    second one by `i[1]` and `j[1]`.
    """

    # these definitions are just for convenience
    n = len(correlation_matrix)
    n_half = n // 2

    # subtract the identity matrix since the diagonal of the correlation matrix
    # contains only 1's.
    correlation_matrix = correlation_matrix - \
        np.identity(len(correlation_matrix))

    # we arg sort the negative correlation matrix to get the biggest element
    # first. This works because all entries in the correlation matrix range
    # from 0 to 1.
    #
    # `indices` is a sorted list of indices of the most correlated pairs. In
    # other words, indices[0] is the index of the largest element of the
    # correlation matrix.
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
                # we are finished after having found n_half pairs
                break

    return most_correlated_pairs


def cluster_randomly(data_length):
    """Performs one step of random clustering and returns the clustering
    indices. `data_length` is the number of neurons in the data.

    Returns `[i, j]` where `i` and `j` are both arrays of indices of length
    `n//2`. These indicate which neurons should be clustered together. For
    example the first cluster is formed by neuron `i[0]` and `j[0]` and the
    second one by `i[1]` and `j[1]`.
    """

    clustering_pairs = np.zeros((2, data_length//2), dtype=int)
    indices = list(range(data_length))

    for pair in range(data_length//2):
        i = indices.pop(np.random.randint(0, len(indices)))
        j = indices.pop(np.random.randint(0, len(indices)))
        clustering_pairs[:, pair] = [i, j]

    return clustering_pairs


def cluster_pairs(data, strategy='correlation'):
    """ Performs a single step of clustering choosing the correct clustering function
    """
    if strategy == 'correlation':
        corr_coef = compute_correlation_coefficients(data)
        return cluster_by_correlation(corr_coef)
    elif strategy == 'random':
        return cluster_randomly(len(data))
    else:
        raise Exception('unknown clustering strategy "{}"'.format(strategy))


def apply_clustering(data, clustering_indices, norm=normalize_data):
    """Returns renormalized data using the given clustering."""
    num_clusters = np.size(clustering_indices, 1)
    rval = np.zeros((num_clusters, np.size(data, 1)))
    for i in clustering_indices:
        rval += data[i]
    return norm(rval)


def cluster_recursive(data,
                      num_clustering_steps,
                      clustering_strategy='correlation',
                      norm=normalize_data):
    if num_clustering_steps == 0:
        return [np.array([np.arange(len(data))])]

    # `clusters` is a matrix where the columns indicate which neurons form a cluster
    cluster_matrices = cluster_recursive(
        data, num_clustering_steps - 1, clustering_strategy, norm)
    clusters = cluster_matrices[-1]

    # now we generate the clustered data and compute the most correlated pairs
    clustered_data = apply_clustering(data, clusters, norm)
    pairs = cluster_pairs(clustered_data, clustering_strategy)

    # Here we use the previous `clusters` array and merge them according to `pairs`.
    # This was complicated to write but seems to work fine. I admit I don't completely understand
    # every aspect of this code.
    with np.nditer([pairs, np.arange(len(clusters)), None],
                   op_axes=[[-1, 0, 1], [0, -1, -1], None]) as itr:
        for a, b, out in itr:
            out[...] = clusters[b, a]
        out = itr.operands[2]
        out.shape = (out.shape[0] * out.shape[1], out.shape[2])
        cluster_matrices.append(out)
        return cluster_matrices
