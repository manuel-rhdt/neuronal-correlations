from .clustering import (apply_clustering, cluster_by_correlation,
                         cluster_randomly)
from .helpers import compute_correlation_coefficients, normalize_data


def perform_renormalization(data, times=1, clustering_strategy="correlation"):
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
        pairs = None
        if clustering_strategy == "correlation":
            correlation_matrix = compute_correlation_coefficients(newdata[-1])
            pairs = cluster_by_correlation(correlation_matrix)
        elif clustering_strategy == "random":
            pairs = cluster_randomly(len(newdata[-1]))
        else:
            raise "No known clustering strategy named '{}'.".format(
                clustering_strategy)

        newdata.append(apply_clustering(newdata[-1], pairs, normalize_data))

    return newdata
