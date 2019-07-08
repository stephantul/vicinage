"""Vector-based neighborhood metrics."""
import numpy as np

from sklearn.metrics import pairwise_distances


def rd_subloop(X, Y, metric, n):
    """Separate method because sometimes we want to analyze the matrix."""
    if X is Y:
        dists = pairwise_distances(X, metric=metric)
    else:
        dists = pairwise_distances(X, Y, metric=metric)

    return np.partition(dists, kth=n, axis=1)[:, :n]
