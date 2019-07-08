"""Radius-based distance."""
import numpy as np
from sklearn.metrics import pairwise_distances


def r_subloop(X,
              Y,
              radius=1,
              function="cosine"):
    """Calculate distance from each word in word_a to each word in word_b."""
    dist_mtr = pairwise_distances(X, Y, metric=function)
    results = []
    for x in radius:
        results.append(np.sum(dist_mtr < x, 1))

    return np.stack(results, 1)
