"""Fast calculation of coltheart's N."""
import numpy as np

from jellyfish import hamming_distance
from functools import partial
from multiprocessing import Pool, cpu_count

# N is symmetric, but also takes a lot of storage.


def calc_dist(x, Y, radii, function):
    """Sub for parallelization."""
    # Assume radius is sorted from large to small
    scores = np.zeros(len(radii))
    for b in Y:
        dist = function(x, b)
        scores += dist <= radii

    return (x, scores)


def n_subloop(X,
              Y,
              function=hamming_distance,
              radius=1,
              n_jobs=-1):
    """Calculate distance from each word in word_a to each word in word_b."""
    func = partial(calc_dist,
                   Y=Y,
                   radii=np.asarray(radius),
                   function=function)

    if n_jobs == -1:
        n_jobs = cpu_count()

    with Pool(n_jobs) as p:
        result = dict(p.map(func, X))

    return np.array([result[x] for x in X])
