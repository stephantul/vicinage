"""Calculate the OLD score for a given n."""
import numpy as np

from jellyfish import levenshtein_distance
from functools import partial
from multiprocessing import Pool, cpu_count


def calc_dist(x, Y, n, function):
    """Sub for parallelization."""
    max_idx = 0
    max_val = np.inf
    row = np.zeros(n+1) + np.inf
    for b in Y:
        dist = function(x, b)
        if max_val > dist:
            row[max_idx] = dist
            m = np.argmax(row)
            max_idx = m
            max_val = row[m]

    return (x, row)


def calc_all(x, Y, n, function):
    """Used as a shortcut if n == length of corpus."""
    row = np.zeros(len(Y))
    for idx, b in enumerate(Y):
        row[idx] = function(x, b)
    return (x, row)


def old_subloop(X,
                Y,
                n=20,
                function=levenshtein_distance,
                n_jobs=-1):
    """Calculate distance from each word in word_a to each word in word_b."""
    if n >= len(Y) - 1:
        job_func = calc_all
    else:
        job_func = calc_dist
    func = partial(job_func, Y=Y, n=n, function=function)

    if n_jobs == -1:
        n_jobs = cpu_count()

    with Pool(n_jobs) as p:
        result = dict(p.map(func, X))

    return np.array([result[x] for x in X])
