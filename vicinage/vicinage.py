"""Calculate the neighborhood."""
import numpy as np
from old20 import old_subloop
from rd import rd_subloop
from functools import partial
from n import n_subloop
from rn import r_subloop


def _check_args(X, Y, n):
    X = np.asarray(X)
    was_int = False
    if Y is None:
        Y = X
    else:
        Y = np.asarray(Y)
    if isinstance(n, int):
        n = [n]
        was_int = True
    n = np.asarray(n)
    if np.any(n <= 0):
        raise ValueError("n should be positive.")
    if np.any([x > X.shape[0]-1 for x in n]):
        raise ValueError("Your n was bigger than the number of words - 1.")
    if np.unique(X, axis=1) != len(X):
        raise ValueError("There are duplicates in your dataset. Please remove"
                         " these, as they will make your estimates unreliable")
    return X, Y, n, was_int


def radius(function,
           X,
           Y=None,
           radius=None,
           memory_safe=False,
           **kwargs):
    X, Y, radius, was_int = _check_args(X, Y, radius)
    largest_n = max(radius)
    vals = function(X, Y, largest_n, **kwargs)
    vals = np.sort(vals, axis=1)

    out = []
    for x in radius:
        out.append(vals[:, :x+1].sum(1))

    if was_int:
        out = vals[0]
    return out


def neighborhood(function,
                 X,
                 Y=None,
                 n=20,
                 memory_safe=False,
                 **kwargs):
    """
    Calculate the representation density for values of n.

    n can either be an int or a list of ints

    If only X is given, the density will be computed based on X * X
    if Y is also given, the density will be computed based on X * Y

    Parameters
    ----------
    X : np.array
        The representations for which to calculate the density
    Y : np.array
        The reference representations to use
    n : int or list of int
        The number of neighbors to take into account.
    memory_safe : bool
        Enables memory safe mode for some metrics. This makes the metrics
        much slower, but does make them fit in memory.
    metric : str or callable
        The string of the metric name or callable.

    Returns
    -------
    densities : np.array
        A vector containing the density for each item.

    """
    X, Y, n, was_int = _check_args(X, Y, n)

    largest_n = max(n)
    vals = function(X, Y, largest_n, **kwargs)
    vals = np.sort(vals, axis=1)

    out = []
    for x in n:
        out.append(vals[:, :x+1].sum(1))

    if was_int:
        out = vals[0]
    return out


old = partial(neighborhood, function=old_subloop)
rd = partial(neighborhood, function=rd_subloop)
n = partial(radius, function=n_subloop)
r = partial(radius, function=r_subloop)
