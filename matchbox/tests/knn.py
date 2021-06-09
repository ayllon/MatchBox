from typing import Union, Tuple

import numpy as np
import pandas
from scipy.spatial.ckdtree import cKDTree as KDTree


# noinspection PyPep8Naming
def knn_test(lhs_data: pandas.DataFrame, rhs_data: pandas.DataFrame, k: int = 5, n_perm: int = 100,
             return_T=False, return_N=False) -> Union[float, Tuple[float, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Runs the K-NN permutations test from
    "Multivariate two-sample tests based on nearest neighbors", Schilling 1986

    Parameters
    ----------
    lhs_data : pandas.DataFrame
        LHS data set
    rhs_data : pandas.DataFrame
        RHS data set
    k : int
        Number of neighbors
    n_perm : int
        Number of permutations for the test
    return_T : bool
        If True, return the test statistics T0 and T
    return_N : bool
        If True, return the nearest neighbor
    Returns
    -------
    p-value : float
        If return_T is False
    (p-value, T0, T) : (float, np.ndarray, np.ndarray)
        If return_T is True
    """
    # To numpy arrays
    if isinstance(lhs_data, pandas.DataFrame):
        lhs_data = lhs_data.dropna(axis=0, how='any').to_numpy()
    if isinstance(rhs_data, pandas.DataFrame):
        rhs_data = rhs_data.dropna(axis=0, how='any').to_numpy()
    # Concatenate all values and group assignment of each point
    points = np.concatenate([lhs_data, rhs_data])
    labels = np.concatenate([np.zeros(len(lhs_data)), np.ones(len(rhs_data))])
    N = len(points)
    # Tree for lookup
    tree = KDTree(points, compact_nodes=False, balanced_tree=False)
    # Permutation test
    _, idx = tree.query(points, k=k + 1)
    T0 = ((labels[idx] == labels[:, np.newaxis]).sum(axis=1) - 1).sum() / (k * N)
    Tj = np.zeros(n_perm)
    for i in range(n_perm):
        np.random.shuffle(labels)
        Tj[i] = ((labels[idx] == labels[:, np.newaxis]).sum(axis=1) - 1).sum() / (k * N)
    # One tail test
    p = 1 - ((T0 > Tj).sum()) / n_perm

    returns = [p]
    if return_T:
        returns.extend([T0, Tj])
    if return_N:
        returns.append(idx[:, 1])

    if len(returns) == 1:
        return returns[0]
    else:
        return tuple(returns)
