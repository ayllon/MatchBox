import logging
from typing import Callable, FrozenSet

import numpy as np

from matchbox.ind import Ind
from matchbox.tests import knn_test
from matchbox.zigzag import node_to_ind

_logger = logging.getLogger(__name__)


def verify_mergers(nind: FrozenSet[Ind], alpha: float = 0.01, test: Callable = knn_test, test_args: dict = None):
    """
    Verify potential merges of n-IND, as per:
    "Integration of heterogeneous databases: Discovery of meta-information and maintenance of schema-restructuring views",
    Koller 2002

    Parameters
    ----------
    nind : FrozenSet[Ind]
        Output from an IND searching algorithm
    alpha : float
        Rejection level of the H0 hypothesis: samples come from the same population.
        If H0 is rejected, then an inclusion dependency is considered to *not* be satisfied.
    test : Callable
        Statistic test to use
    test_args : dict
        Arguments to pass down to the statistic test

    Returns
    -------
    out : Ind
        Maximal valid IND
    """
    if not test_args:
        test_args = dict()

    nind_list = list(nind)
    arities = list(map(lambda s: s.arity, nind_list))
    ordered = np.flip(np.argsort(arities))

    max_set = nind_list[ordered[0]].get_all_unary()
    for i in range(1, len(ordered)):
        candidate_set = max_set.union(nind_list[ordered[i]].get_all_unary())
        candidate = node_to_ind(candidate_set)
        p = test(candidate.lhs.data, candidate.rhs.data, **test_args)
        if p >= alpha:
            _logger.info('Accepted %d-ind %s with p-value %.2f', candidate.arity, str(candidate), p)
            max_set = candidate_set
        else:
            break

    return node_to_ind(max_set)
