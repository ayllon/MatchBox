import logging

import numpy as np
import pandas
from intervaltree import IntervalTree
from scipy.stats import ks_2samp

from .attributeset import AttributeSet
from .ind import Ind
from .util.nooplistener import NoopListener

_logger = logging.getLogger(__name__)


class UIntersectFinder(object):
    """
    Find unary intersection dependencies. It is inspired on the algorithm
    published by Fabien De Marchi in 2002: https://link.springer.com/chapter/10.1007/3-540-45876-X_30

    This one, however, uses an interval tree instead of the inverted index, so it works for
    floating point numbers, where it is hard to find exact inclusion dependencies due to precision errors,
    and because measurements recorded in floating point likely have uncertainties associated with them.

    Parameters
    ----------
    method : str
        Statistic test: 'ks' for Kolmogorov-Smirnov
    """

    def __init__(self, method: str = 'ks'):
        self.__representativity = dict()
        self.__tree = IntervalTree()
        if method == 'ks':
            self.__method = self._ks
        else:
            raise ValueError(f'Unknown method {method}')

    def _ks(self, x_a, x_b):
        """
        Kolmogorov-Smirnov test
        """
        statistics, pvalue = ks_2samp(x_a, x_b, alternative='two-sided', mode='auto')
        return pvalue

    def add(self, relation_name: str, dataset: pandas.DataFrame):
        """
        Add columns from a new data set into the intersect finder. Note that only
        numerical attributes are supported. For categorical data, De Marchi original algorithm could be used.

        Parameters
        ----------
        relation_name : str
            Assign a name to the dataset for readability. Note that duplicated names are *not* accepted.
        dataset : pandas.DataFrame
            Dataset to add. It can be a sample, or this method can re-sample if needed.
        """
        if relation_name in self.__representativity:
            raise KeyError(f'{relation_name} already added')

        self.__representativity[relation_name] = len(dataset)

        for colname in dataset:
            if not np.issubdtype(dataset[colname].dtype, np.number):
                _logger.debug(f'Ignoring {colname} because it is not numerical')
                continue
            if np.issubdtype(dataset[colname].dtype, np.integer) and len(np.unique(dataset[colname])) < 10:
                _logger.debug(f'Ignoring {colname} because it is an integer with less than 10 values')
                continue

            with pandas.option_context('mode.use_inf_as_na', True):
                points = dataset[colname].dropna(axis=0, inplace=False)
            min_val, max_val = points.min(), points.max()
            if min_val >= max_val:
                _logger.debug(f'Ignoring {colname} because it has an empty range')
                continue
            attr_id = AttributeSet(relation_name, colname, dataset)
            self.__tree[min_val:max_val] = (attr_id, points)

    # noinspection PyPep8Naming
    def __call__(self, alpha=0.05, no_symmetric=False, progress_listener=NoopListener):
        """
        Run the modified algorithm over the interval tree

        Parameters
        ----------
        alpha : float
            Rejection level for the statistical test
        progress_listener : type
            An iterable type that can be constructed receiving another iterator. It can be used to report
            progress back to the user (i.e. tqdm)

        Returns
        -------
            A set of pair of Attributes, where the left one is contained on the second one
        """
        # All elements
        U = set()
        for interval in self.__tree:
            U = U.union(interval.data[0])
        # Candidates
        A_rhs = dict()
        confidence = dict()
        for A in U:
            A_rhs[A] = dict()
            confidence[A] = dict()
            for B in U:
                if A.relation_name != B.relation_name:
                    A_rhs[A][B] = 0
        # Intersect
        ntests = 0
        for interval in progress_listener(sorted(self.__tree.all_intervals)):
            A, Av = interval.data
            overlapping = self.__tree.overlap(interval.begin, interval.end)
            for overlap in overlapping:
                B, Bv = overlap.data
                # Skip self-intersection
                if A.relation_name == B.relation_name:
                    continue

                ntests += 1
                pvalue = self.__method(Av, Bv)
                if pvalue < alpha:
                    _logger.debug(f'Statistic check discards {A} ⊆ {B}')
                else:
                    A_rhs[A][B] += 1
                    confidence[A][B] = pvalue

        worst_nstest = sum(map(len, A_rhs.values()))
        if worst_nstest:
            savings = ((worst_nstest - ntests) / worst_nstest) * 100
            _logger.info(f'{ntests} statistical tests done ({worst_nstest} worst case, saved {savings:.2f}%)')
        # Find those within the threshold
        AI = set()
        for A in sorted(U):
            for B, nab in A_rhs[A].items():
                if nab and (not no_symmetric or Ind(B, A) not in AI):
                    AI.add(Ind(A, B, confidence[A][B]))
        return frozenset(AI)
