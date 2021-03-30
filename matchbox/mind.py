import logging
from typing import Callable, Set, Type

import numpy as np
from matchbox.ind import Ind
from matchbox.tests import knn_test

from .gennext import gen_next
from .util.nooplistener import NoopListener

_logger = logging.getLogger(__name__)


class Mind(object):
    """
    Implement the MIND algorithm from De Marchi 2003

    Parameters
    ----------
    test : Callable
        Statistical test to use
    test_args : dict
    """

    def __init__(self, alpha: float = 0.05, test: Callable = knn_test, test_args: dict = None):
        self.__alpha = alpha
        self.__test = test
        self.__test_args = test_args if test_args else dict()

    def __call__(self, ind: Set[Ind], stop: int = np.inf,
                 progress_listener: Type = NoopListener) -> Set[Ind]:
        """
        Run the MIND algorithm

        Parameters
        ----------
        ind : set
            Initial set of Ind
        stop : int
            Stop at the given arity
        progress_listener : Type
            Type to instantiate for progress reporting (compatible with tqdm)

        Returns
        -------
        set :
            Positive inclusion dependencies
        """
        arity = next(iter(ind)).arity
        for i in ind:
            if i.arity != arity:
                raise ValueError(f'All input IND must have the same arity: {i.arity} vs {arity}')

        all_satisfied = set()
        while arity < stop and len(ind) > 0:
            arity += 1
            _logger.info('Generating candidates for %d-ind', arity)
            candidates = gen_next(ind)
            if not candidates:
                _logger.warning('Can not generate candidates for %d-ind', arity)
                break

            _logger.info('Validating %d candidates', len(candidates))

            next_ind = set()

            for candidate in progress_listener(candidates):
                p = self.__test(candidate.lhs.data, candidate.rhs.data, **self.__test_args)
                if p >= self.__alpha:
                    _logger.debug(f'Accepting {candidate} with p-value of {p} >= {self.__alpha}')
                    candidate.confidence = p
                    next_ind.add(candidate)
                else:
                    _logger.debug(f'Dropping {candidate} because the p-value {p} < {self.__alpha}')

            _logger.info('%d-ind: %d positive', arity, len(next_ind))
            all_satisfied.update(next_ind)
            ind = next_ind

        return all_satisfied
