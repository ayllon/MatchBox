"""
Implementation of the ZigZag n-IND algorithm.

Zigzag: a new algorithm for mining large inclusion dependencies in databases, Fabien De Marchi, Jean-Marc Petit 2003

Based on the implementation of Metanome with some adaptations:
https://github.com/HPI-Information-Systems/inclusion-dependency-algorithms/blob/master/adp-algorithms/zigzag/src/main/java/de/metanome/algorithms/zigzag/Zigzag.java
"""
import itertools
import logging
from typing import Set, FrozenSet, Tuple, Callable, Iterable

from .ind import Ind, node_to_ind
from .mind import Mind
from .tests import knn_test

_logger = logging.getLogger(__name__)


def ind_to_nodes(ind: Iterable[Ind]) -> Set[FrozenSet[Ind]]:
    """
    Convert a set of inclusion dependencies into a set of set of unary inclusion dependencies
    Examples
    --------
    {(R.A, R.B  ⊆ S.A, S.B), (R.C, R.D ⊆ S.C, S.D)} => {{(R.A ⊆ S.A), (R.B ⊆ S.B)}, {(R.C ⊆ S.C), (R.D ⊆ S.D)}}
    """
    return set(map(frozenset, map(Ind.get_all_unary, ind)))


# noinspection PyTypeChecker,PyPep8Naming
def calculate_optimistic_border(unary: FrozenSet[Ind], negative: Iterable[Ind], n: int) -> Set[FrozenSet[Ind]]:
    """
    Compute the optimistic border.

    Theorem 1 Zigzag from De Marchi 2003
    Algorithm adapted from Metanome

    Parameters
    ----------
    unary : set of unary inclusion dependencies
    negative : set on n-ary inclusion dependencies known to *not* be satisfied
    n: arity

    Returns
    -------
    A set of set of unary inclusion dependencies from the optimistic border
    """
    N = len(unary)
    unsatisfied_nodes = ind_to_nodes(negative)
    S = set()
    for head in unsatisfied_nodes:
        I = unary.difference(head)
        removed = frozenset(filter(I.issuperset, S))
        S.difference_update(removed)

        add_to_s = set()
        for h in head:
            if len(removed) == 0:
                l = set()
                l.add(h)
                if all(map(l.isdisjoint, S)):
                    add_to_s.add(frozenset(l))
            else:
                for r_i in removed:
                    l = set(r_i)
                    l.add(h)
                    if not any(map(l.issuperset, S)):
                        add_to_s.add(frozenset(l))
        S.update(add_to_s)
    return frozenset(map(unary.difference, filter(lambda s: len(s) <= N - n, S)))


def get_candidates_next(pessimistic: Set[FrozenSet[Ind]], n: int) -> Set[FrozenSet[Ind]]:
    """
    Generate candidates of arity n from the pessimistic set
    """
    next_level = n + 1
    filtered = filter(lambda v: len(v) >= next_level, pessimistic)
    result = set()
    for x in filtered:
        for s in map(frozenset, itertools.permutations(x, next_level)):
            result.add(s)
    return result


def is_generalization(generalization: Set[Ind], specialization: Set[Ind]) -> bool:
    """
    Returns
    -------
    True if 'generalization' generalizes 'specialization' (it is a subset)
    """
    return len(generalization) < len(specialization) and specialization.issuperset(generalization)


def remove_generalizations(border: Set[FrozenSet[Ind]]) -> Set[FrozenSet[Ind]]:
    """
    Remove generalizations from the given border
    Examples
    --------
    Suppose the border contains {(R.{A, B, C} ⊆ S.{A, B, C}), (R.{A, B} ⊆ S.{A, B}) and (R.{B, C} ⊆ S.{B, C})},
    then the result will be {(R.{A, B, C} ⊆ S.{A, B, C})}
    """
    result = set(border)
    for i1, i2 in itertools.product(border, border):
        if is_generalization(i1, i2) and i1 in result:
            result.remove(i1)
    return result


def is_specialization(specialization: Set[Ind], generalization: Set[Ind]) -> bool:
    """
    Returns
    -------
    True if 'specialization' specializes 'generalization' (it is a superset)
    """
    return len(specialization) > len(generalization) and specialization.issuperset(generalization)


def remove_specializations(border: Set[FrozenSet[Ind]]) -> Set[FrozenSet[Ind]]:
    """
    Remove specializations from the given border

    Examples
    --------
    Suppose the border contains {(R.{A, B, C} ⊆ S.{A, B, C}), (R.{A, B} ⊆ S.{A, B}) and (R.{B, C} ⊆ S.{B, C})},
    then the result will be {(R.{A, B} ⊆ S.{A, B}) and (R.{B, C} ⊆ S.{B, C})}
    """
    result = set(border)
    for i1, i2 in itertools.product(border, border):
        if is_specialization(i1, i2) and i1 in result:
            result.remove(i1)
    return result


def get_unary_ind(ind: Set[Ind]) -> FrozenSet[Ind]:
    """
    Extract all unary ind that generalize the set of n-ary ind passed as an argument
    """
    unaries = set()
    for i in ind:
        unaries.update(i.get_all_unary())
    return frozenset(unaries)


def generate_candidates(n: int, unary: Set[Ind]) -> Set[Ind]:
    """
    Generate all possible candidates of arity n from the given set of unary ind
    """
    if len(unary) < n:
        return set()
    candidates = set()
    for p in itertools.permutations(unary, n):
        candidates.add(node_to_ind(p))
    return frozenset(candidates)


def check_nary_ind(n: int, unary: Set[Ind], input_ind: Set[Ind]) -> Tuple[Set[Ind], Set[Ind]]:
    """
    Generates all possible n-ary candidates from the given set of unary ind, and cross-check them
    with the given set of known satisfied n-ary dependencies.

    Returns
    -------
    Set[Ind], Set[Ind]:
        A tuple with all satisfied, and not satisfied, n-ary inclusion dependencies
    """
    satisfied = set()
    unsatisfied = set()
    input_ind_s = ind_to_nodes(input_ind)
    for i in range(2, n + 1):
        candidates = generate_candidates(i, unary)
        for ind in candidates:
            if ind.get_all_unary() in input_ind_s:
                satisfied.add(ind)
            else:
                unsatisfied.add(ind)
    return frozenset(satisfied), frozenset(unsatisfied)


def log_border(label: str, border: Iterable[FrozenSet[Ind]], level=logging.CRITICAL):
    """
    Log the content of the border
    """
    _logger.log(level, label)
    for p in border:
        _logger.log(level, '\t%s', node_to_ind(p))


def generalize_set(opt_di: Set[Set[Ind]], n: int) -> Set[Set[Ind]]:
    """
    Generate the IND that generalize those in opt_di, and with an arity of at least n
    """
    result = set()
    # Note: The zigzag implementation doesn't check on the size of the set, but the paper says:
    # candidats = U i ∈ optDI {j | j < i, |j| = |i| − 1 and |j| > k} (note the last condition)
    for i in filter(lambda fi: len(fi) - 1 > n, opt_di):
        result.update(map(frozenset, itertools.permutations(i, len(i) - 1)))
    return frozenset(result)


class Zigzag(object):
    """
    Parameters
    ----------
    n : int
        Search for satisfied n-ary inclusion dependencies using the selected bootstrap algorithm
    alpha : float
        Rejection level of the H0 hypothesis: samples come from the same population.
        If H0 is rejected, then an inclusion dependency is considered to *not* be satisfied.
    alpha2 : float
        Expected to be less than alpha. If alpha rejects H0, but alpha2 doesn't, the solution will
        be considered, as the paper says, "almost true", and explored top-down.
    bootstrap : Callable
        Method used to bootstrap Zigzag with n-ary IND. Defaults to MIND.
    bootstrap_args : dict
        Arguments to pass down to the bootstrap algorithm
    test : Callable
        Statistic test to use
    test_args : dict
        Arguments to pass down to the statistic test
    """

    def __init__(self, n: int = 2, alpha: float = 0.05, alpha2: float = 0.01,
                 bootstrap: Callable = Mind(), bootstrap_args: dict = None,
                 test: Callable = knn_test, test_args=None):
        self._start_n = n
        self._alpha = alpha
        self._alpha2 = alpha2
        self._bootstrap = bootstrap
        self._bootstrap_args = bootstrap_args if bootstrap_args else dict()
        self._test = test
        self._test_args = test_args if test_args else dict()

    def _validate(self, ind_set):
        ind = node_to_ind(ind_set)
        _logger.debug('Using %s for validating %s', self._test, ind)
        if ind.lhs.has_duplicates or ind.rhs.has_duplicates:
            return False
        return self._test(ind.lhs.data, ind.rhs.data, **self._test_args)

    # noinspection PyPep8Naming
    def __call__(self, uind: FrozenSet[Ind]) -> FrozenSet[Ind]:
        """
        Zigzag algorithm

        Parameters
        ----------
        uind : FrozenSet[Ind]
            Set of satisfied unary inclusion dependencies

        Returns
        -------
        FrozenSet[Ind] :
            Maximum satisfied inclusion dependencies found
        """
        input_ind = self._bootstrap(uind, stop=self._start_n, **self._bootstrap_args)
        unary = get_unary_ind(input_ind)
        n = self._start_n

        _logger.info('Generating list of satisfied/non satisfied')
        satisfied, unsatisfied = check_nary_ind(self._start_n, unary, input_ind)

        _logger.info('Converting %d-IND to sets', self._start_n)
        positive_border = ind_to_nodes(satisfied)
        negative_border = ind_to_nodes(unsatisfied)

        _logger.info('Computing initial optimistic border')
        optimistic_border = calculate_optimistic_border(unary, set(map(node_to_ind, negative_border)), n)
        diff = optimistic_border.difference(positive_border)
        log_border('Initial optimistic border', optimistic_border, level=logging.DEBUG)
        log_border('Initial positive border', positive_border, level=logging.DEBUG)

        while len(diff) > 0:
            # Fail-safe
            if n > len(uind):
                _logger.critical('Loop run for too long!')
                log_border('Positive border', positive_border)
                log_border('Optimistic border', optimistic_border)
                log_border('Negative border', negative_border)
                log_border('Difference', diff)
                log_border('Intersection positive and negative', positive_border.intersection(negative_border))
                log_border('Intersection negative optimistic', negative_border.intersection(optimistic_border))
                raise RuntimeError('Loop run for too long!')

            # Log where are we
            _logger.info('Positive border: %d', len(positive_border))
            _logger.info('Optimistic border: %d', len(optimistic_border))
            _logger.info('Negative border: %d', len(negative_border))

            # Verify optimistic candidates
            pessimistic = set()
            possible_smaller = set()
            _logger.info('Processing %d optimistic options', len(optimistic_border))
            # metanome implementation iterates over the optimistic border, but the paper says
            # for all i ∈ Bd + (I opt ) \ Bd + (I) do
            for ind_set in diff:
                p = self._validate(ind_set)
                if p >= self._alpha:
                    _logger.debug('Optimistic accepted: (%.2f): %s', p, node_to_ind(ind_set))
                    assert ind_set not in negative_border, node_to_ind(ind_set)
                    positive_border.add(ind_set)
                else:
                    _logger.debug('Optimistic rejected (%.2f): %s', p, node_to_ind(ind_set))
                    assert ind_set not in positive_border
                    negative_border.add(ind_set)
                    if p >= self._alpha2 and len(ind_set) > n + 1:
                        _logger.debug('Possible smaller candidate')
                        possible_smaller.add(ind_set)
                    else:
                        pessimistic.add(ind_set)

            _logger.info('%d negative matches', len(pessimistic))

            # Possibly smaller
            while possible_smaller:
                _logger.info('%d possibly smaller', len(possible_smaller))
                candidates_below_optimistic = generalize_set(possible_smaller, n)
                to_remove = set()
                _logger.info('Processing %d below optimistic', len(candidates_below_optimistic))
                for ind_set in candidates_below_optimistic:
                    if is_satisfied(ind_set, positive_border):
                        _logger.debug('Skip %s', node_to_ind(ind_set))
                        to_remove.add(ind_set)
                    elif ind_set in negative_border:
                        _logger.debug('Skip negative %s', node_to_ind(ind_set))
                    else:
                        p = self._validate(ind_set)
                        if p >= self._alpha:
                            _logger.debug('Accepted (%.2f) %s', p, node_to_ind(ind_set))
                            assert ind_set not in negative_border
                            positive_border.add(ind_set)
                            to_remove.add(ind_set)
                        else:
                            _logger.debug('Rejected (%.2f) %s', p, node_to_ind(ind_set))
                            assert ind_set not in positive_border
                            negative_border.add(ind_set)
                possible_smaller = candidates_below_optimistic.difference(to_remove)

            positive_border = remove_generalizations(positive_border)

            # Verify pessimistic candidates
            candidates_next = get_candidates_next(pessimistic, n)
            _logger.info('Verifying %d candidates from the pessimistic', len(candidates_next))
            for ind_set in candidates_next:
                if is_satisfied(ind_set, negative_border):
                    _logger.debug('Skip candidate because it is already on the negative border %s',
                                  node_to_ind(ind_set))
                elif is_satisfied(ind_set, positive_border):
                    _logger.debug('Candidate %s satisfied by the border', node_to_ind(ind_set))
                    positive_border.add(ind_set)
                else:
                    p = self._validate(ind_set)
                    if p >= self._alpha:
                        _logger.debug('Candidate accepted (%.2f) %s', p, node_to_ind(ind_set))
                        assert ind_set not in positive_border
                        positive_border.add(ind_set)
                    else:
                        _logger.debug('Candidate rejected (%.2f) %s', p, node_to_ind(ind_set))
                        assert ind_set not in positive_border
                        negative_border.add(ind_set)

            positive_border = remove_generalizations(positive_border)
            negative_border = remove_specializations(negative_border)

            # There must not be elements shared between the negative and positive borders
            if not positive_border.isdisjoint(negative_border):
                _logger.critical('Positive and negative border must not intersect!')
                log_border('Positive border', positive_border)
                log_border('Optimistic border', optimistic_border)
                log_border('Negative border', negative_border)
                log_border('Intersection of positive and negative', positive_border.intersection(negative_border))
                raise RuntimeError('Positive and negative border must not intersect!')

            # Next iteration
            n += 1
            optimistic_border = calculate_optimistic_border(unary, set(map(node_to_ind, negative_border)), n)
            diff = optimistic_border.difference(positive_border)
            log_border('New optimistic border', optimistic_border, level=logging.DEBUG)

        # Convert candidates back to Ind
        result = set()
        for ind_set in positive_border:
            result.add(node_to_ind(ind_set))
        return frozenset(result)
