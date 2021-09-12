"""
Implementation of the FIND2 algorithm

Discovery of high-dimensional inclusion dependencies, Koeller 2003

Following the pseudocode from the paper with a couple of corrections
"""
import itertools
import logging
from functools import reduce
from typing import Iterable, FrozenSet, Tuple, Set, Callable

import numpy as np

from .hypergraph import Graph, Edge, is_clique, induced_subgraph
from .ind import Ind, is_satisfied
from .mind import Mind
from .tests import knn_test

_logger = logging.getLogger(__name__)


def generate_clique_candidate(G: Graph, e0: Edge) -> FrozenSet[Ind]:
    """
    Generate a clique *candidate*: set of nodes in e0, and set of nodes in the graph
    connected to all of the nodes of e0.

    Parameters
    ----------
    G : Graph
    e0 : Edge

    Returns
    -------
    A set of nodes (Ind)
    """
    S = set(e0.set)
    for v1 in G.V.difference(S):
        flag = True
        for v2 in e0:
            e_set = set(e0.set)
            e_set.remove(v2)
            e_set.add(v1)
            if Edge(e_set) not in G.E:
                flag = False
                break
        if flag:
            S.add(v1)
    return frozenset(S)


# noinspection PyPep8Naming
def find_hypercliques(G: Graph) -> FrozenSet[Edge]:
    """
    HYPERCLIQUE algorithm from Koeller 2003

    Parameters
    ----------
    G : Graph

    Returns
    -------
    out : FrozenSet[Edges]
        Set of maximal hypercliques in G
    """
    result = set()
    while True:
        # Probably another typo, the paper/thesis puts the flag outside the loop,
        # but that causes an infinite loop
        reducible = False
        E_star = set()
        for e in G.E:
            S = generate_clique_candidate(G, e)
            if is_clique(G, S):
                if not any(map(S.issubset, result)):
                    result.add(Edge(S))
                reducible = True
            else:
                E_star.add(e)
        if not reducible:
            G1, G2 = reduce_graph(G)
            result.update(find_hypercliques(G1))
            result.update(find_hypercliques(G2))
        G = Graph(G.V, E_star)
        if not (E_star and reducible):
            break

    return frozenset(map(Edge, result))


# noinspection PyPep8Naming
def reduce_graph(G: Graph) -> Tuple[Graph, Graph]:
    """
    Reduces an irreducible graph removing an edge from it and generating two subgraphs

    Parameters
    ----------
    G : Graph

    Returns
    -------
    G1, G2: Graph, Graph
        G1 is an induced subgraph of G by a random edge e, while G2 is G with the edge e removed
    """
    G1 = Graph(G.V, set())
    G2 = Graph(G.V, set())
    # Note that the paper says |V1| != |V|, but it is likely a typo, and it meant
    # either "until" or "while |V1| == |V|"
    # After all, just looking at Figure 9 it is visible that the number of vertex
    # on G1 is not the number of vertex on the original graph
    while len(G1.V) == len(G.V):
        e = np.random.choice(list(G.E))
        S = generate_clique_candidate(G, e)
        G1 = induced_subgraph(G, S)
        G2.E = G.E.difference({e})
    return G1, G2


def gen_k_ary_ind_from_cliques(k: int, E: Iterable[Edge]) -> FrozenSet[Edge]:
    """
    Generates all k-ary INDs implied by each IND in E

    Parameters
    ----------
    k : int
        arity to generate
    E : Iterable of set of edges

    Returns
    -------
    out : FrozenSet[Edge]
        k-ary edges implied by E
    """
    result = set()
    for i in E:
        result.update(map(Edge, itertools.permutations(i, k)))
    return frozenset(result)


def gen_sub_inds(k: int, G: Graph, result: Set[Edge]) -> FrozenSet[Edge]:
    """
    Generate IND that satisfy:

    1. is a k-subset of an invalid IND from G.E
    2. is not a subset of any valid IND from G.E
    3. is not a subset of any IND in result

    Parameters
    ----------
    k : int
    G : Graph
    result : Set

    Returns
    -------
    out : FrozenSet[Ind]
        k-IND that satisfy the given conditions
    """
    valid = set()
    subs = set()
    for e in G.E:
        if e.valid:
            valid.add(e)
        else:
            subs.update(map(Edge, itertools.permutations(e, k)))

    sigmas = set()
    for s in subs:
        if not is_satisfied(s, valid) and is_satisfied(s, result):
            sigmas.add(s)
    return frozenset(sigmas)


class Find2(object):
    """
    Implementation of the FIND2 algorithm

    Parameters
    ----------
    n : int
        Use the bootstrap algorithm to find IND up to this arity
    alpha : float
        For the test statistic
    bootstrap : Callable
        Bootstrap method. Defaults to Mind
    bootstrap_args : dict
        Arguments to pass to the bootstrap method
    test : Callable
        Test method, defaults to knn_test
    test_args : dict
        Arguments to pass to the test method
    """

    def __init__(self, n: int = 2, alpha: float = 0.05,
                 bootstrap: Callable = Mind(), bootstrap_args: dict = None,
                 test: Callable = knn_test, test_args: dict = None):
        self.__n = n
        self.__alpha = alpha
        self.__bootstrap = bootstrap
        self.__bootstrap_args = bootstrap_args if bootstrap_args else dict()
        self.__test = test
        self.__test_args = test_args if test_args else dict()

    def _validate(self, edge: Edge):
        """
        Run the test method and return True if it accepts H0: both sides of the IND come from the
        same distribution.
        """
        ind = edge.to_ind()
        _logger.debug('Validating %s', ind)
        if ind.lhs.has_duplicates or ind.rhs.has_duplicates:
            return False, 0.
        p = self.__test(ind.lhs.data, ind.rhs.data, **self.__test_args)
        if p >= self.__alpha:
            _logger.debug('Candidate accepted (%.2f) %s', p, ind)
        return p >= self.__alpha, p

    def _validate_all(self, S):
        """
        Convenience method: applies _validate to all elements in S
        """
        for s in S:
            s.valid, s.confidence = self._validate(s)
        return S

    def __call__(self, uind: Set[Ind]) -> FrozenSet[Ind]:
        """
        Find the maximal IND from the given unary IND.

        Parameters
        ----------
        uind : Set[Ind]
            Unary IND

        Returns
        -------
        out : FrozenSet[Ind]
            Maximal IND
        """
        ks = len(uind)
        try:
            start_arity, G = self.generate_graph(uind)
        except StopIteration:
            return frozenset()

        _logger.info('Looking for hypercliques')
        H = find_hypercliques(G)
        _logger.info('Validating %d hypercliques', len(H))
        I = self._validate_all(H)

        result = set(filter(lambda i: len(i) == 1, I))
        for m in range(start_arity + 1, ks):
            _logger.info('Iteration %d (%d candidates)', m, len(I))
            _logger.info('Iteration %d (%d positives)', m, len(result))
            C = set()
            for c in I:
                if c.valid and len(c) >= m - 1:
                    result.add(c)
                if not c.valid and len(c) >= m:
                    C.add(c)
            k_ary = gen_k_ary_ind_from_cliques(m, C)
            _logger.info('%d %d-ary generated from %d', len(k_ary), m, len(C))
            Gm = Graph()
            Gm.E = self._validate_all(k_ary)
            Gm.E = set(filter(lambda e: e.valid, Gm.E))
            if Gm.empty() or True:
                return frozenset(map(Edge.to_ind, result))
            result.update(gen_sub_inds(m, Gm, result))
            Gm.V = frozenset(reduce(frozenset.union, map(lambda e: e.set, Gm.E), frozenset()))
            H = find_hypercliques(Gm)
            I = self._validate_all(H)

        # Convert candidates back to Ind
        return frozenset(map(Edge.to_ind, result))

    def generate_graph(self, ind):
        iarity = next(iter(ind)).arity
        if iarity == 1:
            _logger.info('Bootstrapping FIND2')
            ind = self.__bootstrap(ind, stop=self.__n, **self.__bootstrap_args)
            iarity = self.__n
        else:
            _logger.info('Skipping bootstrap, input arity is %d', iarity)

        all_uind = list(map(frozenset, map(Ind.get_all_unary, ind)))
        E = set()
        for i in ind:
            uind = i.get_all_unary()
            if len(uind) >= self.__n:
                E.add(Edge(uind, valid=True, confidence=i.confidence))
        _logger.info('Found %d %d-hyperedges', len(E), iarity)
        G = Graph(reduce(frozenset.union, all_uind, frozenset()), E)
        return iarity, G
