"""
Implementation of the FIND2 algorithm

Discovery of high-dimensional inclusion dependencies, Koeller 2003

Following the pseudocode from the paper with a couple of corrections
"""
import logging
from functools import reduce
from typing import FrozenSet, Set, Callable, Tuple

import numpy as np

from matchbox import Mind, Ind
from matchbox.find2 import generate_clique_candidate, gen_sub_inds, gen_k_ary_ind_from_cliques
from matchbox.hypergraph import Graph, induced_subgraph, Edge, is_quasi_clique
from matchbox.tests import knn_test

_logger = logging.getLogger(__name__)


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
        edges = list(G.E)
        prob = [1 - e.confidence for e in edges]
        prob /= np.sum(prob)
        e = np.random.choice(edges, p=prob)
        S = generate_clique_candidate(G, e)
        G1 = induced_subgraph(G, S)
        G2.E = G.E.difference({e})
    return G1, G2


def find_quasicliques(G: Graph, lambd: float, gamma: float) -> FrozenSet[Edge]:
    """
    Based on HYPERCLIQUE algorithm from Koeller 2003, but looks for quasicliques,
    instead

    Parameters
    ----------
    G : Graph
    lamb : float
    gamma : float

    Returns
    -------
    out : FrozenSet[Graph]
        Set of maximal quasicliques in G.

    Notes
    -----
    HYPERCLIQUE return edges, but instead we return here the full graph (vertices + edges)
    since the connectivity may not be complete, so it is important to keep track of
    the missing edges and not assume that all are connected with all
    """
    result = set()
    while True:
        _logger.debug('Looking on graph with %d edges', len(G.E))
        # Probably another typo, the paper/thesis puts the flag outside the loop,
        # but that causes an infinite loop
        reducible = False
        E_star = set()
        for e in G.E:
            S = generate_clique_candidate(G, e)
            if is_quasi_clique(G, S, lambd, gamma):
                if not any(map(S.issubset, result)):
                    result.add(Edge(S))
                reducible = True
            else:
                E_star.add(e)
        if not reducible:
            _logger.debug('Not reducible!')
            G1, G2 = reduce_graph(G)
            result.update(find_quasicliques(G1, lambd, gamma))
            result.update(find_quasicliques(G2, lambd, gamma))
        G = Graph(G.V, E_star)
        if not (E_star and reducible):
            break

    count = dict()
    for r in result:
        count[len(r)] = count.get(len(r), 0) + 1
    for k, c in count.items():
        _logger.debug('%d with %d vertex', c, k)

    return frozenset(result)


class FindGamma(object):
    """
    Implementation of the FIND-γ algorithm

    Parameters
    ----------
    n : int
        Use the bootstrap algorithm to find IND up to this arity
    alpha : float
        For the test statistic
    gamma : float
        For the quasi-cliques
    bootstrap : Callable
        Bootstrap method. Defaults to Mind
    bootstrap_args : dict
        Arguments to pass to the bootstrap method
    test : Callable
        Test method, defaults to knn_test
    test_args : dict
        Arguments to pass to the test method
    """

    def __init__(self, n: int = 2, alpha: float = 0.05, lambd: float = 0.85, gamma: float = 0.85,
                 bootstrap: Callable = Mind(), bootstrap_args: dict = None,
                 test: Callable = knn_test, test_args: dict = None):
        self.__n = n
        self.__alpha = alpha
        self.__lambda = lambd
        self.__gamma = gamma
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
        _logger.info('Validating %d candidates', len(S))
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
        start_arity, G = self.generate_graph(uind)

        _logger.info('Looking for %.2f / %.2f quasi hypercliques with %d edges', self.__lambda, self.__gamma, len(G.E))
        H = find_quasicliques(G, self.__lambda, self.__gamma)
        I = self._validate_all(H)

        result = set()
        for m in range(start_arity + 1, ks):
            _logger.info('Iteration %d (%d candidates)', m, len(I))
            _logger.info('Iteration %d (%d positives)', m, len(result))
            C = set()
            for c in I:
                if c.valid and len(c) >= m - 1:
                    result.add(c)
                if not c.valid and len(c) >= m:
                    C.add(c)
            _logger.info('Generating %d-ary from %d', m, len(C))
            k_ary = gen_k_ary_ind_from_cliques(m, C)
            _logger.info('%d %d-ary generated from %d', len(k_ary), m, len(C))
            Gm = Graph()
            Gm.E = self._validate_all(k_ary)
            Gm.E = set(filter(lambda e: e.valid, Gm.E))
            if Gm.empty():
                return frozenset(map(Edge.to_ind, result))
            result.update(gen_sub_inds(m, Gm, result))
            Gm.V = frozenset(reduce(frozenset.union, map(lambda e: e.set, Gm.E), frozenset()))
            _logger.info('Looking for %.2f / %.2f quasi hypercliques with %d edges',
                         self.__lambda, self.__gamma, len(Gm.E))
            H = find_quasicliques(Gm, self.__lambda, self.__gamma)
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
