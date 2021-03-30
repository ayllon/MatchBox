"""
Implementation of the FINDQ algorithm

An efficient algorithm for solving pseudo clique enumeration problem, Uno 2010
"""
import logging
from functools import reduce
from typing import FrozenSet, Set, Callable, Dict, Tuple

import numpy as np

from matchbox import Mind, Ind
from matchbox.hypergraph import Graph, Edge
from matchbox.tests import knn_test

_logger = logging.getLogger(__name__)


def get_connected(G: Graph) -> Dict[Ind, Set[Ind]]:
    """
    For speeding up the lookup of connected vertices, we precompute the list
    of connected Ind to a given Ind

    Parameters
    ----------
    G : Graph
        Full graph

    Returns
    -------
    out : Dictionary of Ind, Set(Ind), pair
        The value is the set of Ind connected to a given Ind
    """
    connected = dict([(n, set()) for n in G.V])
    for e in G.E:
        for unary in e.set:
            connected[unary].update(e.set.difference({unary}))
    return connected


class FindQ(object):
    """
    Based on "An efficient algorithm for solving pseudo clique enumeration problem", Uno 2010,
    but modified for lambda-gamma quasicliques, as defined by Brunato et al 2007

    Parameters
    ----------
    alpha : float
        For the test statistic
    lambd : float
        Ratio of degree of vertices
    gamma : float
        Ratio of number of edges
    bootstrap : Callable
        Bootstrap method. Defaults to Mind
    bootstrap_args : dict
        Arguments to pass to the bootstrap method
    test : Callable
        Test method, defaults to knn_test
    test_args : dict
        Arguments to pass to the test method
    """

    def __init__(self, alpha: float = 0.05, lambd: float = 0., gamma: float = 0.85,
                 bootstrap: Callable = Mind(), bootstrap_args: dict = None,
                 test: Callable = knn_test, test_args: dict = None):
        self.__alpha = alpha
        self.__lambda = lambd
        self.__gamma = gamma
        self.__bootstrap = bootstrap
        self.__bootstrap_args = bootstrap_args if bootstrap_args else dict()
        self.__test = test
        self.__test_args = test_args if test_args else dict()

    @staticmethod
    def __clq(n: int) -> int:
        """
        Number of edges on a clique of n vertices. (n over 2, which is (n*(n-1))/2)

        Parameters
        ----------
        n : int
            Number of vertices

        Returns
        -------
        out : int
            Number of edges on a clique with n vertices

        See Also
        --------
        Uno 2010, page 6
        """
        return (n * (n - 1)) // 2

    def _theta(self, vertex_count: int, edge_count: int) -> float:
        """
        Threshold on the number of edges that a new vertex must contribute to a pseudo-clique
        so it continues being a pseudo-clique

        Parameters
        ----------
        vertex_count : int
            Number of vertices on the known pseudo-clique K

        edge_count : int
            Number of edges on the known pseudo-clique K

        Returns
        -------
        out : float
            Threshold on how many edges a new vertex u must contribute so K u {v} is also a pseudo-clique.
            Note that this threshold may very well be 0 if K already has `gamma * clq(K+1)` edges!
            The new vertex u should be adjacent to at least one member of K too.

        See Also
        --------
        Uno 2010, page 10
        """
        return self.__gamma * self.__clq(vertex_count + 1) - edge_count

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
        G = self.__generate_graph(uind)
        result = set()
        _logger.info('Looking for %.2f / %.2f quasi cliques with %d edges', self.__lambda, self.__gamma, len(G.E))
        self.__find_quasi(
            G, get_connected(G),
            frozenset(), dict(), 0,
            result=result
        )
        return list(map(Edge.to_ind, result))

    def __validate(self, edge: Edge) -> Tuple[bool, float]:
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

    def __find_quasi(self, G: Graph, G_connected: Dict[Ind, Set[Ind]],
                     K: FrozenSet, K_degrees: Dict[Ind, int], edge_count: int,
                     result: Set) -> bool:
        """
        Recursive call that search for quasi-cliques on G.

        Parameters
        ----------
        G : Graph
            Original full graph
        G_connected : Dictionary
            See get_connected
        K : Frozenset
            Vertices forming the current pseudo-clique
        K_degrees : Dictionary
            Degree on the vertices in the pseudo-clique K
        edge_count : int
            Number of edges on K
        result : set
            New Ind are inserted here
        Returns
        -------
        out : bool
            True if K is also a valid Inclusion Dependency. Callers can use this information
            to avoid redundant testing: if a child of K is a valid IND, so is K
        """
        # Test and insert K only if no child is a valid IND
        add_K = True

        # The threshold for a new child pseudo-clique depends only on the current one
        threshold = max(1, np.floor(self._theta(len(K), edge_count))) if K else 0

        # Threshold on the degree of vertices belonging to a clique of size |K|+1
        # (1 edge connecting a given vertix with all others)
        degree_threshold = max(1, np.floor(self.__lambda * len(K))) if K else 0

        # If members of K already do not satisfy the degree threshold on the original graph,
        # no child pseudo-clique will satisfy it
        degree_ok = all(map(lambda n: len(G_connected[n]) >= degree_threshold, K))

        if degree_ok:
            # See Uno 2010, page 10
            for v in G.V.difference(K):
                # If the degree of the new vertex is below the threshold on the original graph,
                # we can just skip it
                if len(G_connected[v]) < degree_threshold:
                    continue

                # Elements in K that are connected to v
                affected = G_connected[v].intersection(K)

                # The degree of v on the induced subgraph is just the number of connected edges
                K_degrees[v] = len(affected)

                # Since we are adding a new vertex, update the degree of the other connected vertices
                for a in affected:
                    K_degrees[a] = K_degrees.get(a, 0) + 1

                # NOTE:
                # It may be tempting to apply the degree_threshold here, but even if K U {v}
                # does not satisfy it, a child might!

                # If K is the empty set, the degree will be 0, but v = v*
                # If K is not the empty set, then at least one node from K must be connected
                if (len(K) == 0 or affected) and K_degrees[v] >= threshold:
                    # TODO: I think this can be optimized? See Uno 2010, Lemma 3, point 2
                    v_star = sorted(K_degrees.items(), key=lambda pair: (pair[1], pair[0]))[0][0]
                    if v == v_star:
                        add_K &= not self.__find_quasi(
                            G, G_connected,
                            K.union({v}), K_degrees, edge_count + len(affected),
                            result=result
                        )

                # Undo counting update for the next iteration
                for a in affected:
                    K_degrees[a] -= 1
                    if K_degrees[a] == 0:
                        del K_degrees[a]
                # v will be deleted when K is the empty set
                if v in K_degrees:
                    del K_degrees[v]

        # Do not bother checking K if a child was a valid IND
        if add_K and K and not any(map(K.issubset, result)):
            e = Edge(K)
            e.valid, e.confidence = self.__validate(e)
            if e.valid:
                result.add(e)
            # Is K valid?
            return e.valid
        # If a child was valid, so is K
        return True

    def __generate_graph(self, ind: Set[Ind]) -> Graph:
        """
        Create the graph from a set of unary, or binary, IND
        """
        iarity = next(iter(ind)).arity
        if iarity == 1:
            _logger.info('Bootstrapping FIND2')
            ind = self.__bootstrap(ind, stop=2, **self.__bootstrap_args)
        elif iarity > 2:
            raise ValueError('Input arity must be either 1 or 2')
        else:
            _logger.info('Skipping bootstrap, input arity is 2')

        all_uind = list(map(frozenset, map(Ind.get_all_unary, ind)))
        E = set()
        for i in ind:
            uind = i.get_all_unary()
            if len(uind) == 2:
                E.add(Edge(uind, valid=True, confidence=i.confidence))
        _logger.info('Found %d edges', len(E))
        G = Graph(reduce(frozenset.union, all_uind, frozenset()), E)
        return G
