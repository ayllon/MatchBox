"""
Implementation of the FINDQ algorithm

Based on
- Discovery of high-dimensional inclusion dependencies, Koeller 2003
- An efficient algorithm for solving pseudo clique enumeration problem, Uno 2010
"""
import logging
from functools import reduce
from typing import FrozenSet, Set, Callable, Tuple, Dict

import numpy as np
from scipy.special import comb

from .find2 import generate_clique_candidate, gen_sub_inds, gen_k_ary_ind_from_cliques
from .hypergraph import Graph, induced_subgraph, Edge, is_quasi_clique, compute_thresholds, get_degrees
from .ind import Ind
from .mind import Mind
from .tests import knn_test

_logger = logging.getLogger(__name__)


def get_connected(E: FrozenSet[Edge], S: FrozenSet) -> Dict[Ind, Set[Ind]]:
    """
    For speeding up the lookup of connected vertices, we precompute the list
    of connected Ind to a given Ind

    Parameters
    ----------
    S
    G : Graph
        Full graph

    Returns
    -------
    out : Dictionary of Ind, Set(Ind), pair
        The value is the set of Ind connected to a given Ind
    """
    connected = dict([(v, set()) for v in S])
    for e in E:
        if S.issuperset(e.set):
            for unary in e.set:
                connected[unary].update(e.set.difference({unary}))
    return connected


def clq(n: int, k: int) -> int:
    """
    Number of edges on a clique of n vertices on a k-uniform graph

    See Also
    --------
    Uno 2010, page 6
    """
    return comb(n, k)


def theta(k: int, gamma: float, vertex_count: int, edge_count: int) -> float:
    """
    Threshold on the number of edges that a new vertex must contribute to a quasi-clique
    so it continues being a quasi-clique

    Parameters
    ----------
    k : int
        Graph rank
    gamma : float
        Threshold for the number of edges
    vertex_count : int
        Number of vertices on the known quasi-clique K
    edge_count : int
        Number of edges on the known quasi-clique K

    Returns
    -------
    out : float
        Threshold on how many edges a new vertex u must contribute so K u {v} is also a pseudo-clique.
        Note that this threshold may very well be 0 if K already has `gamma * clq(K+1)` edges!
        The new vertex u should be adjacent to at least one member of K too.

    See Also
    --------
    Uno 2010, page 10, lemma 2
    """
    return gamma * clq(vertex_count + 1, k) - edge_count


# noinspection PyPep8Naming
def grow_clique(G: Graph, K: FrozenSet, gamma: float, Lambda: float,
                Seed: Set = None, Candidates: Set = None, G_connected=None,
                K_degrees: Dict[Ind, int] = None,
                edge_count: int = None) -> FrozenSet[Edge]:
    # Seed
    if Seed is None:
        Seed = K

    # Get candidate set
    if Candidates is None:
        Candidates = G.V.difference(K)

    # Get degrees for the current clique
    if K_degrees is None:
        K_degrees = get_degrees(G, K)

    # Set of vertices connected to nodes in G.V
    if G_connected is None:
        G_connected = get_connected(G.E, G.V)

    # How many edges on the current quasi-clique?
    if edge_count is None:
        edge_count = len(induced_subgraph(G, K).E)

    # Result set
    result = set()

    # The threshold for a new child quasi-clique depends only on the current one
    # i.e. A new vertex *must* provide enough edges
    gamma_degree_min = max(1, np.floor(theta(G.rank, gamma, len(K), edge_count))) if K else 0

    # We can make an optimistic prediction on the degree, assuming as many as gamma * edges
    # can be missing for adding any one vertex
    # Any vertex with a degree lower than this over the whole graph can be removed, it will
    # never pass the degree check
    _, lambda_min_degree = compute_thresholds(G.rank, len(K) + 1, -1, Lambda, gamma)

    # We can remove all v that do not satisfy the threshold
    children = dict()
    for v in Candidates:
        # Graph induced by K U {v}
        G_candidate = induced_subgraph(G, K.union({v}))
        # Degree of v
        K_degrees = get_degrees(G_candidate)
        v_degree = K_degrees[v]
        # Uno, lemma 2, adapted to k > 2, plus custom condition
        if v_degree >= gamma_degree_min and v_degree >= lambda_min_degree:
            children[v] = K_degrees

    # See Uno 2010, page 10
    for v, K_degrees in children.items():
        # We have removed all candidates with a low degree, so no need to re-check the degree of v
        # We need to check if we grow following this one, would it be v*?
        # Only taking into account candidates, since we grow from a seed, we need to consider all vertex
        # from the initial clique as if they were precedent
        v_star = [p[0] for p in sorted(K_degrees.items(), key=lambda pair: (pair[1], pair[0])) if p[0] not in Seed][0]
        _, min_degree = compute_thresholds(G.rank, len(K) + 1, edge_count + K_degrees[v], Lambda, gamma)
        if v == v_star:
            if min_degree <= K_degrees[v]:
                nested = grow_clique(
                    G, K.union({v}), gamma, Lambda,
                    Seed, Candidates.difference({v}), G_connected, K_degrees, edge_count + K_degrees[v]
                )
                result.update(nested)
    result.add(frozenset(K))
    return frozenset(result)


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
        edges = list(G.E)
        prob = [1 - e.confidence for e in edges]
        prob /= np.sum(prob)
        e = np.random.choice(edges, p=prob)
        S = generate_clique_candidate(G, e)
        G1 = induced_subgraph(G, S)
        G2.E = G.E.difference({e})
    return G1, G2


# noinspection PyPep8Naming
def find_seeds(G: Graph, lambd: float, gamma: float) -> FrozenSet[Edge]:
    """
    Based on a combination of the HYPERCLIQUE algorithm from Koeller 2003,
    and a modification of Uno 2010 adapted to hypercliques, defined
    as a generalization of Brunato 2007

    Parameters
    ----------
    G : Graph
    lambd : float
    gamma : float

    Returns
    -------
    out : FrozenSet[Graph]
        Set of maximal quasi-cliques in G.
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
            result.update(find_seeds(G1, lambd, gamma))
            result.update(find_seeds(G2, lambd, gamma))
        if not (E_star and reducible):
            break
        G = Graph(G.V, E_star)

    count = dict()
    for r in result:
        count[len(r)] = count.get(len(r), 0) + 1
    for k, c in count.items():
        _logger.debug('%d with %d vertex', c, k)

    return frozenset(result)


def find_quasicliques(G: Graph, lambd: float, gamma: float, grow: bool) -> FrozenSet[Edge]:
    """

    Parameters
    ----------
    G
    lambd
    gamma

    Returns
    -------

    """
    # Find seeds
    seeds = find_seeds(G, lambd, gamma)
    if not grow:
        return seeds
    _logger.info('Got %d seeds', len(seeds))
    # Sort seeds by cardinality, highest first
    seeds = sorted(seeds, key=len, reverse=True)

    # Output set
    result = set()

    # For each seed
    for seed in seeds:
        # Skip if the seed is already a subset
        if any(map(seed.issubset, result)):
            continue
        # Apply the growing step
        _logger.info('Growing from %d vertices', len(seed))
        Sg = grow_clique(G, seed, gamma=gamma, Lambda=lambd)
        _logger.info('Expanded %d vertices into %d candidates', len(seed), len(Sg))
        # Again, sort by cardinality, highest first
        Sg = sorted(Sg, key=len, reverse=True)
        for S in Sg:
            if is_quasi_clique(G, S, lambd, gamma):
                if not any(map(S.issubset, result)):
                    _logger.info('Got a positive with %d vertices from an initial seed of %d', len(S), len(seed))
                    result.add(Edge(S))
    return result


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

    def __init__(self, n: int = 2, alpha: float = 0.05, lambd: float = 0.85, gamma: float = 0.85, grow: bool = True,
                 bootstrap: Callable = Mind(), bootstrap_args: dict = None,
                 test: Callable = knn_test, test_args: dict = None):
        self.__n = n
        self.__alpha = alpha
        self.__lambda = lambd
        self.__gamma = gamma
        self.__grow = grow
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
        try:
            start_arity, G = self.generate_graph(uind)
        except StopIteration:
            return frozenset()

        _logger.info('Looking for %.2f / %.2f quasi hypercliques with %d edges', self.__lambda, self.__gamma, len(G.E))
        H = find_quasicliques(G, self.__lambda, self.__gamma, self.__grow)
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
            Gm.V = set(reduce(set.union, map(lambda e: e.set, Gm.E), set()))
            _logger.info('Looking for %.2f / %.2f quasi hypercliques with %d edges',
                         self.__lambda, self.__gamma, len(Gm.E))
            H = find_quasicliques(Gm, self.__lambda, self.__gamma, self.__grow)
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
