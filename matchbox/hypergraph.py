from functools import reduce
from typing import Iterable, FrozenSet, Tuple

import numpy as np
from scipy.special import comb
from scipy.stats import hypergeom

from .attributeset import AttributeSet
from .ind import Ind


class Edge(FrozenSet):
    """
    A frozenset of vertex with the 'valid' attribute
    """

    def __init__(self, args, valid=False, confidence: float = 0):
        super().__init__()
        self.set = frozenset(args)
        self.valid = valid
        self.confidence = confidence

    def __repr__(self):
        return repr(self.set)

    def __hash__(self):
        return hash(self.set)

    def __eq__(self, other):
        return self.set == other.set

    def __iter__(self):
        return iter(self.set)

    def __len__(self):
        return len(self.set)

    def issuperset(self, other) -> bool:
        return self.set.issuperset(other)

    def to_ind(self) -> Ind:
        """
        Transform a n-ary inclusion dependency modeled as a set of unary ind, to a proper n-ind
        """
        lhs_attr = []
        rhs_attr = []
        ind = None
        for ind in self.set:
            lhs_attr.extend(ind.lhs.attr_names)
            rhs_attr.extend(ind.rhs.attr_names)
        assert len(lhs_attr), self
        return Ind(
            lhs=AttributeSet(ind.lhs.relation_name, lhs_attr, ind.lhs.relation),
            rhs=AttributeSet(ind.rhs.relation_name, rhs_attr, ind.rhs.relation),
            confidence=self.confidence
        )


class Graph(object):
    """
    A Graph is just a set of vertex plus a set of edges (which, in turn, are a set of vertex)
    """

    def __init__(self, V: Iterable[Ind] = set(), E: Iterable[Edge] = set()):
        self.V = set(V)
        self.E = set(E)
        self._rank = None

    @property
    def rank(self):
        if self._rank is None:
            self._rank = len(next(iter(self.E)))
        return self._rank

    def empty(self) -> bool:
        """
        Returns
        -------
        out : bool
            True if there are no edges
        """
        return len(self.E) == 0

    def __hash__(self):
        return hash(self.E)

    def __eq__(self, other):
        return self.V == other.V and self.E == other.E

    def __repr__(self) -> str:
        entries = []
        for e in self.E:
            entries.append(repr(e))
        return 'Graph[' + repr(self.V) + '\n' + '\n'.join(entries) + ']'


def is_clique(G: Graph, S: FrozenSet[Ind]) -> bool:
    """
    Parameters
    ----------
    G : Graph
    S : Set of nodes (Ind)

    Returns
    -------
    out : bool
        True if S is a clique on the graph G
    """
    counts = dict()
    n = len(S)
    k = None
    for e in G.E:
        if k is None:
            k = len(e)
        if S.issuperset(e.set):
            for unary in e.set:
                counts[unary] = counts.get(unary, 0) + 1
    min_degree = comb(n - 1, k - 1)
    return all(map(lambda v: v >= min_degree, counts.values())) and len(counts) == len(S)


def induced_subgraph(G: Graph, S: FrozenSet[Ind]) -> Graph:
    """
    Generate the subgraph of G induced by the set of nodes S.
    See Also
    --------
    https://en.wikipedia.org/wiki/Induced_subgraph

    Parameters
    ----------
    G : Graph
    S : Set of nodes

    Returns
    -------
    out : Graph
        Induced subgraph
    """
    E = set()
    for e in G.E:
        if S.issuperset(e.set):
            E.add(e)
    return Graph(S, E)


def get_degrees(G: Graph, S: FrozenSet[Ind] = None):
    """
    Compute the degree of a set of nodes. Default to all vertices on a graph.

    Parameters
    ----------
    G : Graph
    S : Vertices

    Returns
    -------
    out : dict
        A dictionary where the key is the node, and the value is the degree
    """
    if not S:
        S = G.V
    node_degree = dict([(n, 0) for n in S])
    for e in G.E:
        if S.issuperset(e.set):
            for unary in e.set:
                node_degree[unary] += 1
    return node_degree


def compute_thresholds(k: int, n: int, e: int, Lambda: float, gamma: float) -> Tuple[int, int]:
    """
    Compute the edge and degree thresholds

    Parameters
    ----------
    k : int
        Rank of the hypergraph
    n : int
        Number of nodes
    e : int
        Number of edges, if -1, assume gamma * max_cardinality
    Lambda : float
        Significance level for the rejection of all edges are equally likely to be
        rejected
    gamma : float
        Ratio of edges that must be present

    Returns
    -------
    min_cardinality, min_degree : int, int
    """
    # Edge ratio
    max_cardinality = comb(n, k)
    min_cardinality = max(1, np.floor(gamma * max_cardinality))

    if e < 0:
        e = max(1, np.floor(max_cardinality * gamma))

    # Expectation of degree
    # If we have ne = |E| edges and all are similar, the degree of a node
    # follows an hypergeometric distribution (i.e. if the clique has 108 out of 120 edges,
    # how likely is it to have only a degree of less than 10?)
    max_degree = comb(n - 1, k - 1)
    if Lambda > 0:
        h = hypergeom(max_cardinality, e, max_degree)
        min_degree = h.ppf(Lambda)
    elif Lambda < 0:
        min_degree = max_degree * abs(Lambda)
    else:
        min_degree = 0

    return min_cardinality, min_degree


def is_quasi_clique(G: Graph, S: FrozenSet[Ind], Lambda: float, gamma: float) -> bool:
    """
    Parameters
    ----------
    G : Graph
    S : Set of nodes (Ind)
    Lambda : float
        Assuming H0: every edge is equally likely to be missing, the expected distribution
        of the degree of each node follows an hyper-geometric distribution (take n where only k are
        present). This parameter is the significance level used to reject H0 with this test.
    gamma : float
        Ratio of number of edges (Brunato et al 2007)

    Returns
    -------
    out : bool
        True if S is a clique on the graph G
    """

    node_degree = dict()
    n = len(S)
    k = None

    E = set()
    for e in G.E:
        if k is None:
            k = len(e)
        assert k == len(e)
        if S.issuperset(e.set):
            E.add(e)
            for unary in e.set:
                node_degree[unary] = node_degree.get(unary, 0) + 1

    min_cardinality, min_degree = compute_thresholds(k, n, len(E), Lambda, gamma)
    if len(E) < min_cardinality or len(node_degree) != len(S):
        return False
    return all(map(lambda v: v >= min_degree, node_degree.values()))


def generate_graph(ind: Iterable[Ind], arity=None):
    if not arity:
        arity = max(map(lambda i: i.arity, ind))
    nodes = set()
    edges = set()
    for i in ind:
        uind = i.get_all_unary()
        if len(uind) >= arity:
            edges.add(Edge(uind, valid=True, confidence=i.confidence))
            nodes.update(uind)
    graph = Graph(nodes, edges)
    return graph, arity
