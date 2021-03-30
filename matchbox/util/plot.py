from typing import Iterable, Union

import networkx as nx
from typing.io import IO

from matchbox.hypergraph import Edge


def to_networkx(edges: Iterable[Edge]):
    G = nx.Graph()
    knot = 0
    for e in edges:
        for v in e:
            for r in e.set.difference({v}):
                G.add_edge(v, r, p=e.confidence)
        knot += 1
    return G


def to_dot_file(edges: Iterable[Edge], path: Union[str, IO]):
    G = to_networkx(edges)
    nx.drawing.nx_agraph.write_dot(G, path)
