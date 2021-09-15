from typing import Iterable, Union

import networkx as nx
from typing.io import IO

from matchbox.hypergraph import Edge, Graph


def to_networkx(graph: Graph):
    G = nx.Graph()
    knot = 0
    for e in graph.E:
        for v in e:
            for r in e.set.difference({v}):
                G.add_edge(v, r, p=e.confidence)
        knot += 1
    for n in graph.V:
        G.add_node(n)
    return G


def to_dot_file(graph: Graph, path: Union[str, IO]):
    G = to_networkx(graph)
    nx.drawing.nx_agraph.write_dot(G, path)
