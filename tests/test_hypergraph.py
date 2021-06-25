from matchbox.hypergraph import get_degrees


def test_get_degrees_2graph(graph2):
    deg = get_degrees(graph2)
    assert len(deg) == 4
    assert deg[1] == 2
    assert deg[2] == 3
    assert deg[3] == 2
    assert deg[4] == 3


def test_get_degrees_3graph(hyperclique48):
    deg = get_degrees(hyperclique48)
    assert len(deg) == 5
    assert deg[1] == 4
    assert deg[2] == 4
    assert deg[3] == 4
    assert deg[4] == 4
    assert deg[5] == 2


def test_get_degrees_3graph_oneout(hyperclique48):
    """
    Edges that connect one of S, but including any *not* on S must not be count
    """
    deg = get_degrees(hyperclique48, {1, 2, 3, 4})
    assert len(deg) == 4
    assert deg[1] == 3
    assert deg[2] == 3
    assert deg[3] == 3
    assert deg[4] == 3
