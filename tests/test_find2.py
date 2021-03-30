from matchbox.find2 import *
import pytest


@pytest.fixture
def hyperclique48():
    """
    From Koeller 2002, Example 4.5
    """
    return Graph(
        V={1, 2, 3, 4, 5},
        E=set(map(Edge, [
            {1, 2, 3}, {1, 3, 4}, {1, 2, 4}, {1, 5, 2},
            {2, 3, 4}, {3, 4, 5}
        ]))
    )


@pytest.fixture
def hyperclique49():
    """
    From Koeller 2002, Example 4.9
    """
    return Graph(
        V={0, 1, 2, 3, 4, 5},
        E=set(map(Edge, [
            {0, 1}, {0, 2}, {1, 2}, {1, 3}, {1, 4},
            {2, 4}, {2, 5}, {3, 4}, {4, 5}
        ]))
    )


@pytest.fixture
def irreducible():
    """
    From Koeller 2002, Example 4.7
    """
    return Graph(
        V={0, 1, 2, 3, 4, 5, 6, 7, 8},
        E=set(map(Edge, [
            {0, 3}, {0, 4}, {0, 5}, {0, 6}, {0, 7}, {0, 8},
            {1, 3}, {1, 4}, {1, 5}, {1, 6}, {1, 7}, {1, 8},
            {2, 3}, {2, 4}, {2, 5}, {2, 6}, {2, 7}, {2, 8},
            {3, 0}, {3, 1}, {3, 2}, {3, 6}, {3, 7}, {3, 8},
            {4, 0}, {4, 1}, {4, 2}, {4, 6}, {4, 7}, {4, 8},
            {5, 0}, {5, 1}, {5, 2}, {5, 6}, {5, 7}, {5, 8},
            {6, 0}, {6, 1}, {6, 2}, {6, 3}, {6, 4}, {6, 5},
            {7, 0}, {7, 1}, {7, 2}, {7, 3}, {7, 4}, {7, 5},
            {8, 0}, {8, 1}, {8, 2}, {8, 3}, {8, 4}, {8, 5},
        ]))
    )


@pytest.fixture
def square():
    """
    The original implementation of is_clique is based on Metanomes' implementation.
    See Also
        https://github.com/HPI-Information-Systems/inclusion-dependency-algorithms/blob/master/adp-algorithms/find2/src/main/java/de/metanome/algorithms/find2/Hypergraph.java#L122
    However, it would seem that test may give false positives. This candidate is an example:

    It is true, however, that generate_clique_candidate will not consider the full graph
    a candidate, but I am not so sure for higher arities if this may not happen
    """
    return Graph(
        V={0, 1, 2, 3},
        E=set(map(Edge, [
            {0, 1}, {1, 2}, {2, 3}, {3, 0}
        ]))
    )


def test_generate_clique_candidate(hyperclique48):
    S = generate_clique_candidate(hyperclique48, Edge({1, 2, 3}))
    assert {1, 2, 3, 4} == S

    S = generate_clique_candidate(hyperclique48, Edge({1, 2, 5}))
    assert {1, 2, 5} == S

    S = generate_clique_candidate(hyperclique48, Edge({3, 4, 5}))
    assert {3, 4, 5} == S



def test_is_clique(hyperclique48, square):
    assert is_clique(hyperclique48, {1, 2, 3})
    assert is_clique(hyperclique48, {1, 2, 5})
    assert is_clique(hyperclique48, {3, 4, 5})
    assert is_clique(hyperclique48, {1, 2, 3, 4})
    assert not is_clique(hyperclique48, {1, 2, 3, 4, 5})
    assert not is_clique(hyperclique48, {2, 3, 4, 5})
    assert not is_clique(square, {0, 1, 2, 3})


def test_induced_subgraph(hyperclique49):
    Gi = induced_subgraph(hyperclique49, {1, 2, 4})
    assert Gi.V == {1, 2, 4}
    assert len(Gi.E) == 3
    assert Edge({1, 2}) in Gi.E
    assert Edge({1, 4}) in Gi.E
    assert Edge({2, 4}) in Gi.E
    assert Edge({4, 2}) in Gi.E


def test_find_hypercliques(hyperclique49):
    cliques = find_hypercliques(hyperclique49)
    assert 4 == len(cliques)
    assert Edge({0, 1, 2}) in cliques
    assert Edge({1, 3, 4}) in cliques
    assert Edge({1, 2, 4}) in cliques
    assert Edge({2, 4, 5}) in cliques


def test_reduce_graph(irreducible):
    G1, G2 = reduce_graph(irreducible)

    assert len(G1.E) > 0
    assert len(G2.E) > 0

    assert G1.V.issubset(irreducible.V) and len(G1.V) < len(irreducible.V)
    assert G1.E.issubset(irreducible.E) and len(G1.E) < len(irreducible.E)

    assert G2.V == irreducible.V
    assert G2.E.issubset(irreducible.E) and len(G2.E) == len(irreducible.E) - 1

    assert len(G1.E.difference(G2.E)) == 1
    assert G2.E.union(G1.E) == irreducible.E


def test_gen_k_ary_ind_from_cliques():
    karies = gen_k_ary_ind_from_cliques(3, [{4, 5, 6, 7}])
    assert len(karies) == 4
    assert Edge({4, 5, 6}) in karies
    assert Edge({4, 5, 7}) in karies
    assert Edge({4, 6, 7}) in karies
    assert Edge({5, 6, 7}) in karies


def test_gen_sub_inds():
    pass


def test_find_hypercliques_irreducible(irreducible):
    cliques = find_hypercliques(irreducible)
    assert 27 == len(cliques)
    assert Edge({1, 3, 8}) in cliques
    assert Edge({0, 7, 4}) in cliques
    assert Edge({7, 4, 0}) in cliques
    assert Edge({1, 0, 8}) not in cliques
