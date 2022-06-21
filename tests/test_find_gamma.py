import pytest

from matchbox.find_gamma import *


@pytest.fixture
def hyperclique_simple():
    return Graph(
        V={1, 2, 3, 4},
        E=set(map(Edge, [
            {1, 2, 3}, {1, 3, 4}
        ]))
    )


def test_is_quasi_clique_equiv(hyperclique48):
    """
    gamma = 1., so it must be strictly equivalent to is_clique
    """
    assert is_quasi_clique(hyperclique48, {1, 2, 5}, 1., 1.)
    assert is_quasi_clique(hyperclique48, {3, 4, 5}, 1., 1.)
    assert is_quasi_clique(hyperclique48, {1, 2, 3, 4}, 1., 1.)
    assert not is_quasi_clique(hyperclique48, {1, 2, 3, 4, 5}, 1., 1.)


def test_is_quasi_clique(hyperclique48):
    """
    gamma = 0.6
    """
    # Hypercliques are also quasi cliques
    assert is_quasi_clique(hyperclique48, {1, 2, 5}, Lambda=.6, gamma=.5)
    assert is_quasi_clique(hyperclique48, {3, 4, 5}, Lambda=.6, gamma=.5)
    assert is_quasi_clique(hyperclique48, {1, 2, 3, 4}, Lambda=.6, gamma=.5)
    # We have 6 edges, comb(5,3) = 10, so gamma 0.6 must pass
    assert not is_quasi_clique(hyperclique48, {1, 2, 3, 4, 5}, Lambda=0., gamma=0.8)
    assert is_quasi_clique(hyperclique48, {1, 2, 3, 4, 5}, Lambda=0., gamma=0.3)
    # H0: every edge is equally likely to be missing. A lower lambd means
    #     the probability of rejecting H0 is lower
    assert not is_quasi_clique(hyperclique48, {1, 2, 3, 4, 5}, Lambda=0.1, gamma=0.)
    assert is_quasi_clique(hyperclique48, {1, 2, 3, 4, 5}, Lambda=0.01, gamma=0.)


def test_find_hypercliques(hyperclique48):
    """
    gamma = 1., so it must be strictly equivalent to is_clique
    """
    cliques = list(find_quasicliques(hyperclique48, lambd=0., gamma=1., grow=False))
    assert 3 == len(cliques)
    assert Edge({1, 2, 5}) in cliques
    assert Edge({3, 4, 5}) in cliques
    assert Edge({1, 2, 3, 4}) in cliques


def test_grow_quasi(hyperclique48):
    """
    Test only the grow_clique function, which must take a known clique
    and grow applying Uno algorithm
    """
    cliques = grow_clique(hyperclique48, {1, 2}, gamma=0.6, Lambda=0.01)
    assert {1, 2, 3, 4} in cliques

    cliques = grow_clique(hyperclique48, {2, 3}, gamma=0.6, Lambda=0.01)
    assert {1, 2, 3, 4} in cliques

    cliques = grow_clique(hyperclique48, {3, 4}, gamma=0.6, Lambda=0.01)
    assert {1, 2, 3, 4} in cliques

    cliques = grow_clique(hyperclique48, {4, 1}, gamma=0.6, Lambda=0.01)
    assert {1, 2, 3, 4} in cliques


def test_find_quasi_grow(hyperclique48):
    """
    The modified HYPERCLIQUE is not able to find a 0.6/0.6 quasiclique
    since there are no edges connecting two nodes connected to all other nodes.
    Applying the modified Uno's algorithm must grow the first candidate and find the full quasiclique
    """
    cliques = list(find_quasicliques(hyperclique48, lambd=0.01, gamma=0.6, grow=True))
    assert 1 == len(cliques)
    assert Edge({1, 2, 3, 4, 5}) in cliques
