import pytest

from matchbox.attributeset import AttributeSet
from matchbox.gennext import gen_next
from matchbox.ind import Ind


@pytest.fixture
def uind():
    return {
        Ind(AttributeSet('R', 'A'), AttributeSet('S', 'A')),
        Ind(AttributeSet('R', 'B'), AttributeSet('S', 'B')),
        Ind(AttributeSet('R', 'C'), AttributeSet('S', 'C')),
    }


def test_gen2(uind):
    """
    Test generation of 2-ind
    """
    ind2 = gen_next(uind)
    assert len(ind2) == 3
    for i in ind2:
        assert i.arity == 2


def test_gen3(uind):
    """
    Test generation of 3-ind from the previous 2-ind
    """
    ind2 = gen_next(uind)
    ind3 = gen_next(ind2)
    assert len(ind3) == 1
    for i in ind3:
        assert i.arity == 3


def test_prune():
    """
    Test taken from the example from De Marchi et al.
    R::[A, B, C] ⊆ S::[E, F, G] will be generated, but it must be pruned as
    R::[B, C] ⊆ S::[F, G] is not satisfied.
    """
    ind2 = {
        Ind(AttributeSet('R', ['A', 'B']), AttributeSet('S', ['E', 'F'])),
        Ind(AttributeSet('R', ['A', 'C']), AttributeSet('S', ['E', 'G'])),
    }
    ind3 = gen_next(ind2)
    assert len(ind3) == 0


def test_prune2():
    """
    Test taken from the example from De Marchi et al.
    In this case, R::[B, C] ⊆ S::[F, G] is satisfied.
    """
    ind2 = {
        Ind(AttributeSet('R', ['A', 'B']), AttributeSet('S', ['E', 'F'])),
        Ind(AttributeSet('R', ['A', 'C']), AttributeSet('S', ['E', 'G'])),
        Ind(AttributeSet('R', ['B', 'C']), AttributeSet('S', ['F', 'G'])),
    }
    ind3 = gen_next(ind2)
    assert len(ind3) == 1
    for i in ind3:
        assert i.arity == 3


def test_multiple():
    """
    From Table 2 of De Marchi et al.
    """
    ind1 = {
        # R to S
        Ind(AttributeSet('R', 'A'), AttributeSet('S', 'E')),
        Ind(AttributeSet('R', 'B'), AttributeSet('S', 'F')),
        Ind(AttributeSet('R', 'C'), AttributeSet('S', 'G')),
        # R to T
        Ind(AttributeSet('R', 'A'), AttributeSet('T', 'K')),
        Ind(AttributeSet('R', 'B'), AttributeSet('T', 'L')),
        Ind(AttributeSet('R', 'D'), AttributeSet('T', 'I')),
        Ind(AttributeSet('R', 'D'), AttributeSet('T', 'J')),
        # S to T
        Ind(AttributeSet('S', 'E'), AttributeSet('T', 'K')),
        Ind(AttributeSet('S', 'F'), AttributeSet('T', 'L')),
        Ind(AttributeSet('S', 'H'), AttributeSet('T', 'J')),
        # T to R
        Ind(AttributeSet('T', 'I'), AttributeSet('R', 'D')),
        # T to T
        Ind(AttributeSet('T', 'I'), AttributeSet('T', 'J')),
    }

    expected2 = {
        # R to S
        Ind(AttributeSet('R', ['A', 'B']), AttributeSet('S', ['E', 'F'])),
        Ind(AttributeSet('R', ['A', 'C']), AttributeSet('S', ['E', 'G'])),
        Ind(AttributeSet('R', ['B', 'C']), AttributeSet('S', ['F', 'G'])),
        # R to T
        Ind(AttributeSet('R', ['A', 'B']), AttributeSet('T', ['K', 'L'])),
        Ind(AttributeSet('R', ['A', 'D']), AttributeSet('T', ['K', 'I'])),
        Ind(AttributeSet('R', ['A', 'D']), AttributeSet('T', ['K', 'J'])),
        Ind(AttributeSet('R', ['B', 'D']), AttributeSet('T', ['L', 'I'])),
        Ind(AttributeSet('R', ['B', 'D']), AttributeSet('T', ['L', 'J'])),
        # S to T
        Ind(AttributeSet('S', ['E', 'F']), AttributeSet('T', ['K', 'L'])),
        Ind(AttributeSet('S', ['E', 'H']), AttributeSet('T', ['K', 'J'])),
        Ind(AttributeSet('S', ['F', 'H']), AttributeSet('T', ['L', 'J'])),
    }

    candidates2 = gen_next(ind1)
    assert len(expected2) == len(candidates2)
    assert expected2 == candidates2
