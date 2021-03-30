import pytest

from matchbox.attributeset import AttributeSet
from matchbox.ind import Ind


@pytest.fixture
def ind3_permuted():
    return Ind(
        lhs=AttributeSet('R', ['C', 'A', 'B']),
        rhs=AttributeSet('S', ['G', 'E', 'F'])
    )


def test_arity(ind, ind2, ind3):
    """
    Test the arity property
    """
    assert ind.arity == 1
    assert ind2.arity == 2
    assert ind3.arity == 3


def test_eq(ind3):
    ind_copy = Ind(lhs=ind3.lhs, rhs=ind3.rhs)
    assert hash(ind3) == hash(ind_copy)
    assert ind3 == ind_copy


def test_eq_permuted(ind3, ind3_permuted):
    assert hash(ind3) == hash(ind3_permuted)
    assert ind3 == ind3_permuted


def test_ind_in_set(ind):
    s = {Ind(lhs=ind.lhs, rhs=ind.rhs)}
    assert ind in s


def test_ind_permuted_in_set(ind3, ind3_permuted):
    s = {ind3}
    assert ind3_permuted in s


def test_ind_not_in_set(ind):
    s = {Ind(lhs=ind.rhs, rhs=ind.lhs)}
    assert ind not in s


def test_specializes(ind):
    """
    Unary intersections can not specialize anything
    """
    assert len(ind.generalizations()) == 0


def test_specializes2(ind2):
    """
    Test the generation of relations that this 2-Ind specializes
    """
    s = ind2.generalizations()
    assert len(s) == 2


def test_specializes(ind3):
    """
    Test the generation of relations that this 3-Ind specializes
    """
    specializes = ind3.generalizations()
    assert len(specializes) == 6


def test_order(ind, ind2, ind3):
    """
    Test ordering between ind
    """
    assert ind < ind2
    assert ind2 < ind3
    assert ind < ind3
