import pytest

from matchbox.mind import Mind
from matchbox.uintersect import UIntersectFinder


def test_mixing_arities(ind, ind2, ind3):
    mind = Mind()
    with pytest.raises(ValueError):
        mind(ind=set([ind, ind2, ind3]))


def test_self(table1):
    # Build fron the uintersect
    # This is like test_self in test_uintersect.py
    uind_finder = UIntersectFinder()

    uind_finder.add('L', table1)
    uind_finder.add('R', table1)
    uinds = uind_finder()
    assert len(uinds) == 6

    # Get the 2-ind candidates
    # All candidates are good!
    mind = Mind()
    ind2 = mind(uinds, stop=2)
    assert len(ind2) == 6

    # Build 3-ind
    ind3 = mind(ind2, stop=3)
    assert len(ind3) == 2

    # Check the actual results
    expected = {
        "L::(V, X, Y) ⊆ R::(V, X, Y)",
        "R::(V, X, Y) ⊆ L::(V, X, Y)"
    }
    got = set()
    for i in ind3:
        print(str(i))
        string = str(i)
        assert string in expected
        got.add(string)
    for i in expected:
        assert i in got
