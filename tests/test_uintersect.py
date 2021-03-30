from matchbox.uintersect import UIntersectFinder


def test_one_table(table1):
    uind_finder = UIntersectFinder()
    uind_finder.add('L', table1)
    # In this case, none is contained in any other
    uinds = uind_finder()
    assert len(uinds) == 0


def test_self(table1):
    uind_finder = UIntersectFinder()

    uind_finder.add('L', table1)
    uind_finder.add('R', table1)

    # In this case, none is contained in any other
    uinds = uind_finder()

    # Note that in this case we are using the same table twice! So
    # L.X ⊆ R.X and R.X ⊆ L.X
    assert len(uinds) == 6
    for ind in uinds:
        assert ind.arity == 1


def test_self_ks(table1):
    uind_finder = UIntersectFinder(method='ks')

    uind_finder.add('L', table1)
    uind_finder.add('R', table1)

    # In this case, none is contained in any other
    uinds = uind_finder()

    # Note that in this case we are using the same table twice! So
    # L.X ⊆ R.X and R.X ⊆ L.X
    assert len(uinds) == 6
    for ind in uinds:
        assert ind.arity == 1


def test_different_dist(table1, table2):
    uind_finder = UIntersectFinder()

    uind_finder.add('L', table1)
    uind_finder.add('R', table2)

    uinds = uind_finder()

    assert len(uinds) == 4
    for ind in uinds:
        assert ind.arity == 1
