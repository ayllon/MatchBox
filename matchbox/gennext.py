import itertools
from typing import FrozenSet, Set

from .ind import Ind


def gen_next(ind: Set) -> FrozenSet[Ind]:
    """
    From De Marchi 2003

    Parameters
    ----------
    ind : list of sets
        Inclusion dependencies of size i

    Returns
    -------
    out : list of sets of cardinality i + 1
    """
    temp_results = list()

    # Empty input, noop
    if not len(ind):
        return frozenset()

    # Populate
    arity = next(iter(ind)).arity
    for p, q in itertools.product(ind, ind):
        if p.lhs.relation_name == q.lhs.relation_name and p.rhs.relation_name == q.rhs.relation_name:
            match = True
            for i in range(arity - 1):
                match &= p.lhs.attr_names[i] == q.lhs.attr_names[i]
                match &= p.rhs.attr_names[i] == q.rhs.attr_names[i]
            if match and p.lhs.attr_names[-1] < q.lhs.attr_names[-1] and p.rhs.attr_names[-1] != q.rhs.attr_names[-1]:
                temp_results.append(Ind(p.lhs + q.lhs[-1], p.rhs + q.rhs[-1]))

    # Prune
    results = set()
    for i in temp_results:
        use = True
        for J in i.generalizations():
            if J.arity == arity and J not in ind:
                use = False
                break
        if use:
            results.add(i)
    return frozenset(results)
