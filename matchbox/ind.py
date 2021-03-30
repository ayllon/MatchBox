import itertools
from typing import Set

import numpy as np
from scipy.spatial.ckdtree import cKDTree as KDTree

from matchbox import AttributeSet
from matchbox.util import combine_hash


class Ind(object):
    """
    Intersection Dependency

    Parameters
    ----------
    lhs : AttributeSet
        Left hand-side of the Intersection Dependency
    rhs : AttributeSet
        Right hand-side of the Intersection dependency
    """

    def __init__(self, lhs: AttributeSet, rhs: AttributeSet, confidence: float = np.nan):
        self.lhs = lhs
        self.rhs = rhs
        self.confidence = confidence
        self._arity = len(self.lhs)
        # Make sure the hash is the same for R::[A, B] ⊆ S::[E, G] and for R::[B, A] ⊆ S::[G, E]
        self._hash = combine_hash(hash(self.lhs.relation_name), hash(self.rhs.relation_name))
        self._lhs_attrs = []
        self._rhs_attrs = []
        for i in np.argsort(self.lhs.attr_names):
            self._lhs_attrs.append(self.lhs.attr_names[i])
            self._rhs_attrs.append(self.rhs.attr_names[i])
            pair_hash = combine_hash(hash(self.lhs.attr_names[i]), hash(self.rhs.attr_names[i]))
            self._hash = combine_hash(self._hash, pair_hash)

    @property
    def arity(self) -> int:
        """
        Returns
        -------
        int : the arity of this inclusion dependency
        """
        return self._arity

    def generalizations(self) -> Set['Ind']:
        """
        Computes a set of Intersection Dependencies that generalize this one.
        i.e. R::[A] ⊆ S::[E] and R::[B] ⊆ S::[G] generalize R::[A, B] ⊆ S::[E, G]
        Returns
        -------
        out : set
            A set of Intersection Dependencies
        """
        s = set()
        for i in range(1, self.arity):
            lhs_com = itertools.combinations(self.lhs.attr_names, i)
            rhs_com = itertools.combinations(self.rhs.attr_names, i)
            for lhs_attr, rhs_attr in zip(lhs_com, rhs_com):
                s.add(Ind(
                    lhs=AttributeSet(self.lhs.relation_name, list(lhs_attr), self.lhs.relation),
                    rhs=AttributeSet(self.rhs.relation_name, list(rhs_attr), self.rhs.relation),
                ))
        return s

    def get_all_unary(self) -> Set['Ind']:
        """
        Returns
        -------
        out : set
            All unary inclusion dependencies that generalize this one
        """
        result = set()
        for l, r in zip(self.lhs.attr_names, self.rhs.attr_names):
            result.add(Ind(
                lhs=AttributeSet(self.lhs.relation_name, [l], self.lhs.relation),
                rhs=AttributeSet(self.rhs.relation_name, [r], self.rhs.relation)
            ))
        return result

    def __hash__(self) -> int:
        return self._hash

    def __str__(self) -> str:
        return ', '.join(map(' ⊆ '.join, zip(map(str, self.lhs.attr_names), map(str, self.rhs.attr_names))))# + f' {self.confidence:.2f}'

    def __repr__(self) -> str:
        return self.__str__()

    def __eq__(self, other: 'Ind') -> bool:
        """
        Two Ind are equal if both LHS and RHS are equal
        """
        return self._hash == other._hash \
               and self.lhs.relation_name == other.lhs.relation_name \
               and self.rhs.relation_name == other.rhs.relation_name \
               and self._lhs_attrs == other._lhs_attrs \
               and self._rhs_attrs == other._rhs_attrs

    def __lt__(self, other: 'Ind') -> bool:
        """
        We define the "less than" lexicographically for convenience, so results
        can be sorted and printed with similar Ind close by
        """
        return self.lhs < other.lhs or (self.lhs == other.lhs and self.rhs < other.rhs)

    def join(self, n=None):
        tree = KDTree(self.rhs.data)
        sample = self.lhs.relation.sample(n=n)
        dist, idx = tree.query(sample[list(self.lhs.attr_names)])
        return dist, sample.join(self.rhs.relation, on=idx, lsuffix='_lhs', rsuffix='_rhs')
