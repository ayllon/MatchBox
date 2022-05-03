from functools import reduce
from operator import xor
from typing import Union, Iterable

import numpy as np
import pandas


class AttributeSet(object):
    """
    Wraps a data set and one of its column into an hashable object that can be used
    in sets, dictionaries, etc.

    Parameters
    ----------
    relation_name : str
        Relation name
    attr_names : tuple
        Tuple (or iterable, or single string) of attribute names
    dataset : object (Optional)
        For convenience, a reference to the actual dataset.
    """

    def __init__(self, relation_name: str, attr_names: Iterable, dataset: pandas.DataFrame = None):
        self.__relation_name = relation_name
        if isinstance(attr_names, tuple):
            self.__attr_names = attr_names
        elif isinstance(attr_names, str):
            self.__attr_names = (attr_names,)
        elif hasattr(attr_names, '__iter__'):
            self.__attr_names = tuple(attr_names)
        else:
            self.__attr_names = (attr_names,)
        self.__relation = dataset
        self._hash = hash(self.__relation_name) ^ reduce(xor, map(hash, self.__attr_names))

    @property
    def relation_name(self) -> str:
        return self.__relation_name

    @property
    def attr_names(self) -> tuple:
        return self.__attr_names

    def add_attributes(self, attrs: Union[Iterable[str], str]):
        if not hasattr(attrs, '__iter__'):
            attrs = (attrs,)
        self.__attr_names += attrs

    @property
    def relation(self) -> pandas.DataFrame:
        return self.__relation

    @property
    def data(self) -> np.ndarray:
        return self.__relation[list(self.__attr_names)]

    @property
    def size(self) -> int:
        return len(self.__relation)

    def __len__(self):
        """
        Returns
        -------
        out : int
            Number of attributes in the attribute set
        """
        return len(self.__attr_names)

    def __getitem__(self, item) -> 'AttributeSet':
        """
        Get a subset of the attribute set
        Parameters
        ----------
        item : int, or slice
            Attribute positions
        Returns
        -------
        out : AttributeSet
            This method returns a new AttributeSet, even when item is a slice
        """
        return AttributeSet(self.__relation_name, self.__attr_names[item], self.__relation)

    def __add__(self, other) -> 'AttributeSet':
        """
        Create a new AttributeSet adding the attribute names from other

        Parameters
        ----------
        other : AttributeSet
            Its relation must be the same as the relation of self.
        Returns
        -------
        out : AttributeSet
            A new attribute set with the attributes of self, plus the attributes of other
        """
        assert isinstance(other, AttributeSet)
        assert other.__relation_name == self.__relation_name
        return AttributeSet(self.__relation_name, self.__attr_names + other.__attr_names, self.__relation)

    def __hash__(self) -> int:
        return self._hash

    def __str__(self) -> str:
        return f'{self.__relation_name}[{", ".join(map(str, self.__attr_names))}]'

    def __repr__(self) -> str:
        return str(self)

    def __eq__(self, other: 'AttributeSet') -> bool:
        """
        Two attribute sets are equal if the relation and the attribute list are equal
        """
        return self._hash == other._hash and \
               self.__relation_name == other.relation_name and \
               self.__attr_names == other.attr_names

    def __lt__(self, other: 'AttributeSet') -> bool:
        """
        One attribute set is less than another if their lexicographical representation is less
        """
        return self.__relation_name < other.relation_name or \
               (self.__relation_name == other.relation_name and self.__attr_names < other.attr_names)

    @property
    def has_duplicates(self) -> bool:
        """
        Returns
        -------
        bool : True if there are duplicate attribute names
        """
        return len(self.__attr_names) != len(set(self.__attr_names))
