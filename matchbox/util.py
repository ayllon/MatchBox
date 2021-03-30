import pandas
from typing import Callable, Union


class NoopListener(object):
    """
    Do-nothing listener
    """

    def __init__(self, container):
        self.__container = container

    def __iter__(self):
        return iter(self.__container)


def longest_prefix(strings):
    """
    Return the longest common prefix for the given set of strings
    """
    min_len = min(map(len, strings))
    i = 0
    while i < min_len:
        c = strings[0][i]
        if not all(map(lambda s: s[i] == c, strings)):
            break
        i += 1
    return strings[0][:i]


class CallCounter(object):
    """
    Utility to keep track of how many times a given call is used
    """

    def __init__(self, test: Callable):
        self.__test = test
        self.counter = 0

    def __call__(self, *args, **kwargs):
        self.counter += 1
        return self.__test(*args, **kwargs)


def sample(lhs_data: pandas.DataFrame, rhs_data: pandas.DataFrame, samples: Union[float, int], replace: bool):
    """
    Get a sample from both LHS and RHS
    """
    if isinstance(samples, float):
        lhs_samples = int(len(lhs_data) * samples)
        rhs_samples = int(len(rhs_data) * samples)
    elif isinstance(samples, int):
        lhs_samples = rhs_samples = samples
    else:
        lhs_samples, rhs_samples = samples
    lhs_data = lhs_data.sample(lhs_samples, replace=replace)
    rhs_data = rhs_data.sample(rhs_samples, replace=replace)
    return lhs_data, rhs_data


def combine_hash(lhash: int, rhash: int) -> int:
    """
    As boost::hash_combine
    """
    lhash ^= rhash + 0x9e3779b9 + (lhash << 6) + (rhash >> 2)
    return lhash
